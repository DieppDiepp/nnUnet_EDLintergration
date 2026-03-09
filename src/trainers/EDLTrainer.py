import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper

class EDLLoss(nn.Module):
    def __init__(self, num_classes, annealing_step=10, lamb=1.0):
        super(EDLLoss, self).__init__()
        self.num_classes = num_classes
        self.annealing_step = annealing_step
        self.lamb = lamb
        self.current_epoch = 0 
        self.dice_loss = MemoryEfficientSoftDiceLoss(batch_dice=True, do_bg=False, smooth=1e-5, ddp=False)

    def KL(self, alpha):
        beta = torch.ones((1, self.num_classes) + alpha.shape[2:]).to(alpha.device)
        S_alpha = torch.sum(alpha, dim=1, keepdim=True)
        S_beta = torch.sum(beta, dim=1, keepdim=True)
        
        lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
        lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
        
        dg0 = torch.digamma(S_alpha)
        dg1 = torch.digamma(alpha)
        
        kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
        return kl

    def forward(self, outputs, target):
        if target.dim() == outputs.dim(): 
            target = target.squeeze(1) 
            
        target_one_hot = F.one_hot(target.long(), num_classes=self.num_classes)
        target_one_hot = target_one_hot.permute(0, 4, 1, 2, 3).contiguous().type_as(outputs)

        evidence = F.softplus(outputs) 
        alpha = evidence + 1 
        S = torch.sum(alpha, dim=1, keepdim=True) 
        
        edl_loss = torch.sum(target_one_hot * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)
        edl_loss = torch.mean(edl_loss) 

        annealing_coef = min(1.0, self.current_epoch / self.annealing_step)
        
        kl_alpha = (alpha - 1) * (1 - target_one_hot) + 1
        kl_div = self.KL(kl_alpha)
        kl_loss = annealing_coef * torch.mean(kl_div)
        
        p = alpha / S   
        loss_dice = self.dice_loss(p, target.unsqueeze(1))
        
        final_loss = edl_loss + kl_loss + self.lamb * loss_dice
        return final_loss


class EDLTrainer(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, 
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        
        # --- CẤU HÌNH SỐ EPOCH ---
        # Ưu tiên đọc từ biến môi trường nnUNet_EPOCHS
        env_epochs = os.environ.get('nnUNet_EPOCHS')
        
        if env_epochs is not None:
            self.num_epochs = int(env_epochs)
        else:
            self.num_epochs = 50

        # Lưu checkpoint mỗi 50 epoch (mặc định của nnU-Net)
        self.save_every = 50

    def _build_loss(self):
        num_classes = self.label_manager.num_segmentation_heads
        anneal_step = min(100, self.num_epochs)

        # Khởi tạo EDLLoss
        loss = EDLLoss(num_classes=num_classes, annealing_step=anneal_step, lamb=1.0)

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0
            weights = weights / weights.sum()
            return DeepSupervisionWrapper(loss, weights)
        
        return loss

    def train_step(self, batch: dict):
        data = batch['data']
        target = batch['target']
        
        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)
            
        self.optimizer.zero_grad()
        
        if isinstance(self.loss, DeepSupervisionWrapper):
            self.loss.loss.current_epoch = self.current_epoch
        else:
            self.loss.current_epoch = self.current_epoch
            
        with torch.autocast(device_type=self.device.type, enabled=True):
            output = self.network(data)
            l = self.loss(output, target)
            
        self.grad_scaler.scale(l).backward()
        self.grad_scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()
        
        return {'loss': l.detach().cpu().numpy()}
    
class EDLTrainer_250epochs(EDLTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, 
                device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        
        # Chốt cứng 250 epoch, không cần truyền biến môi trường nnUNet_EPOCHS nữa
        self.num_epochs = 250
        
        # Lưu checkpoint mỗi 50 epoch
        self.save_every = 50