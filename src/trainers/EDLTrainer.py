import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper

# --- PHẦN 1: HÀM LOSS EDL CHUẨN HOÁ ---
class EDLLoss(nn.Module):
    def __init__(self, num_classes, annealing_step=10, lamb=1.0):
        super(EDLLoss, self).__init__()
        self.num_classes = num_classes
        self.annealing_step = annealing_step # annealing_step=10: Đây là tham số "ủ nhiệt". Trong 10 epoch đầu tiên, ta sẽ không phạt nặng model về độ bất định (KL Divergence). Lý do là lúc đầu model chưa học được gì, nếu ép nó phải "biết mình không biết" ngay thì nó sẽ bị rối và không học được các đặc trưng ảnh.
        self.lamb = lamb
        
        # Biến này sẽ được update từ Trainer mỗi epoch
        self.current_epoch = 0 # Biến này cực kỳ quan trọng. Nó được cập nhật liên tục từ Trainer để hàm Loss biết đang ở giai đoạn nào của quá trình huấn luyện.
        
        # Dice Loss: tắt do_bg (nếu cần), batch_dice=True để ổn định
        self.dice_loss = MemoryEfficientSoftDiceLoss(batch_dice=True, do_bg=False, smooth=1e-5, ddp=False)

    def KL(self, alpha):
        # KL Divergence: Dirichlet(alpha) || Dirichlet(1)
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
        """
        LƯU Ý: Không truyền current_epoch vào argument của forward 
        để tránh xung đột với DeepSupervisionWrapper.
        Ta dùng self.current_epoch đã được gán từ bên ngoài.
        """
        
        # 1. One-hot encoding an toàn cho nnU-Net (Target: [B, 1, X, Y, Z])
        # Chuyển target sang one-hot: [B, C, X, Y, Z]
        if target.dim() == outputs.dim(): # Nếu target đã có dim channel (thường là 1)
            target = target.squeeze(1) # Bỏ dim channel tạm thời để dùng one_hot
            
        # Tạo one-hot
        target_one_hot = F.one_hot(target.long(), num_classes=self.num_classes)
        # Permute để đưa channel về đúng vị trí số 2 (index 1): [B, X, Y, Z, C] -> [B, C, X, Y, Z]
        # Shape cũ: [Batch, X, Y, Z, Channel]  Thứ tự index là: (0, 1, 2, 3, 4)
        # Shape mong muốn: [Batch, Channel, X, Y, Z]
        target_one_hot = target_one_hot.permute(0, 4, 1, 2, 3).contiguous().type_as(outputs) # Đưa nhãn gốc (ví dụ: lớp 0, 1, 2) về dạng vector One-hot (ví dụ: [1,0,0], [0,1,0]). Điều này cần thiết để nhân ma trận trong công thức Bayes Risk.

        # 2. Tính Evidence và Alpha
        evidence = F.softplus(outputs) # Hàm Softplus ln(1+e^x) biến đổi Logits (âm/dương) thành Evidence (số dương). 
        # Tại sao không dùng ReLU? ReLU bị chết ở miền âm (bằng 0). Softplus mượt hơn và luôn dương, giúp gradient lan truyền tốt hơn.
        # Tại sao không dùng Softmax? Softmax ép tổng bằng 1 (mất thông tin về độ lớn của bằng chứng). Softplus giữ nguyên độ lớn: Bằng chứng càng nhiều, giá trị càng to.
        
        alpha = evidence + 1 # Đây là tham số alpha của phân phối Dirichlet.
        # Nếu Evidence = 0 (không biết gì) alpha = 1 (đại diện nghiệm đồng nhất, Phân phối phẳng/Uniform - Bất định tối đa).
        # Nếu Evidence cao alpha cao Xác suất tập trung (Tự tin).

        S = torch.sum(alpha, dim=1, keepdim=True) # Tổng sức mạnh bằng chứng của tất cả các lớp.
        
        # 3. Tính Bayes Risk (Cross Entropy cải biên cho Dirichlet)
        # Dùng Digamma thay vì Log để đúng toán học hơn (Digamma là đạo hàm bậc nhất của hàm log-gamma). Nó đóng vai trò tương tự như hàm log trong CrossEntropy nhưng hoạt động trên các tham số của phân phối xác suất.
        # Loss = Sum( y * (digamma(S) - digamma(alpha)) )
        edl_loss = torch.sum(target_one_hot * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)
        edl_loss = torch.mean(edl_loss) # Đây là hàm mất mát chính để model học phân loại đúng sai. Nó dựa trên công thức kỳ vọng của hàm log-likelihood trên phân phối Dirichlet.
        # Nó ép alpha của lớp đúng phải thật lớn, và alpha của các lớp sai phải nhỏ.

        # 4. KL Divergence (Regularization)
        # Annealing: Tăng dần trọng số KL
        annealing_coef = min(1.0, self.current_epoch / self.annealing_step)
        # Mục tiêu: Phạt model nếu nó quá tự tin vào những dự đoán sai.
                
        # Chỉ tính KL cho các pixel không phải là Ground Truth (để ép evidence -> 0 ở chỗ sai)
        # Theo paper gốc: KL_loss = KL(alpha || 1)
        kl_alpha = (alpha - 1) * (1 - target_one_hot) + 1
        kl_div = self.KL(kl_alpha)
        kl_loss = annealing_coef * torch.mean(kl_div)
        # Cơ chế: Tính khoảng cách (KL) giữa phân phối dự đoán và phân phối Uniform (Flat).
        # Logic: Với các mẫu khó hoặc nhiễu (không khớp Ground Truth), ta muốn model quay về trạng thái Uniform ("Tôi không biết" - Uncertainty cao) thay vì cố đoán bừa.

        
        # 5. Dice Loss
        # Expected probability: p = alpha / S
        p = alpha / S   
        # Dice loss của nnU-Net cần shape target gốc [B, 1, X, Y, Z]
        loss_dice = self.dice_loss(p, target.unsqueeze(1))
        
        # 6. Tổng hợp
        final_loss = edl_loss + kl_loss + self.lamb * loss_dice
        return final_loss

# --- PHẦN 2: TRAINER ---
class EDLTrainer(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, 
                 device: torch.device = torch.device('cuda')):
        # FIX: Bỏ unpack_dataset khi gọi super().__init__ vì nnU-Net v2 mới đã bỏ tham số này
        super().__init__(plans, configuration, fold, dataset_json, device=device)
        
        # Có thể tăng epoch nếu cần thiết
        # self.num_epochs = 1000

    def _build_loss(self):
        num_classes = self.label_manager.num_segmentation_heads
        
        # Khởi tạo EDLLoss
        loss = EDLLoss(num_classes=num_classes, annealing_step=50, lamb=1.0)
        
        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0
            weights = weights / weights.sum()
            # Wrap loss
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
        
        # --- UPDATE EPOCH CHO LOSS ---
        # Đây là bước quan trọng để xử lý Wrapper
        if isinstance(self.loss, DeepSupervisionWrapper):
            # Nếu bị wrap, biến loss thực sự nằm ở self.loss.loss
            self.loss.loss.current_epoch = self.current_epoch
        else:
            self.loss.current_epoch = self.current_epoch
            
        with torch.autocast(device_type=self.device.type, enabled=True):
            output = self.network(data)
            
            # Gọi hàm loss bình thường (không truyền current_epoch vào đây nữa)
            l = self.loss(output, target)
            
        self.grad_scaler.scale(l).backward()
        self.grad_scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()
        
        return {'loss': l.detach().cpu().numpy()}