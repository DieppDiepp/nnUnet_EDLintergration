"""
🧠 EDL ENGINE (UPDATED V5 - ROBUST MERGE)
Kết hợp logic phân rã Uncertainty (Aleatoric/Epistemic) với khung code xử lý lỗi an toàn.
Tự động inject EDLTrainer vào hệ thống nnU-Net để tránh lỗi "Class not found".
"""
import os
import shutil
import torch
import torch.nn.functional as F
import numpy as np
import nibabel as nib
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
import nnunetv2

class EDLInferenceEngine:
    def __init__(self, config):
        self.config = config
        print(f"🔧 Initializing EDL Engine with config...")
        self._inject_custom_trainer() # <--- Bước quan trọng: Tiêm Trainer
        self.predictor = self._initialize_predictor()
        self.preprocessor = self.predictor.configuration_manager.preprocessor_class(verbose=False)

    def _inject_custom_trainer(self):
        """
        Copy file EDLTrainer.py từ src/trainers vào thư mục cài đặt của nnunetv2
        để hàm recursive_find_python_class có thể tìm thấy nó.
        """
        try:
            # 1. Tìm vị trí cài đặt nnunetv2
            nnunet_path = os.path.dirname(nnunetv2.__file__)
            target_folder = os.path.join(nnunet_path, "training", "nnUNetTrainer")
            
            # 2. Tìm file source trong src/trainers
            current_dir = os.path.dirname(os.path.abspath(__file__))
            source_file = os.path.join(current_dir, "trainers", "EDLTrainer.py")
            
            if not os.path.exists(source_file):
                print(f"⚠️ Warning: Không tìm thấy file trainer tại {source_file}. Bỏ qua bước inject.")
                return

            # 3. Copy file
            target_file = os.path.join(target_folder, "EDLTrainer.py")
            # print(f"💉 Injecting EDLTrainer...\\n   From: {source_file}\\n   To:   {target_file}")
            shutil.copy(source_file, target_file)
            print("✅ Inject thành công! nnU-Net sẽ nhận diện được EDLTrainer.")
            
        except Exception as e:
            print(f"❌ Lỗi khi inject trainer: {e}")

    def _initialize_predictor(self):
        print("🚀 Initializing nnU-Net Predictor...")
        try:
            predictor = nnUNetPredictor(
                tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
                device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
                verbose=False
            )
            
            ckpt_path = self.config["checkpoint_path"]
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"❌ Checkpoint not found: {ckpt_path}")
                
            checkpoint_folder = os.path.dirname(os.path.dirname(ckpt_path))
            predictor.initialize_from_trained_model_folder(
                checkpoint_folder, use_folds=(0,), checkpoint_name="checkpoint_best.pth"
            )
            print(f"📂 Model loaded from: {checkpoint_folder}")
            return predictor
        except Exception as e:
            print(f"❌ Critical Error initializing predictor: {e}")
            raise e

    def save_nifti(self, data, affine, output_path):
        """Lưu mảng numpy thành file .nii.gz"""
        try:
            # data shape: [X, Y, Z] -> Phải ép kiểu về float32 để tránh lỗi format
            img = nib.Nifti1Image(data.astype(np.float32), affine)
            nib.save(img, output_path)
        except Exception as e:
            print(f"⚠️ Error saving NIfTI {output_path}: {e}")

    def process_case(self, case_id):
        """
        Xử lý một ca bệnh: Preprocess -> Inference -> EDL Decomposition -> Save
        """
        print(f"\n🔍 Processing: {case_id}...")
        
        # --- 1. SETUP PATHS (Code cũ - Robust) ---
        img_folder = self.config["image_folder"]
        lbl_folder = self.config["label_folder"]
        
        base_file = os.path.join(img_folder, f"{case_id}_0000.nii")
        ext = ".nii" if os.path.exists(base_file) else ".nii.gz"
        
        image_files = [os.path.join(img_folder, f"{case_id}_{i:04d}{ext}") for i in range(4)]
        
        # Kiểm tra file input tồn tại không
        if not os.path.exists(image_files[0]):
            print(f"❌ Error: Input files for {case_id} not found.")
            return None, None, None, None, None

        gt_file = os.path.join(lbl_folder, f"{case_id}{ext}")
        if not os.path.exists(gt_file): 
            # print(f"⚠️ Ground truth for {case_id} not found (Inference only mode).")
            gt_file = None

        # --- LẤY AFFINE MATRIX GỐC ĐỂ LƯU NIFTI ---
        try:
            tmp_img = nib.load(image_files[0])
            original_affine = tmp_img.affine
        except Exception as e:
            print(f"❌ Error loading affine from input image: {e}")
            return None, None, None, None, None

        # --- 2. PREPROCESSING ---
        try:
            data, seg, properties = self.preprocessor.run_case(
                image_files, gt_file, 
                self.predictor.plans_manager, 
                self.predictor.configuration_manager, 
                self.predictor.dataset_json
            )
        except Exception as e:
            print(f"❌ Error during preprocessing: {e}")
            return None, None, None, None, None
        
        # --- 3. INFERENCE ---
        # Thêm torch.no_grad() để tiết kiệm bộ nhớ (từ code mới)
        data_tensor = torch.from_numpy(data).to(self.predictor.device)
        with torch.no_grad():
            pred_logits = self.predictor.predict_logits_from_preprocessed_data(data_tensor)
        
        # ======================================================================
        # 4. EDL UNCERTAINTY DECOMPOSITION (LOGIC MỚI)
        # ======================================================================
        # a. Tính tham số Dirichlet
        evidence = F.softplus(pred_logits)
        alpha = evidence + 1
        S = torch.sum(alpha, dim=0, keepdim=True) # Sum strength
        probs = alpha / S                         # Expected Probability
        
        # b. Total Uncertainty (Entropy of Expected Probabilities)
        # H(p) = - sum(p * log(p))
        # Cộng thêm 1e-7 để tránh log(0)
        total_unc = -torch.sum(probs * torch.log(probs + 1e-7), dim=0)
        
        # c. Aleatoric Uncertainty (Expected Entropy of Dirichlet)
        # E[H(p)] approx sum(p * (digamma(S+1) - digamma(alpha+1)))
        digamma_S = torch.digamma(S + 1)
        digamma_alpha = torch.digamma(alpha + 1)
        aleatoric_unc = torch.sum(probs * (digamma_S - digamma_alpha), dim=0)
        
        # d. Epistemic Uncertainty (Mutual Information)
        # I = H(p) - E[H(p)]
        epistemic_unc = total_unc - aleatoric_unc
        
        # e. Chuẩn hóa về Numpy & Dictionary
        # Clamp để tránh số âm nhỏ do sai số tính toán float
        unc_dict = {
            "total": torch.clamp(total_unc, min=0).cpu().numpy(),
            "aleatoric": torch.clamp(aleatoric_unc, min=0).cpu().numpy(),
            "epistemic": torch.clamp(epistemic_unc, min=0).cpu().numpy()
        }
        
        segmentation = torch.argmax(pred_logits, dim=0).cpu().numpy()
        
        # Xử lý seg nếu không có GT
        if seg is None: seg = np.zeros((1, *segmentation.shape))
        
        # ======================================================================
        
        # --- 5. LƯU FILE 3D (LOGIC CŨ + FILE MỚI) ---
        if self.config.get("save_3d_nifti", False):
            try:
                nifti_folder_name = self.config.get("dir_nifti", "3d_nifti")
                out_dir = os.path.join(self.config["output_folder"], nifti_folder_name, case_id)
                os.makedirs(out_dir, exist_ok=True)
                
                # Lưu bộ 3 file Uncertainty (Mới)
                self.save_nifti(unc_dict["total"], original_affine, os.path.join(out_dir, "unc_total.nii.gz"))
                self.save_nifti(unc_dict["aleatoric"], original_affine, os.path.join(out_dir, "unc_aleatoric.nii.gz"))
                self.save_nifti(unc_dict["epistemic"], original_affine, os.path.join(out_dir, "unc_epistemic.nii.gz"))
                
                # Lưu Prediction (Cũ)
                self.save_nifti(segmentation, original_affine, os.path.join(out_dir, "prediction.nii.gz"))
                
                # Lưu Ground Truth (Cũ)
                self.save_nifti(seg[0], original_affine, os.path.join(out_dir, "ground_truth.nii.gz"))
                
                # Lưu MRI nền (Cũ)
                self.save_nifti(data[0], original_affine, os.path.join(out_dir, "mri_crop.nii.gz"))
                
            except Exception as e:
                print(f"⚠️ Error saving NIfTI files for {case_id}: {e}")

        # Return dict uncertainty thay vì 1 biến đơn lẻ để tương thích với logic mới
        return data, seg, segmentation, unc_dict, properties