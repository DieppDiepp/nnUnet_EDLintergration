"""
🧠 EDL ENGINE (UPDATED)
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
        self._inject_custom_trainer() # <--- Bước quan trọng: Tiêm Trainer
        self.predictor = self._initialize_predictor()
        self.preprocessor = self.predictor.configuration_manager.preprocessor_class(verbose=False)

    def _inject_custom_trainer(self):
        """
        Copy file EDLTrainer.py từ src/trainers vào thư mục cài đặt của nnunetv2
        để hàm recursive_find_python_class có thể tìm thấy nó.
        """
        try:
            # 1. Tìm vị trí cài đặt nnunetv2 trên Colab
            nnunet_path = os.path.dirname(nnunetv2.__file__)
            target_folder = os.path.join(nnunet_path, "training", "nnUNetTrainer")
            
            # 2. Tìm file source trong src/trainers
            # Giả sử file engine này nằm ở src/edl_engine.py -> root/src/trainers
            current_dir = os.path.dirname(os.path.abspath(__file__))
            source_file = os.path.join(current_dir, "trainers", "EDLTrainer.py")
            
            if not os.path.exists(source_file):
                print(f"⚠️ Warning: Không tìm thấy file trainer tại {source_file}. Bỏ qua bước inject.")
                return

            # 3. Copy file
            target_file = os.path.join(target_folder, "EDLTrainer.py")
            print(f"💉 Injecting EDLTrainer...\\n   From: {source_file}\\n   To:   {target_file}")
            shutil.copy(source_file, target_file)
            print("✅ Inject thành công! nnU-Net sẽ nhận diện được EDLTrainer.")
            
        except Exception as e:
            print(f"❌ Lỗi khi inject trainer: {e}")

    def _initialize_predictor(self):
        print("🚀 Initializing nnU-Net Predictor...")
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

    def save_nifti(self, data, affine, output_path):
        """Lưu mảng numpy thành file .nii.gz"""
        # data shape: [X, Y, Z]
        img = nib.Nifti1Image(data.astype(np.float32), affine)
        nib.save(img, output_path)
        # print(f"    💾 Saved NIfTI: {os.path.basename(output_path)}")

    def process_case(self, case_id):
        # ... (Giữ nguyên logic cũ) ...
        print(f"\\n🔍 Processing: {case_id}...")
        img_folder = self.config["image_folder"]
        lbl_folder = self.config["label_folder"]
        
        base_file = os.path.join(img_folder, f"{case_id}_0000.nii")
        ext = ".nii" if os.path.exists(base_file) else ".nii.gz"
        
        image_files = [os.path.join(img_folder, f"{case_id}_{i:04d}{ext}") for i in range(4)]
        gt_file = os.path.join(lbl_folder, f"{case_id}{ext}")
        if not os.path.exists(gt_file): gt_file = None

        # --- LẤY AFFINE MATRIX GỐC ĐỂ LƯU NIFTI ---
        # Ta load tạm 1 ảnh gốc để lấy thông tin không gian (vị trí, hướng)
        tmp_img = nib.load(image_files[0])
        original_affine = tmp_img.affine

        # 2. Preprocessing
        data, seg, properties = self.preprocessor.run_case(
            image_files, gt_file, 
            self.predictor.plans_manager, 
            self.predictor.configuration_manager, 
            self.predictor.dataset_json
        )
        
        # 3. Inference
        data_tensor = torch.from_numpy(data).to(self.predictor.device)
        with torch.no_grad():
            pred_logits = self.predictor.predict_logits_from_preprocessed_data(data_tensor)
        
        # 4. EDL Calculation
        evidence = F.softplus(pred_logits)
        alpha = evidence + 1
        S = torch.sum(alpha, dim=0)
        K = alpha.shape[0]
        
        uncertainty = (K / S).cpu().numpy()
        segmentation = torch.argmax(pred_logits, dim=0).cpu().numpy()
        
        if seg is None: seg = np.zeros((1, *segmentation.shape))
        
        # --- LƯU FILE 3D NẾU CẦN ---
        if self.config.get("save_3d_nifti", False):
            # out_dir = os.path.join(self.config["output_folder"], "3d_nifti", case_id)
            nifti_folder_name = self.config.get("dir_nifti", "3d_nifti")
            out_dir = os.path.join(self.config["output_folder"], nifti_folder_name, case_id)

            os.makedirs(out_dir, exist_ok=True)
            
            # Lưu Uncertainty Map
            self.save_nifti(uncertainty, original_affine, os.path.join(out_dir, "uncertainty.nii.gz"))
            
            # Lưu Prediction
            self.save_nifti(segmentation, original_affine, os.path.join(out_dir, "prediction.nii.gz"))
            
            # Lưu Ground Truth (đã crop) để đối chiếu
            self.save_nifti(seg[0], original_affine, os.path.join(out_dir, "ground_truth.nii.gz"))
            
            # Lưu MRI nền (Channel 0 - T1/Flair)
            self.save_nifti(data[0], original_affine, os.path.join(out_dir, "mri_crop.nii.gz"))

        return data, seg, segmentation, uncertainty, properties