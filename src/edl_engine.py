"""
🧠 EDL ENGINE (FINAL ROBUST V7)
Hỗ trợ chạy Inference cho cả 2 chế độ:
1. EDL Model: Tính toán Uncertainty Decomposition (Aleatoric/Epistemic).
2. Baseline Model: Chỉ chạy Segmentation chuẩn (nhanh hơn).

Đặc điểm:
- Tự động inject EDLTrainer để tránh lỗi class.
- Xử lý lỗi (Error Handling) chặt chẽ, không crash khi thiếu file.
- Comment chi tiết để dễ hiểu logic toán học.
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
        mode = config.get('model_mode', 'edl').upper()
        print(f"🔧 Initializing Engine | Mode: {mode}...")
        
        self._inject_custom_trainer() # <--- Bước quan trọng: Tiêm Trainer
        self.predictor = self._initialize_predictor()
        
        # Preprocessor dùng để crop và chuẩn hóa dữ liệu đầu vào
        self.preprocessor = self.predictor.configuration_manager.preprocessor_class(verbose=False)

    def _inject_custom_trainer(self):
        """
        Copy file EDLTrainer.py từ src/trainers vào thư mục cài đặt của nnunetv2
        để hàm recursive_find_python_class có thể tìm thấy nó.
        Có bắt lỗi try-except để không làm dừng chương trình nếu copy thất bại.
        """
        try:
            # 1. Tìm vị trí cài đặt nnunetv2 trong môi trường Python hiện tại
            nnunet_path = os.path.dirname(nnunetv2.__file__)
            target_folder = os.path.join(nnunet_path, "training", "nnUNetTrainer")
            
            # 2. Tìm file source trong thư mục dự án (src/trainers)
            current_dir = os.path.dirname(os.path.abspath(__file__))
            source_file = os.path.join(current_dir, "trainers", "EDLTrainer.py")
            
            if not os.path.exists(source_file):
                # print(f"⚠️ Warning: Không tìm thấy file trainer tại {source_file}. Bỏ qua bước inject.")
                return

            # 3. Copy file (Overwrite nếu đã tồn tại)
            target_file = os.path.join(target_folder, "EDLTrainer.py")
            shutil.copy(source_file, target_file)
            # print("✅ Inject thành công! nnU-Net sẽ nhận diện được EDLTrainer.")
            
        except Exception as e:
            print(f"⚠️ Warning: Lỗi khi inject trainer (Có thể bỏ qua nếu đang dùng Standard Trainer): {e}")

    def _initialize_predictor(self):
        """Khởi tạo và load trọng số Model"""
        print("🚀 Initializing nnU-Net Predictor...")
        # 1. Lấy 2 công tắc từ config (nếu không có thì máy tự gán True và 0.5)
        use_tta = self.config.get("use_mirroring", True)
        step_size = self.config.get("tile_step_size", 0.5)
        print(f"   -> Cấu hình: Lật ảnh={use_tta} | Bước trượt={step_size}")

        try:
            # 2. Truyền 2 công tắc đó vào biến của nnUNetPredictor
            predictor = nnUNetPredictor(
                tile_step_size=step_size, 
                use_gaussian=True, 
                use_mirroring=use_tta,
                device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
                verbose=False
            )
            
            ckpt_path = self.config["checkpoint_path"]
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"❌ Checkpoint not found: {ckpt_path}")
                
            # nnU-Net yêu cầu đường dẫn folder cha chứa file checkpoint
            checkpoint_folder = os.path.dirname(os.path.dirname(ckpt_path))
            
            predictor.initialize_from_trained_model_folder(
                checkpoint_folder, use_folds=(0,), checkpoint_name="checkpoint_best.pth"
            )
            print(f"📂 Model loaded from: {checkpoint_folder}")
            return predictor
            
        except Exception as e:
            print(f"❌ Critical Error initializing predictor: {e}")
            raise e # Lỗi này nghiêm trọng, cần raise để dừng chương trình

    def save_nifti(self, data, affine, output_path):
        """Hàm phụ trợ lưu mảng numpy thành file .nii.gz"""
        try:
            # data shape: [X, Y, Z] -> Phải ép kiểu về float32 để tránh lỗi format header
            img = nib.Nifti1Image(data.astype(np.float32), affine)
            nib.save(img, output_path)
        except Exception as e:
            print(f"⚠️ Error saving NIfTI {output_path}: {e}")

    def process_case(self, case_id):
        """
        Xử lý trọn vẹn một ca bệnh:
        1. Load ảnh & Preprocess
        2. Inference (Dự đoán)
        3. Tính Uncertainty (Nếu mode=EDL)
        4. Lưu file kết quả
        """
        # print(f"\n🔍 Processing: {case_id}...")
        
        # --- 1. SETUP PATHS & CHECK FILES (Cơ chế bảo vệ) ---
        img_folder = self.config["image_folder"]
        lbl_folder = self.config["label_folder"]
        
        base_file = os.path.join(img_folder, f"{case_id}_0000.nii")
        ext = ".nii" if os.path.exists(base_file) else ".nii.gz"
        
        # Tạo danh sách 4 kênh (FLAIR, T1w, T1gd, T2w)
        image_files = [os.path.join(img_folder, f"{case_id}_{i:04d}{ext}") for i in range(4)]
        
        # [QUAN TRỌNG] Kiểm tra file input có tồn tại không
        if not os.path.exists(image_files[0]):
            print(f"❌ Error: Input files for {case_id} not found.")
            return None, None, None, None, None

        # [FIX] Tìm file GT thông minh
        gt_file = None
        gt_path_gz = os.path.join(lbl_folder, f"{case_id}.nii.gz")
        gt_path_nii = os.path.join(lbl_folder, f"{case_id}.nii")
        
        # --- THÊM ĐOẠN DEBUG NÀY ---
        print(f"🔍 DEBUG: Đang tìm GT cho {case_id}...")
        print(f"   - Thử: {gt_path_gz} -> {'CÓ' if os.path.exists(gt_path_gz) else 'KHÔNG'}")
        print(f"   - Thử: {gt_path_nii} -> {'CÓ' if os.path.exists(gt_path_nii) else 'KHÔNG'}")
        # ---------------------------

        if os.path.exists(gt_path_gz):
            gt_file = gt_path_gz
        elif os.path.exists(gt_path_nii):
            gt_file = gt_path_nii
        else:
            print(f"⚠️ WARNING: Không tìm thấy GT! Code sẽ chạy với GT đen sì.")
            gt_file = None

        # --- LẤY AFFINE MATRIX GỐC ---
        # Để đảm bảo file output chồng khít lên ảnh gốc trong ITK-SNAP
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
        # Dùng torch.no_grad() để tiết kiệm VRAM, không lưu gradient
        data_tensor = torch.from_numpy(data).to(self.predictor.device)
        with torch.no_grad():
            pred_logits = self.predictor.predict_logits_from_preprocessed_data(data_tensor)
        
        # --- 4. LOGIC PHÂN NHÁNH (EDL vs BASELINE) ---
        segmentation = torch.argmax(pred_logits, dim=0).cpu().numpy()
        
        # Khởi tạo dict rỗng (đen sì) để code Visualizer không bị lỗi
        unc_dict = {
            "total": np.zeros(segmentation.shape),
            "aleatoric": np.zeros(segmentation.shape),
            "epistemic": np.zeros(segmentation.shape)
        }

        # Lấy mode từ config, mặc định là 'edl' nếu không khai báo
        model_mode = self.config.get("model_mode", "edl")

        if model_mode == "edl":
            # --- TÍNH TOÁN UNCERTAINTY DECOMPOSITION ---
            # Công thức dựa trên Information Theory (Entropy của phân phối Dirichlet)
            
            # a. Tính tham số Dirichlet (alpha)
            evidence = F.softplus(pred_logits)
            alpha = evidence + 1
            S = torch.sum(alpha, dim=0, keepdim=True) # Tổng sức mạnh bằng chứng
            probs = alpha / S                         # Xác suất kỳ vọng
            
            # b. Total Uncertainty (Entropy)
            # H(p) = - sum(p * log(p))
            # Cộng thêm 1e-7 để tránh lỗi log(0) -> NaN
            total_unc = -torch.sum(probs * torch.log(probs + 1e-7), dim=0)
            
            # c. Aleatoric Uncertainty (Expected Entropy)
            # E[H(p)] approx sum(p * (digamma(S+1) - digamma(alpha+1)))
            digamma_S = torch.digamma(S + 1)
            digamma_alpha = torch.digamma(alpha + 1)
            aleatoric_unc = torch.sum(probs * (digamma_S - digamma_alpha), dim=0)
            
            # d. Epistemic Uncertainty (Mutual Information)
            # I = Total - Aleatoric
            epistemic_unc = total_unc - aleatoric_unc
            
            # e. Chuẩn hóa về Numpy & Clamp giá trị
            # Clamp min=0 để tránh sai số dấu chấm động làm ra số âm cực nhỏ
            unc_dict = {
                "total": torch.clamp(total_unc, min=0).cpu().numpy(),
                "aleatoric": torch.clamp(aleatoric_unc, min=0).cpu().numpy(),
                "epistemic": torch.clamp(epistemic_unc, min=0).cpu().numpy()
            }
        
        # Xử lý seg nếu không có GT (tạo ảnh đen để visualize không lỗi)
        if seg is None: seg = np.zeros((1, *segmentation.shape))
        
        # --- 5. SAVE NIFTI FILES ---
        if self.config.get("save_3d_nifti", False):
            try:
                nifti_folder_name = self.config.get("dir_nifti", "3d_nifti")
                out_dir = os.path.join(self.config["output_folder"], nifti_folder_name, case_id)
                os.makedirs(out_dir, exist_ok=True)
                
                # Chỉ lưu Uncertainty Maps nếu đang chạy mode EDL
                # (Baseline mà lưu cái này thì toàn ảnh đen, tốn dung lượng vô ích)
                if model_mode == "edl":
                    self.save_nifti(unc_dict["total"], original_affine, os.path.join(out_dir, "unc_total.nii.gz"))
                    self.save_nifti(unc_dict["aleatoric"], original_affine, os.path.join(out_dir, "unc_aleatoric.nii.gz"))
                    self.save_nifti(unc_dict["epistemic"], original_affine, os.path.join(out_dir, "unc_epistemic.nii.gz"))
                
                # Các file cơ bản (Luôn lưu)
                self.save_nifti(segmentation, original_affine, os.path.join(out_dir, "prediction.nii.gz"))
                self.save_nifti(seg[0], original_affine, os.path.join(out_dir, "ground_truth.nii.gz"))
                self.save_nifti(data[0], original_affine, os.path.join(out_dir, "mri_crop.nii.gz"))
                
            except Exception as e:
                print(f"⚠️ Error saving NIfTI files for {case_id}: {e}")

        # Return dict uncertainty đầy đủ để Visualizer vẽ
        return data, seg, segmentation, unc_dict, properties