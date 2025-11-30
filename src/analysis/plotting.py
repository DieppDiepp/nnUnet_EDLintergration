# src/analysis/plotting.py
import os
import matplotlib.pyplot as plt
import numpy as np

def plot_reliability_curve(
    valid_retention, mean_opt, mean_dice,
    agg_data, target, ause_total, qu_score,
    output_dir, zoom_config=[1.0, 0.5]
):
    """
    Vẽ biểu đồ Risk-Coverage Curve.
    Hỗ trợ 2 chế độ: Full Range và Zoomed Range (từ Config).
    """
    
    def _draw_plot(is_zoom=False):
        plt.figure(figsize=(8, 6))
        
        # Vẽ các đường chính
        plt.plot(valid_retention, mean_opt, 'k--', label='Optimal', linewidth=2, alpha=0.7)
        plt.plot(valid_retention, mean_dice, 'r-', label=f'Total (AUSE={ause_total:.3f})', linewidth=2)
        
        # Vẽ Aleatoric/Epistemic nếu có dữ liệu
        if agg_data["aleatoric_dice"]:
            mean_alea = np.mean(agg_data["aleatoric_dice"], axis=0)
            # Tính AUSE phụ ở đây nếu cần, hoặc truyền vào
            plt.plot(valid_retention, mean_alea, 'g:', label='Aleatoric', linewidth=2)
            
        if agg_data["epistemic_dice"]:
            mean_epis = np.mean(agg_data["epistemic_dice"], axis=0)
            plt.plot(valid_retention, mean_epis, 'b-.', label='Epistemic', linewidth=2)
        
        plt.xlabel("Retention Fraction", fontsize=12)
        plt.ylabel(f"Dice Score ({target})", fontsize=12)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.legend(loc='lower left', fontsize=10)

        # Xử lý Zoom theo Config
        if is_zoom:
            # zoom_config ví dụ: [1.0, 0.5]
            # Matplotlib xlim cần (max, min) để đảo trục, hoặc (min, max)
            # Ta muốn trục X chạy từ 1.0 về 0.5 (trái qua phải)
            # Nên set xlim(right, left) -> xlim(1.02, 0.48)
            
            max_r = max(zoom_config) + 0.02
            min_r = min(zoom_config) - 0.02
            
            plt.xlim(max_r, min_r) 
            plt.ylim(0.5, 1.02) # Dice thường > 0.5
            plt.title(f"Uncertainty Reliability ({target}) - Clinical Focus", fontsize=14)
            filename = f"ause_curve_{target}_zoomed.png"
        else:
            plt.gca().invert_xaxis() # Mặc định 1.0 -> 0.0
            plt.title(f"Uncertainty Reliability ({target}) - Full Range\nScore={qu_score:.3f}", fontsize=14)
            filename = f"ause_curve_{target}_full.png"
            
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename), dpi=300)
        plt.close()

    # Vẽ cả 2 bản
    _draw_plot(is_zoom=False)
    _draw_plot(is_zoom=True)