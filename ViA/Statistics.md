# Benchmarking & Reporting Framework for Vision in Action (ViA) Diffusion Policy

## Executive Summary

Vision in Action (ViA) là mô hình Diffusion Policy kết hợp Vision Foundation Model (DINOv2) nhằm dự đoán chuỗi hành động điều khiển robot hai tay (bi-manual humanoid / ALOHA robot) với không gian hành động 23 chiều (23-DOF action space).

Báo cáo benchmark và thống kê của mô hình ViA được thiết kế toàn diện theo các trục trọng yếu:

1. **Độ chính xác hành động (Policy Accuracy):** Sai số góc khớp (Joint angle error in radians/degrees) đo bằng Action MSE, Action MAE, và phân phối sai số Euclidean (Mean, Median, P95, P99).
2. **Đặc thù hệ thống cơ điện tử (Subsystem Breakdown):** Tách biệt sai số giữa Active Perception (Neck 7 DOF) và Manipulation (Right Arm 8 DOF, Left Arm 8 DOF, Grippers 2 DOF).
3. **Chất lượng động học & độ mượt mà (Trajectory Smoothness / Jerk):** Đo đạo hàm bậc 2 của vận tốc (Jerk) để phát hiện rung giật robot (motor shudder/jitter).
4. **Khả năng đáp ứng thời gian thực (Real-time Control Frequency):** Tần số phát lệnh điều khiển (Hz) so với ngân sách điều khiển robot thực tế (>= 10 Hz).
5. **Chi phí tính toán & VRAM (Computational Cost & Memory):** Tối ưu hóa TensorFloat-32 (TF32), Peak VRAM, CUDA Allocator Fragmentation.
6. **Bằng chứng thị giác trực quan (Visual Evidence):** Trực quan hóa bảng điều khiển 6 panel (Dashboard), xuất video rollout MP4 và từng frame ảnh độc lập.

> **Quy ước:** Đối với sai số (MSE, MAE, Jerk) và tài nguyên (VRAM, Latency), giá trị **thấp hơn** là tốt hơn. Đối với tần số điều khiển (Hz) và thông lượng (Throughput), giá trị **cao hơn** là tốt hơn.

---

## I. Thống kê trong quá trình Training (ViA Diffusion Policy)

Tất cả log quá trình huấn luyện, bảng số liệu CSV, biểu đồ loss curves và checkpoint model được lưu trữ tách biệt tại thư mục `training_logs/` (logs, metrics) và `via_model_ckps/` (weights `.pth`).

### A. Training / Validation Policy Performance

| Parameter | Full English Name (Abbreviation) | Tên tiếng Việt | Giải thích | Đơn vị | Cao/thấp tốt hơn? | Note |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Training Noise Loss** | Training Diffusion Noise MSE Loss | Hàm mất mát nhiễu khi train | MSE loss giữa nhiễu dự đoán bởi Conditional U-Net $\epsilon_\theta(a_t, t, c)$ và nhiễu chuẩn $\epsilon \sim \mathcal{N}(0, I)$ | — | Thấp hơn | Metric loss chính được tối ưu hóa bằng AdamW |
| **Validation Action MSE** | Validation Action Mean Squared Error (MSE) | Sai số bình phương trung bình góc khớp | $\frac{1}{23T} \sum_{t=1}^T \sum_{d=1}^{23} (a_{t,d} - \hat{a}_{t,d})^2$ trên tập validation sau DDIM denoising | $\text{rad}^2$ | Thấp hơn | Metric đánh giá chính xác chuẩn của ViA |
| **Best Validation MSE** | Best Validation Action MSE | Validation Action MSE tốt nhất | Giá trị Validation MSE thấp nhất đạt được trong toàn bộ quá trình training | $\text{rad}^2$ | Thấp hơn | Dùng để chọn checkpoint EMA tốt nhất |
| **Validation Action MAE** | Validation Action Mean Absolute Error (MAE) | Sai số tuyệt đối trung bình góc khớp | $\frac{1}{23T} \sum_{t=1}^T \sum_{d=1}^{23} \|a_{t,d} - \hat{a}_{t,d}\|$ | rad / deg | Thấp hơn | Cho biết góc lệch trung bình trực quan của các khớp |
| **Neck Validation MSE** | Active Perception Neck MSE (7D) | MSE cụm camera cổ robot | MSE trên 7 bậc tự do của cổ (Joints 0–6) | $\text{rad}^2$ | Thấp hơn | Đánh giá khả năng bám đuổi thị giác (gaze tracking) |
| **Right Arm Validation MSE** | Right Arm Manipulation MSE (8D) | MSE cụm tay phải | MSE trên 8 bậc tự do tay phải (Joints 7–14, bao gồm kẹp phải) | $\text{rad}^2$ | Thấp hơn | Đánh giá độ chính xác vươn tay và gắp vật thể |
| **Left Arm Validation MSE** | Left Arm Manipulation MSE (8D) | MSE cụm tay trái | MSE trên 8 bậc tự do tay trái (Joints 15–22, bao gồm kẹp trái) | $\text{rad}^2$ | Thấp hơn | Đánh giá độ chính xác phối hợp hai tay |
| **Grippers Validation MSE** | Grippers Actuation MSE (2D) | MSE động tác đóng/mở kẹp | MSE trên 2 khớp ngón kẹp (Joints 14 & 22) | $\text{rad}^2$ | Thấp hơn | Quyết định thành bại của việc gắp và nhả vật |
| **Train–Val Loss Gap** | Training–Validation Generalization Gap | Độ lệch train-validation | Chênh lệch giữa train loss và validation error | — | Thấp hơn | Phát hiện hiện tượng overfitting |

---

### B. Training Convergence & Optimization

| Parameter | Full English Name (Abbreviation) | Tên tiếng Việt | Giải thích | Đơn vị | Cao/thấp tốt hơn? | Note |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Learning Rate** | Learning Rate (LR) | Tốc độ học | Giá trị LR tại epoch hiện tại | — | — | Schedule cosine annealing có warmup |
| **Warmup Steps** | Learning Rate Warmup Steps | Số bước khởi động | Số batch tăng tuyến tính LR từ 0 lên max LR | steps | — | Mặc định $\min(500, \text{batches}/2)$ |
| **EMA Decay Rate** | Exponential Moving Average Power | Hệ số trích xuất trọng số EMA | Trọng số cập nhật trung bình lũy thừa cho inference: $\theta_{\text{EMA}} \leftarrow \beta \theta_{\text{EMA}} + (1-\beta) \theta$ | — | — | Mặc định power = 0.75 trong ViA |
| **Diffusion Train Timesteps** | Number of Train DDIM Timesteps | Số bước khuếch tán huấn luyện | Số bước nhiễu rời rạc hóa trong quá trình thêm nhiễu | steps | — | Mặc định 50 timesteps |
| **Epoch Duration** | Epoch Elapsed Time | Thời gian hoàn thành 1 epoch | Thời gian chạy hết 1 epoch dữ liệu | seconds | Thấp hơn | Lưu tự động trong `training_metrics.csv` |

---

### C. Training Computational Cost & Resource Profiling

| Parameter | Full English Name (Abbreviation) | Tên tiếng Việt | Giải thích | Đơn vị | Cao/thấp tốt hơn? | Note |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Total Training Time** | Total Training Time | Tổng thời gian huấn luyện | Tổng thời gian chạy từ epoch 0 đến khi kết thúc | s / min / h | Thấp hơn | Ghi vào `training_summary.json` |
| **Training Throughput** | Training Sample Throughput | Tốc độ xử lý mẫu huấn luyện | Số frame mẫu được nạp và tính toán trên mỗi giây | samples/s | Cao hơn | Đánh giá hiệu năng data pipeline |
| **Peak GPU Memory** | Peak Training VRAM Usage | Bộ nhớ VRAM cực đại khi train | Lượng VRAM lớn nhất mà PyTorch cấp phát | MB / GB | Thấp hơn | Giám sát an toàn bộ nhớ GPU |
| **TF32 Precision Status** | TensorFloat-32 Acceleration | Trạng thái tăng tốc TF32 | Sử dụng `torch.set_float32_matmul_precision('high')` | Boolean | Bật (True) | Tăng tốc 3x–8x ma trận trên GPU NVIDIA Ampere+ |

---

## II. Thống kê trong quá trình Inference & Policy Rollout

### A. Primary Policy Accuracy (Sai số hành động tổng thể)

| Parameter | Full English Name (Abbreviation) | Tên tiếng Việt | Giải thích | Đơn vị | Cao/thấp tốt hơn? | Note |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Overall Action MSE** | Overall Action Mean Squared Error | MSE sai số quỹ đạo góc khớp | $\frac{1}{23T} \sum_{t=1}^T \sum_{d=1}^{23} (a_{t,d} - \hat{a}_{t,d})^2$ | $\text{rad}^2$ | Thấp hơn | Chỉ số đánh giá cốt lõi |
| **Overall Action MAE** | Overall Action Mean Absolute Error | MAE sai số quỹ đạo góc khớp | $\frac{1}{23T} \sum_{t=1}^T \sum_{d=1}^{23} \|a_{t,d} - \hat{a}_{t,d}\|$ | rad | Thấp hơn | Sai số góc trung bình |
| **Overall Action MAE (Deg)** | Action Mean Absolute Error in Degrees | Sai số góc trung bình theo độ | Chuyển đổi MAE sang độ: $\text{MAE} \times \frac{180^\circ}{\pi}$ | độ (deg / °) | Thấp hơn | Đánh giá trực quan cơ học góc quay |
| **Mean Step Euclidean Error** | Mean Step Euclidean Distance | Khoảng cách sai số Euclidean trung bình | $e_t = \|\hat{a}_t - a^*_t\|_2$ trung bình trên $T$ bước | rad | Thấp hơn | Độ lệch vector 23D tại mỗi timestep |
| **Median Step Error** | Median Euclidean Distance | Trung vị sai số Euclidean (P50) | Giá trị sai số tại percentile 50 | rad | Thấp hơn | Bền vững với các điểm outlier |
| **Step Error P95** | 95th Percentile Step Error (P95) | Sai số Euclidean percentile 95 | 95% số bước thực thi có sai số không vượt quá ngưỡng này | rad | Thấp hơn | Đánh giá hành vi ở các khúc rẽ/chuyển động khó |
| **Step Error P99** | 99th Percentile Step Error (P99) | Sai số Euclidean percentile 99 | Đánh giá sai số cực hạn (worst-case tail error) | rad | Thấp hơn | Đánh giá an toàn chống va đập |
| **Min / Max Step Error** | Minimum / Maximum Step Error | Sai số nhỏ nhất và lớn nhất | Cận dưới và cận trên của sai số Euclidean | rad | Thấp hơn | Xác định đỉnh sai số đột biến |

---

### B. Subsystem Grouped Breakdown (Phân tích theo cụm cơ cấu robot)

| Parameter | Full English Name (Abbreviation) | Tên tiếng Việt | Khớp đại diện (Joint Slice) | Đơn vị | Cao/thấp tốt hơn? | Note |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Neck Tracking MSE** | Active Perception Neck Tracking MSE | MSE cụm cổ robot (7 DOF) | Joints 0 đến 6 (Pan, Tilt, Roll cổ) | $\text{rad}^2$ | Thấp hơn | Sai số bám mục tiêu của camera |
| **Neck Tracking MAE** | Active Perception Neck Tracking MAE | MAE cụm cổ robot | Joints 0 đến 6 | rad / deg | Thấp hơn | Lệch góc camera trung bình |
| **Right Arm MSE** | Right Arm Manipulation MSE | MSE tay phải thao tác (8 DOF) | Joints 7 đến 14 (Vai, khuỷu, cổ tay, kẹp phải) | $\text{rad}^2$ | Thấp hơn | Tay thao tác chính |
| **Right Arm MAE** | Right Arm Manipulation MAE | MAE tay phải thao tác | Joints 7 đến 14 | rad / deg | Thấp hơn | Độ lệch góc trung bình tay phải |
| **Left Arm MSE** | Left Arm Manipulation MSE | MSE tay trái phối hợp (8 DOF) | Joints 15 đến 22 (Vai, khuỷu, cổ tay, kẹp trái) | $\text{rad}^2$ | Thấp hơn | Tay phối hợp nâng đỡ |
| **Left Arm MAE** | Left Arm Manipulation MAE | MAE tay trái phối hợp | Joints 15 đến 22 | rad / deg | Thấp hơn | Độ lệch góc trung bình tay trái |
| **Grippers MSE** | Grippers Actuation MSE | MSE cơ cấu kẹp 2 tay (2 DOF) | Joint 14 (kẹp phải) & Joint 22 (kẹp trái) | $\text{rad}^2$ | Thấp hơn | Quyết định thời điểm đóng/mở kẹp vật thể |
| **Grippers MAE** | Grippers Actuation MAE | MAE cơ cấu kẹp 2 tay | Joint 14 & Joint 22 | rad / deg | Thấp hơn | Sai số độ mở ngón kẹp |

---

### C. Trajectory Dynamics & Smoothness (Độ mượt mà và rung giật quỹ đạo)

| Parameter | Full English Name (Abbreviation) | Tên tiếng Việt | Giải thích | Đơn vị | Cao/thấp tốt hơn? | Note |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Predicted Jerk** | Predicted Trajectory Jerk | Độ rung giật quỹ đạo dự đoán | $\frac{1}{T} \sum_{t=2}^{T-1} \|\hat{a}_{t+1} - 2\hat{a}_t + \hat{a}_{t-1}\|_2^2$ | $\text{rad}^2/\text{step}^4$ | Thấp hơn | Đo biến thiên gia tốc (giật động cơ) |
| **Ground Truth Jerk** | Ground Truth Trajectory Jerk | Độ rung giật quỹ đạo mẫu chuyên gia | $\frac{1}{T} \sum_{t=2}^{T-1} \|a^*_{t+1} - 2a^*_t + a^*_{t-1}\|_2^2$ | $\text{rad}^2/\text{step}^4$ | Thấp hơn | Chuẩn tham chiếu của chuyên gia |
| **Jerk Ratio** | Trajectory Jerk Ratio | Tỷ lệ rung giật tương đối | $\frac{\text{Predicted Jerk}}{\text{Ground Truth Jerk} + 10^{-8}}$ | Ratio (x) | Gần 1.0 tốt nhất | Tỷ lệ quá cao cho thấy chuyển động bị giật cục |

---

### D. Inference Latency & Hardware Efficiency (Độ trễ và tần số điều khiển)

| Parameter | Full English Name (Abbreviation) | Tên tiếng Việt | Giải thích | Đơn vị | Cao/thấp tốt hơn? | Note |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Mean Latency** | Mean Inference Latency | Độ trễ inference trung bình | Thời gian chạy 1 lần DINOv2 encode + 16 bước DDIM sampling | ms/chunk | Thấp hơn | Đo chính xác bằng `torch.cuda.Event` |
| **Median Latency** | Median Inference Latency | Trung vị độ trễ | Độ trễ tại percentile 50 | ms | Thấp hơn | Ổn định, loại trừ nhiễu hệ điều hành |
| **P95 Latency** | 95th Percentile Latency (P95) | Độ trễ percentile 95 | 95% số lần inference không vượt quá giá trị này | ms | Thấp hơn | Tiêu chuẩn đảm bảo real-time an toàn |
| **P99 Latency** | 99th Percentile Latency (P99) | Độ trễ percentile 99 | Độ trễ đuôi trường hợp xấu nhất | ms | Thấp hơn | Giám sát điểm nghẽn đột ngột |
| **Control Frequency** | Real-Time Control Frequency | Tần số phát lệnh điều khiển | $f = \frac{1000}{\text{Mean Latency (ms)}}$ | Hz | Cao hơn | Ngân sách điều khiển robot: $\ge 10\text{ Hz}$ |
| **Peak Allocated VRAM** | Peak Allocated GPU VRAM | VRAM Tensor thực tế | Bộ nhớ GPU thực tế chứa weights, activations, KV tokens | MB / GB | Thấp hơn | Bộ nhớ tối thiểu yêu cầu trên thiết bị Edge |
| **Peak Reserved VRAM** | Peak Reserved GPU Memory | VRAM PyTorch giữ chỗ | Bộ nhớ mà CUDA memory caching allocator giữ lại | MB / GB | Thấp hơn | Dùng để phân tích hiện tượng fragmentation |
| **Memory Fragmentation** | Allocator Memory Fragmentation | Độ phân mảnh bộ nhớ VRAM | $\text{Reserved VRAM} - \text{Allocated VRAM}$ | MB | Thấp hơn | Đánh giá rủi ro crash bộ nhớ dài hạn |

---

### E. Model Footprint & Computational Complexity

| Parameter | Full English Name (Abbreviation) | Tên tiếng Việt | Giải thích | Đơn vị | Giá trị thực tế (ViA) | Note |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Vision Foundation Model Params** | DINOv2 ViT-S/14 Parameters | Số tham số Vision Encoder | DINOv2 backbone trích xuất visual tokens | M (Million) | 22.06 M | Backbone thị giác nền tảng |
| **Diffusion Policy UNet Params** | 1D Conditional U-Net Parameters | Số tham số Diffusion Policy | Conditional U-Net 1D dự đoán nhiễu khuếch tán | M (Million) | 49.02 M | Policy model chuyên biệt |
| **Total Model Parameters** | Total Number of Parameters | Tổng số tham số toàn mô hình | DINOv2 + Conditional U-Net | M (Million) | 71.08 M | Mô hình tương đối gọn nhẹ |
| **Model Checkpoint Size** | Checkpoint File Size on Disk | Dung lượng file checkpoint | Dung lượng file `.pth` chứa EMA weights | MB | 372.7 MB | Dễ dàng deploy lên robot |
| **DDIM Denoising Timesteps** | Test-Time DDIM Sampling Steps | Số bước lặp lấy mẫu DDIM | Số lần lặp qua U-Net trong quá trình sinh quỹ đạo | steps | 16 timesteps | Cân bằng giữa tốc độ và chất lượng |

---

## III. Baseline Benchmark Results (via_dinov2_cls_cup_ema_0)

Dưới đây là kết quả thực tế đo đạc trực tiếp từ checkpoint đã huấn luyện (`via_dinov2_cls_cup_ema_0.pth`) trên tập kiểm định `cup_val.zarr.zip` chạy trên GPU NVIDIA RTX 3050 (8GB VRAM, TF32 Precision):

| Nhóm Metric | Thông số đo đạc | Giá trị thực tế | Đơn vị | Ghi chú đánh giá |
| :--- | :--- | :--- | :--- | :--- |
| **Policy Accuracy** | Overall Action MSE | **0.086757** | $\text{rad}^2$ | Sai số bình phương trung bình trên 23 chiều |
| | Overall Action MAE | **0.207911** | rad | Sai số góc trung bình tương đương **11.91°** |
| | Mean Step Distance | **1.3895** | rad | Độ lệch norm vector 23D mỗi bước |
| **Subsystem Breakdown** | Active Perception Neck (7D) | **0.032225** | $\text{rad}^2$ | Cổ bám mục tiêu chính xác nhất (MAE: 8.24°) |
| | Left Arm Manipulation (8D) | **0.092393** | $\text{rad}^2$ | Tay trái phối hợp (MAE: 11.84°) |
| | Right Arm Manipulation (8D) | **0.128837** | $\text{rad}^2$ | Tay phải thao tác gắp (MAE: 15.20°) |
| | Grippers Actuation (2D) | **0.439336** | $\text{rad}^2$ | Ngón kẹp đóng/mở (MAE: 31.08°) |
| **Trajectory Dynamics** | Ground Truth Jerk | **0.000058** | $\text{rad}^2/\text{step}^4$ | Quỹ đạo mẫu chuyên gia rất mượt |
| | Predicted Jerk | **7.5126** | $\text{rad}^2/\text{step}^4$ | Quỹ đạo khuếch tán cần lọc mượt |
| **Speed & Hardware** | Mean Inference Latency | **435.8** | ms | 16 bước DDIM trên RTX 3050 |
| | Control Frequency | **2.3** | Hz | Tần số sinh chuỗi hành động 16 bước |
| | Checkpoint Disk Size | **372.7** | MB | File weights `.pth` lưu trên đĩa |
| | Total Parameters | **71.08** | M | 22.06M Vision + 49.02M Policy |

---

## IV. Danh mục công cụ Benchmark & Visualizer đã triển khai

Hệ thống cung cấp các script tự động trích xuất và lưu trữ toàn bộ bằng chứng thí nghiệm theo quy chuẩn:

1. **Bộ Engine Đánh giá Toàn diện (`benchmark_via.py`):**
   - Chạy kiểm thử hàng loạt, phân tích phân phối sai số (MSE, MAE, P50, P95, P99), đo thời gian CUDA Events và bộ nhớ VRAM.
   - Xuất tự động: `benchmark_report.json`, `benchmark_summary.md`, `benchmark_metrics.csv`, và thư mục đồ thị `plots/`.
   - Lệnh chạy:

     ```powershell
     python benchmark_via.py --config config/train_windows_small.yaml --checkpoint via_model_ckps/via_dinov2_cls_cup_ema_0.pth --dataset dataset/cup_val.zarr.zip --output-dir benchmark_evidence
     ```

2. **Công cụ Trực quan hóa Mẫu & Sinh Video Quỹ đạo (`infer_single_sample.py`):**
   - Thực hiện inference trên 1 sample (1 quan sát hình ảnh), phân tích chuỗi 16 bước hành động tương ứng.
   - Xuất dashboard 6 panel phân tích trực quan (`inference_sample_0.png`), xuất số liệu JSON (`inference_sample_0.json`).
   - Hỗ trợ xuất từng frame ảnh độc lập (`--save-frames`) và ghép video MP4 (`--video`) mô phỏng 16 bước hành động với camera overlay.
   - Lệnh chạy:

     ```powershell
     python infer_single_sample.py --sample-idx 0 --video --save-frames --output-dir inference_output
     ```
