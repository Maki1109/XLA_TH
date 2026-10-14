# XLA - Bài thực hành 4: Biến đổi ảnh trong miền tần số

DFT, DCT và lọc ảnh trong miền tần số, tự cài đặt (không dùng `np.fft`, `cv2.dct`, `scipy.fft`). Ảnh thử nghiệm: `lena.png`.

| File | Nội dung |
|---|---|
| `bai1_dft_idft.py` | Bài 1: DFT/IDFT 2D, phổ biên độ/pha, dịch tâm, MSE |
| `bai2_dct_idct.py` | Bài 2: DCT/IDCT khối 8x8, bỏ 50/75/90% hệ số |
| `bai3_loc_ideal.py` | Bài 3: lọc Ideal thông thấp/thông cao (import Bài 1) |
| `bai4_so_sanh_bo_loc.py` | Bài 4: so sánh Ideal, Gaussian, Butterworth (import Bài 1, 3) |
| `main.tex` | Báo cáo (LaTeX); ảnh `bai*.jpg`, `logo.png` cùng thư mục |
| `results/` | Hình kết quả (PNG gốc) |

Chạy (cần `numpy`, `matplotlib`, `pillow`):

```bash
python bai1_dft_idft.py
python bai2_dct_idct.py [ảnh] [hàng cột]
python bai3_loc_ideal.py [ảnh] [size]
python bai4_so_sanh_bo_loc.py [ảnh] [size]
```
