"""Bai 3: Loc thong thap / thong cao ly tuong (Ideal) trong mien tan so.

Dung lai dft2/idft2/center_shift cua Bai 1 (file bai1_dft_idft.py phai nam cung thu muc).
Usage: python bai3_loc_ideal.py [image_path] [size]   (mac dinh: lena.png, 128)
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from bai1_dft_idft import load_gray, dft2, idft2, center_shift, center_unshift, log_spectrum, mse


def distance_map(M, N):
    """Eq.(11): D(u,v) = sqrt((u - M/2)^2 + (v - N/2)^2)  (pho da dich tam)."""
    u = np.arange(M).reshape(-1, 1)
    v = np.arange(N).reshape(1, -1)
    return np.sqrt((u - M / 2) ** 2 + (v - N / 2) ** 2)


def ideal_lowpass(M, N, D0):
    """Eq.(12)."""
    return (distance_map(M, N) <= D0).astype(np.float64)


def ideal_highpass(M, N, D0):
    """Eq.(14): H_HP = 1 - H_LP."""
    return 1.0 - ideal_lowpass(M, N, D0)


def apply_filter(Fs, H):
    """Eq.(13): G = H * F (tren pho da dich tam), roi IDFT -> anh."""
    G = H * Fs
    return idft2(center_unshift(G)).real


def main():
    args = sys.argv[1:]
    size = int(args.pop()) if args and args[-1].isdigit() else 128
    path = args[0] if args else "lena.png"
    f = load_gray(path, size)
    M, N = f.shape

    F = dft2(f)
    Fs = center_shift(F)
    spec = log_spectrum(Fs)

    D0_list = [10, 30, 50]
    fig, ax = plt.subplots(len(D0_list), 6, figsize=(20, 3.4 * len(D0_list)))
    fig.canvas.manager.set_window_title("Bai 3 - Loc Ideal thong thap / thong cao")
    print(f"Anh {M}x{N}")
    for i, D0 in enumerate(D0_list):
        Hlp, Hhp = ideal_lowpass(M, N, D0), ideal_highpass(M, N, D0)
        g_lp, g_hp = apply_filter(Fs, Hlp), apply_filter(Fs, Hhp)
        energy = (np.abs(Fs * Hlp) ** 2).sum() / (np.abs(Fs) ** 2).sum()
        print(f"D0 = {D0:3d}: MSE(goc, LPF) = {mse(f, g_lp):8.2f} | nang luong giu lai trong LPF = {100*energy:.2f}%")

        ax[i, 0].imshow(f, cmap="gray", vmin=0, vmax=255); ax[i, 0].set_title("Ảnh gốc")
        ax[i, 1].imshow(spec, cmap="gray"); ax[i, 1].set_title("Phổ Fourier log(1+|F|)")
        ax[i, 2].imshow(Hlp, cmap="gray"); ax[i, 2].set_title(f"Mask thông thấp (D0={D0})")
        ax[i, 3].imshow(np.clip(g_lp, 0, 255), cmap="gray", vmin=0, vmax=255); ax[i, 3].set_title("Ảnh sau lọc thông thấp")
        ax[i, 4].imshow(Hhp, cmap="gray"); ax[i, 4].set_title(f"Mask thông cao (D0={D0})")
        # anh thong cao co gia tri am/quanh 0 -> co gian ve [0,255] de hien thi
        ax[i, 5].imshow(g_hp, cmap="gray"); ax[i, 5].set_title("Ảnh sau lọc thông cao")
        for a in ax[i]:
            a.axis("off")
    fig.tight_layout()
    fig.savefig("bai3_result.png", dpi=110)

    print("""
Nhận xét:
- D0 nhỏ (10): chỉ giữ vài tần số thấp nhất -> ảnh rất mờ, xuất hiện vân sóng (ringing) quanh cạnh;
  D0 tăng (30, 50) -> giữ thêm nhiều tần số -> ảnh rõ dần và tiến về ảnh gốc, ringing giảm.
- Với lọc thông cao ngược lại: D0 càng nhỏ thì chỉ bỏ ít tần số thấp, ảnh còn nhiều nội dung;
  D0 càng lớn thì chỉ còn cạnh/chi tiết mảnh, vùng phẳng chuyển về xám/tối.
- Tần số cao biểu diễn thay đổi mức xám đột ngột (cạnh, chi tiết, nhiễu). Loại bỏ chúng (LPF) làm cạnh mất độ dốc -> ảnh mờ.
- Tần số thấp biểu diễn vùng phẳng, độ sáng nền (kể cả DC). Loại bỏ chúng (HPF) xóa vùng phẳng,
  chỉ giữ lại biến thiên nhanh -> cạnh nổi bật.
- Mask Ideal cắt cứng (0/1) nên sinh ringing (hàm sinc trong miền không gian).""")
    plt.show()


if __name__ == "__main__":
    main()
