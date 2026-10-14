"""Bai 4: So sanh bo loc Ideal - Gaussian - Butterworth trong mien tan so.

Dung lai ham cua Bai 1 va Bai 3 (cac file phai nam cung thu muc).
Usage: python bai4_so_sanh_bo_loc.py [image_path] [size]   (mac dinh: lena.png, 128)
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from bai1_dft_idft import load_gray, dft2, center_shift, log_spectrum, mse
from bai3_loc_ideal import distance_map, ideal_lowpass, apply_filter


def gaussian_lowpass(M, N, D0):
    """Eq.(15): H = exp(-D^2 / (2 D0^2))."""
    D = distance_map(M, N)
    return np.exp(-(D ** 2) / (2 * D0 ** 2))


def butterworth_lowpass(M, N, D0, n):
    """Eq.(16): H = 1 / (1 + (D/D0)^(2n))."""
    D = distance_map(M, N)
    return 1.0 / (1.0 + (D / D0) ** (2 * n))


def highpass(H_lp):
    """Eq.(17): H_HP = 1 - H_LP."""
    return 1.0 - H_lp


def psnr(a, b):
    m = mse(a, b)
    return np.inf if m == 0 else 10 * np.log10(255.0 ** 2 / m)


def show(a, img, title, vmin=0, vmax=255):
    a.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
    a.set_title(title, fontsize=9)
    a.axis("off")


def main():
    args = sys.argv[1:]
    size = int(args.pop()) if args and args[-1].isdigit() else 128
    path = args[0] if args else "lena.png"
    f = load_gray(path, size)
    M, N = f.shape
    Fs = center_shift(dft2(f))
    spec = log_spectrum(Fs)

    # ---- Hinh 1: cung D0, mask - pho sau loc - anh ket qua cua tung bo loc ----
    D0 = 30
    filters = {
        "Ideal": ideal_lowpass(M, N, D0),
        "Gaussian": gaussian_lowpass(M, N, D0),
        "Butterworth (n=2)": butterworth_lowpass(M, N, D0, 2),
    }
    fig1, ax = plt.subplots(3, 4, figsize=(15, 11))
    fig1.canvas.manager.set_window_title("Bai 4 - So sanh 3 bo loc thong thap (D0=30)")
    print(f"=== So sanh Ideal/Gaussian/Butterworth, D0 = {D0} ===")
    for i, (name, H) in enumerate(filters.items()):
        g = apply_filter(Fs, H)
        show(ax[i, 0], f, "Ảnh gốc")
        show(ax[i, 1], H, f"Mask {name}", 0, 1)
        show(ax[i, 2], log_spectrum(H * Fs), "Phổ sau lọc log(1+|G|)", spec.min(), spec.max())
        show(ax[i, 3], np.clip(g, 0, 255), f"Kết quả {name}\nMSE={mse(f, g):.1f}, PSNR={psnr(f, g):.1f} dB")
        print(f"  {name:18s}: MSE = {mse(f, g):7.2f} | PSNR = {psnr(f, g):.2f} dB")
    fig1.tight_layout()
    fig1.savefig("bai4_so_sanh_D0_30.png", dpi=110)

    # ---- Hinh 2: Butterworth voi n = 1,2,4 va D0 = 10,30,50,80 ----
    ns, D0s = [1, 2, 4], [10, 30, 50, 80]
    fig2, ax2 = plt.subplots(len(ns), len(D0s), figsize=(14, 10.5))
    fig2.canvas.manager.set_window_title("Bai 4 - Butterworth theo n va D0")
    print("\n=== Butterworth LPF: MSE(goc, loc) theo n va D0 ===")
    print("   n \\ D0 " + "".join(f"{d:>9d}" for d in D0s))
    for r, n in enumerate(ns):
        row = []
        for c, d in enumerate(D0s):
            g = apply_filter(Fs, butterworth_lowpass(M, N, d, n))
            row.append(mse(f, g))
            show(ax2[r, c], np.clip(g, 0, 255), f"n={n}, D0={d}\nPSNR={psnr(f, g):.1f} dB")
        print(f"   n={n}    " + "".join(f"{e:9.1f}" for e in row))
    fig2.tight_layout()
    fig2.savefig("bai4_butterworth_n_D0.png", dpi=110)

    # ---- Hinh 3: Ideal / Gaussian / Butterworth voi cac D0 ----
    fig3, ax3 = plt.subplots(3, len(D0s), figsize=(14, 10.5))
    fig3.canvas.manager.set_window_title("Bai 4 - Anh huong cua D0")
    for c, d in enumerate(D0s):
        for r, (name, H) in enumerate([("Ideal", ideal_lowpass(M, N, d)),
                                       ("Gaussian", gaussian_lowpass(M, N, d)),
                                       ("Butterworth n=2", butterworth_lowpass(M, N, d, 2))]):
            g = apply_filter(Fs, H)
            show(ax3[r, c], np.clip(g, 0, 255), f"{name}, D0={d}\nPSNR={psnr(f, g):.1f} dB")
    fig3.tight_layout()
    fig3.savefig("bai4_D0.png", dpi=110)

    # ---- Hinh 4: loc thong cao & lam sac net ----
    D0 = 30
    fig4, ax4 = plt.subplots(3, 4, figsize=(15, 11))
    fig4.canvas.manager.set_window_title("Bai 4 - Loc thong cao va lam sac net")
    hp = {
        "Ideal": highpass(ideal_lowpass(M, N, D0)),
        "Gaussian": highpass(gaussian_lowpass(M, N, D0)),
        "Butterworth (n=2)": highpass(butterworth_lowpass(M, N, D0, 2)),
    }
    k = 1.0  # he so sharpen: f_sharp = f + k * highpass(f)
    for i, (name, H) in enumerate(hp.items()):
        g = apply_filter(Fs, H)
        sharp = f + k * g
        show(ax4[i, 0], H, f"Mask thông cao {name}", 0, 1)
        show(ax4[i, 1], g, "Ảnh thông cao (cạnh)", g.min(), g.max())
        show(ax4[i, 2], f, "Ảnh gốc")
        show(ax4[i, 3], np.clip(sharp, 0, 255), f"Làm sắc nét: f + {k:g}·HP ({name})")
    fig4.tight_layout()
    fig4.savefig("bai4_thong_cao_sharpen.png", dpi=110)

    print("""
Nhận xét so sánh:
- Ideal: kết quả có vân sóng (ringing) quanh cạnh; Gaussian: mịn, không ringing nhưng mờ hơn khi cùng D0;
  Butterworth: nằm giữa hai loại trên, n càng lớn càng gần Ideal (ringing tăng), n nhỏ gần Gaussian.
- D0 tăng -> giữ nhiều tần số hơn -> ảnh rõ hơn, PSNR tăng.

Câu hỏi thảo luận:
- Ideal có bước nhảy 0/1 đột ngột trong miền tần số. Biến đổi ngược của hàm chữ nhật là hàm sinc (có các thùy dao động
  kéo dài), nên khi nhân chập với ảnh, các cạnh mạnh sinh ra các vân sóng quanh biên (ringing/Gibbs).
  Gaussian có biến đổi ngược cũng là Gaussian (dương, không dao động) và chuyển tiếp mềm nên không ringing.
- Bậc n trong Butterworth điều khiển độ dốc quanh D0: n=1 chuyển tiếp rất êm (gần Gaussian, ít ringing, nhưng lọc kém sắc);
  n lớn -> dốc hơn, tiệm cận Ideal (cắt gắt, ringing rõ).""")
    plt.show()


if __name__ == "__main__":
    main()
