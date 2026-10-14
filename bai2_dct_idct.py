"""Bai 2: DCT/IDCT 2 chieu cho khoi anh 8x8 (khong dung cv2.dct / scipy.fft).

Usage: python bai2_dct_idct.py [image_path] [row col]   (mac dinh: lena.png, khoi tai (240, 240))
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

N = 8


def alpha(k, n=N):
    """Eq.(8)."""
    return np.sqrt(1.0 / n) if k == 0 else np.sqrt(2.0 / n)


def dct_basis(n=N):
    """A[u, x] = alpha(u) * cos((2x+1) u pi / 2n)."""
    A = np.zeros((n, n))
    for u in range(n):
        for x in range(n):
            A[u, x] = alpha(u, n) * np.cos((2 * x + 1) * u * np.pi / (2 * n))
    return A


def dct2(f):
    """Eq.(7): C(u,v) = a(u)a(v) sum_x sum_y f(x,y) cos(..) cos(..)
    Tach duoc: C = A f A^T."""
    A = dct_basis(f.shape[0])
    return A @ f @ A.T


def idct2(C):
    """Eq.(9): f(x,y) = sum_u sum_v a(u)a(v) C(u,v) cos(..) cos(..)  ->  f = A^T C A."""
    A = dct_basis(C.shape[0])
    return A.T @ C @ A


def dct2_naive(f):
    """Cai dat truc tiep Eq.(7) bang vong lap, de kiem chung dct2."""
    n = f.shape[0]
    C = np.zeros((n, n))
    for u in range(n):
        for v in range(n):
            s = 0.0
            for x in range(n):
                for y in range(n):
                    s += f[x, y] * np.cos((2 * x + 1) * u * np.pi / (2 * n)) \
                                 * np.cos((2 * y + 1) * v * np.pi / (2 * n))
            C[u, v] = alpha(u, n) * alpha(v, n) * s
    return C


def zero_smallest(C, ratio):
    """Dat `ratio` (0..1) cac he so co |C| nho nhat ve 0."""
    k = int(round(ratio * C.size))
    out = C.copy()
    idx = np.argsort(np.abs(out).ravel(), kind="stable")[:k]
    out.ravel()[idx] = 0.0
    return out


def mse(a, b):
    return np.mean((a - b) ** 2)


def psnr(a, b):
    m = mse(a, b)
    return np.inf if m == 0 else 10 * np.log10(255.0 ** 2 / m)


def main():
    args = sys.argv[1:]
    path = args[0] if args and not args[0].isdigit() else "lena.png"
    nums = [int(a) for a in args if a.isdigit()]
    r0, c0 = (nums + [240, 240])[:2] if len(nums) >= 2 else (240, 240)

    img = np.asarray(Image.open(path).convert("L"), dtype=np.float64)
    f = img[r0:r0 + N, c0:c0 + N]
    assert f.shape == (N, N), "Khoi vuot ra ngoai anh"

    # DCT cua khoi (khong tru 128, dung dung cong thuc de bai)
    C = dct2(f)
    f_rec = idct2(C)
    print("Kiem tra dct2 vs cong thuc truc tiep: max|diff| =", np.abs(C - dct2_naive(f)).max())
    print("Sai so khoi goc vs IDCT: MSE =", mse(f, f_rec))

    np.set_printoptions(precision=2, suppress=True, linewidth=150)
    print("\nMa tran he so DCT 8x8:\n", C)
    print("\nHe so DC C(0,0) =", C[0, 0], " (= 8 * gia tri trung binh =", 8 * f.mean(), ")")
    print("Mot so he so AC: C(0,1) = %.2f, C(1,0) = %.2f, C(1,1) = %.2f, C(7,7) = %.2f"
          % (C[0, 1], C[1, 0], C[1, 1], C[7, 7]))
    np.savetxt("bai2_dct_coefficients.txt", C, fmt="%.4f")
    print("Da luu ma tran he so vao bai2_dct_coefficients.txt")

    # --- Hien thi yeu cau 4 ---
    fig, ax = plt.subplots(1, 5, figsize=(19, 4.2))
    fig.canvas.manager.set_window_title("Bai 2 - DCT/IDCT khoi 8x8")
    ax[0].imshow(f, cmap="gray", vmin=0, vmax=255); ax[0].set_title(f"Khối ảnh gốc 8×8\n(tại hàng {r0}, cột {c0})")
    im = ax[1].imshow(C, cmap="coolwarm"); ax[1].set_title("Ma trận hệ số DCT 8×8")
    for u in range(N):
        for v in range(N):
            ax[1].text(v, u, f"{C[u, v]:.0f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=ax[1], fraction=0.046)
    ax[2].bar(["DC\nC(0,0)"], [C[0, 0]], color="tab:red")
    ax[2].set_title("Hệ số DC")
    ac = {"C(0,1)": C[0, 1], "C(1,0)": C[1, 0], "C(1,1)": C[1, 1], "C(2,2)": C[2, 2], "C(7,7)": C[7, 7]}
    ax[3].bar(list(ac.keys()), list(ac.values()), color="tab:blue")
    ax[3].set_title("Một số hệ số AC"); ax[3].tick_params(axis="x", labelsize=8)
    ax[4].imshow(f_rec, cmap="gray", vmin=0, vmax=255); ax[4].set_title(f"Khối sau IDCT\nMSE={mse(f, f_rec):.2e}")
    for a in (ax[0], ax[4]):
        a.axis("off")
    fig.tight_layout()
    fig.savefig("bai2_result.png", dpi=120)

    # --- Yeu cau 6: loai 50%, 75%, 90% he so nho nhat ---
    ratios = [0.50, 0.75, 0.90]
    fig2, ax2 = plt.subplots(2, 4, figsize=(15, 7.5))
    fig2.canvas.manager.set_window_title("Bai 2 - Loai bo he so DCT")
    ax2[0, 0].imshow(f, cmap="gray", vmin=0, vmax=255); ax2[0, 0].set_title("Khối gốc")
    ax2[1, 0].imshow(np.abs(C) > 0, cmap="gray"); ax2[1, 0].set_title("Hệ số khác 0: 64/64")
    print("\nLoai bo he so DCT co do lon nho nhat:")
    for i, r in enumerate(ratios, start=1):
        Cz = zero_smallest(C, r)
        fz = idct2(Cz)
        kept = int(np.count_nonzero(Cz))
        e, p = mse(f, fz), psnr(f, fz)
        print(f"  {int(r*100)}%: giu {kept}/64 he so | MSE = {e:.3f} | PSNR = {p:.2f} dB")
        ax2[0, i].imshow(np.clip(fz, 0, 255), cmap="gray", vmin=0, vmax=255)
        ax2[0, i].set_title(f"Bỏ {int(r*100)}% hệ số\nMSE={e:.2f}, PSNR={p:.1f} dB")
        ax2[1, i].imshow(Cz != 0, cmap="gray"); ax2[1, i].set_title(f"Hệ số giữ lại: {kept}/64")
    for a in ax2.ravel():
        a.axis("off")
    fig2.tight_layout()
    fig2.savefig("bai2_loai_he_so.png", dpi=120)

    print("""
Trả lời câu hỏi thảo luận:
- C(0,0) (hệ số DC) = sqrt(1/N)*sqrt(1/N)*tổng pixel = N * giá trị trung bình khối. Nó mang độ sáng trung bình của khối.
- Ảnh tự nhiên biến đổi chậm (các vùng mịn, lân cận pixel tương quan mạnh) nên năng lượng dồn vào DC và
  các hệ số tần số thấp (góc trên-trái); hệ số tần số cao thường rất nhỏ.
- C(u,v) với (u,v)≠(0,0) (AC) mô tả sự biến thiên quanh giá trị trung bình: u theo chiều dọc, v theo chiều ngang;
  (u,v) càng lớn thì chi tiết/cạnh/nhiễu càng mịn. Bỏ các hệ số AC nhỏ ít làm méo ảnh -> cơ sở của nén JPEG.""")
    plt.show()


if __name__ == "__main__":
    main()
