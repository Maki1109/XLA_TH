"""Bai 1: Xay dung DFT va IDFT 2 chieu (khong dung np.fft).

Usage: python bai1_dft_idft.py [image_path] [size]   (mac dinh: lena.png, 64)
"""

import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image


def load_gray(path=None, size=64):
    """Read image as grayscale float array of size x size."""
    if path is None:
        # synthetic test image: rectangle + circle + gradient
        y, x = np.mgrid[0:size, 0:size]
        img = np.zeros((size, size))
        img[size // 4: size // 2, size // 4: 3 * size // 4] = 200
        img[((x - 0.65 * size) ** 2 + (y - 0.7 * size) ** 2) < (size / 6) ** 2] = 255
        img += 40 * x / size
        return img
    return np.asarray(Image.open(path).convert("L").resize((size, size)), dtype=np.float64)


def dft_matrix(n, inverse=False):
    """W[k, m] = exp(-/+ j*2*pi*k*m/n)."""
    k = np.arange(n).reshape(-1, 1)
    m = np.arange(n).reshape(1, -1)
    sign = 1 if inverse else -1
    return np.exp(sign * 2j * np.pi * k * m / n)


def dft2(f):
    """F(u,v) = sum_x sum_y f(x,y) exp(-j2pi(ux/M + vy/N)).
    Separable: F = W_M @ f @ W_N  (W symmetric)."""
    M, N = f.shape
    return dft_matrix(M) @ f.astype(np.complex128) @ dft_matrix(N)


def idft2(F):
    """f(x,y) = 1/(MN) sum_u sum_v F(u,v) exp(+j2pi(ux/M + vy/N))."""
    M, N = F.shape
    return (dft_matrix(M, True) @ F @ dft_matrix(N, True)) / (M * N)


def dft2_naive(f):
    """Direct 4-loop-free but literal version of Eq.(1), used to verify dft2."""
    M, N = f.shape
    x = np.arange(M).reshape(M, 1)
    y = np.arange(N).reshape(1, N)
    F = np.zeros((M, N), dtype=np.complex128)
    for u in range(M):
        for v in range(N):
            F[u, v] = np.sum(f * np.exp(-2j * np.pi * (u * x / M + v * y / N)))
    return F


def center_shift(F):
    """Move the zero frequency to the centre (same as fftshift, no fftshift):
    swap quadrants via index permutation."""
    M, N = F.shape
    rows = (np.arange(M) - M // 2) % M
    cols = (np.arange(N) - N // 2) % N
    return F[np.ix_(rows, cols)]


def center_unshift(F):
    """Inverse of center_shift (same as ifftshift)."""
    M, N = F.shape
    rows = (np.arange(M) + M // 2) % M
    cols = (np.arange(N) + N // 2) % N
    return F[np.ix_(rows, cols)]


def magnitude(F):
    return np.sqrt(F.real ** 2 + F.imag ** 2)          # Eq.(2)


def phase(F):
    return np.arctan2(F.imag, F.real)                   # Eq.(3)


def log_spectrum(F):
    return np.log(1 + magnitude(F))                     # Eq.(5)


def mse(a, b):
    return np.mean((a - b) ** 2)                        # Eq.(6)


def main():
    args = sys.argv[1:]
    size = int(args.pop()) if args and args[-1].isdigit() else 64
    path = args[0] if args else "lena.png"
    f = load_gray(path, size)

    F = dft2(f)                  # Eq.(1)
    Fs = center_shift(F)         # dich tam pho
    f_rec = idft2(F).real        # Eq.(4)

    if size <= 32:
        print("max |dft2 - dft2_naive| :", np.abs(F - dft2_naive(f)).max())
    print("MSE(anh goc, anh sau IDFT):", mse(f, f_rec))
    print("MSE sau khi lam tron 8-bit:", mse(f, np.round(f_rec)))
    print("|F(0,0)| =", magnitude(F)[0, 0], "| max |F| (khong tinh DC) =",
          np.delete(magnitude(F).ravel(), 0).max())

    fig, ax = plt.subplots(2, 4, figsize=(16, 9))
    fig.canvas.manager.set_window_title("Bai 1 - DFT/IDFT")

    # Hang 1 - yeu cau 6
    ax[0, 0].imshow(f, cmap="gray"); ax[0, 0].set_title("Ảnh gốc f(x,y)")
    ax[0, 1].imshow(log_spectrum(F), cmap="gray"); ax[0, 1].set_title("Phổ trước dịch tâm\nlog(1+|F|)")
    ax[0, 2].imshow(log_spectrum(Fs), cmap="gray"); ax[0, 2].set_title("Phổ sau dịch tâm\nlog(1+|F|)")
    ax[0, 3].imshow(f_rec, cmap="gray"); ax[0, 3].set_title(f"Ảnh sau IDFT (MSE={mse(f, f_rec):.2e})")

    # Hang 2 - yeu cau 3 (da dich tam de de nhin)
    ax[1, 0].imshow(Fs.real, cmap="gray"); ax[1, 0].set_title("Phần thực R(u,v)")
    ax[1, 1].imshow(Fs.imag, cmap="gray"); ax[1, 1].set_title("Phần ảo I(u,v)")
    ax[1, 2].imshow(log_spectrum(Fs), cmap="gray"); ax[1, 2].set_title("Phổ biên độ log(1+|F|)")
    ax[1, 3].imshow(phase(Fs), cmap="gray"); ax[1, 3].set_title("Phổ pha atan2(I,R)")

    for a in ax.ravel():
        a.axis("off")
    fig.tight_layout(pad=1.5)
    fig.savefig("bai1_result.png", dpi=120)
    plt.show()


if __name__ == "__main__":
    main()
