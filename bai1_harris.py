import os
import time
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FOLDER = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(FOLDER, "output")
os.makedirs(OUT, exist_ok=True)

# ================= a. Đọc ảnh đầu vào (bàn cờ) =================
img = cv2.imread(os.path.join(FOLDER, "2026-07-02-18-32-12-576x765.jpg"))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_f = np.float32(gray)

BLOCK, KSIZE, K = 2, 3, 0.04
THRESH = 0.01  # ngưỡng tương đối so với R max


# ================= b1. Harris có sẵn của OpenCV =================
t0 = time.perf_counter()
R_cv = cv2.cornerHarris(gray_f, BLOCK, KSIZE, K)
t_cv = time.perf_counter() - t0


# ================= b2. Harris tự xây dựng =================
def harris_manual(gray, block=2, ksize=3, k=0.04):
    """R = det(M) - k * trace(M)^2, M là ma trận cấu trúc lấy tổng trong cửa sổ block x block."""
    I = gray.astype(np.float64)
    # đạo hàm theo x, y bằng Sobel
    Ix = cv2.Sobel(I, cv2.CV_64F, 1, 0, ksize=ksize)
    Iy = cv2.Sobel(I, cv2.CV_64F, 0, 1, ksize=ksize)
    Ixx, Iyy, Ixy = Ix * Ix, Iy * Iy, Ix * Iy
    # tổng các tích đạo hàm trong cửa sổ lân cận (cửa sổ hộp giống OpenCV)
    Sxx = cv2.boxFilter(Ixx, -1, (block, block), normalize=False)
    Syy = cv2.boxFilter(Iyy, -1, (block, block), normalize=False)
    Sxy = cv2.boxFilter(Ixy, -1, (block, block), normalize=False)
    det = Sxx * Syy - Sxy ** 2
    trace = Sxx + Syy
    return det - k * trace ** 2


def nms_corners(R, thresh_ratio, radius=5):
    """Ngưỡng + non-maximum suppression, trả về toạ độ (x, y) các góc."""
    dil = cv2.dilate(R, np.ones((2 * radius + 1, 2 * radius + 1), np.uint8))
    mask = (R == dil) & (R > thresh_ratio * R.max())
    ys, xs = np.nonzero(mask)
    return list(zip(xs, ys))


t0 = time.perf_counter()
R_my = harris_manual(gray_f, BLOCK, KSIZE, K)
t_my = time.perf_counter() - t0

corners_cv = nms_corners(R_cv, THRESH)
corners_my = nms_corners(R_my, THRESH)

# so sánh hai bản đồ đáp ứng (chuẩn hoá về cùng thang)
diff = np.abs(R_cv / np.abs(R_cv).max() - R_my / np.abs(R_my).max()).max()
print(f"Harris OpenCV : {len(corners_cv)} góc, thời gian {t_cv * 1000:.2f} ms")
print(f"Harris tự viết: {len(corners_my)} góc, thời gian {t_my * 1000:.2f} ms")
print(f"Sai khác lớn nhất giữa 2 bản đồ R (đã chuẩn hoá): {diff:.2e}")
common = len(set(corners_cv) & set(corners_my))
print(f"Số góc trùng vị trí: {common}")


# ================= c. Hiển thị kết quả =================
def draw(img, corners, color=(0, 0, 255)):
    out = img.copy()
    for x, y in corners:
        cv2.circle(out, (int(x), int(y)), 4, color, 2)
    return out


res_cv = draw(img, corners_cv)
res_my = draw(img, corners_my)
cv2.imwrite(os.path.join(OUT, "bai1_harris_opencv.png"), res_cv)
cv2.imwrite(os.path.join(OUT, "bai1_harris_tu_viet.png"), res_my)

fig, ax = plt.subplots(1, 4, figsize=(22, 8))
ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); ax[0].set_title("Ảnh gốc")
ax[1].imshow(cv2.cvtColor(res_cv, cv2.COLOR_BGR2RGB))
ax[1].set_title(f"cv2.cornerHarris: {len(corners_cv)} góc")
ax[2].imshow(cv2.cvtColor(res_my, cv2.COLOR_BGR2RGB))
ax[2].set_title(f"Harris tự xây dựng: {len(corners_my)} góc")
ax[3].imshow(np.log1p(np.maximum(R_my, 0)), cmap="jet")
ax[3].set_title("Bản đồ đáp ứng R (log, R>0)")
for a in ax:
    a.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "bai1_harris_so_sanh.png"), dpi=90)
plt.close()
print("Đã lưu kết quả vào", OUT)
