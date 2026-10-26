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

img = cv2.imread(os.path.join(FOLDER, "2026-07-02-18-32-12-576x765.jpg"))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def run(detector, name, repeat=10):
    # lần gọi đầu có chi phí khởi tạo nên chạy khởi động trước, lấy trung vị nhiều lần đo
    kp, des = detector.detectAndCompute(gray, None)
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        detector.detectAndCompute(gray, None)
        times.append(time.perf_counter() - t0)
    dt = float(np.median(times))
    print(f"[{name}] Số keypoint: {len(kp)}")
    print(f"[{name}] Kích thước ma trận descriptors: {des.shape}, kiểu {des.dtype}")
    print(f"[{name}] Thời gian detectAndCompute: {dt * 1000:.2f} ms")
    # DRAW_RICH_KEYPOINTS vẽ cả kích thước (scale) và hướng của mỗi keypoint
    vis = cv2.drawKeypoints(img, kp, None, color=(0, 255, 0),
                            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite(os.path.join(OUT, f"bai2_{name.lower()}.png"), vis)
    return kp, des, dt, vis


# ================= a. SIFT =================
sift = cv2.SIFT_create()
kp_s, des_s, t_s, vis_s = run(sift, "SIFT")

# ================= b. ORB =================
# ORB giới hạn số keypoint bằng nfeatures (mặc định 500), SIFT thì không giới hạn
orb = cv2.ORB_create(nfeatures=500)
kp_o, des_o, t_o, vis_o = run(orb, "ORB")

fig, ax = plt.subplots(1, 3, figsize=(18, 8))
ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); ax[0].set_title("Ảnh gốc")
ax[1].imshow(cv2.cvtColor(vis_s, cv2.COLOR_BGR2RGB))
ax[1].set_title(f"SIFT: {len(kp_s)} kp, des {des_s.shape}, {t_s * 1000:.0f} ms")
ax[2].imshow(cv2.cvtColor(vis_o, cv2.COLOR_BGR2RGB))
ax[2].set_title(f"ORB: {len(kp_o)} kp, des {des_o.shape}, {t_o * 1000:.0f} ms")
for a in ax:
    a.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "bai2_sift_orb_so_sanh.png"), dpi=90, bbox_inches="tight")
plt.close()
print("Đã lưu kết quả vào", OUT)
