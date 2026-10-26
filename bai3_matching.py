import os
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FOLDER = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(FOLDER, "output")
os.makedirs(OUT, exist_ok=True)

# ================= a. Chuẩn bị hai ảnh cùng cảnh, khác góc nhìn =================
img1 = cv2.imread(os.path.join(FOLDER, "2026-07-02-18-32-12-576x765.jpg"))
h, w = img1.shape[:2]

path2 = os.path.join(FOLDER, "anh2.jpg")
if os.path.exists(path2):
    # nếu có ảnh thật chụp từ góc khác thì dùng ảnh đó
    img2 = cv2.imread(path2)
else:
    # mô phỏng góc chụp khác: biến đổi phối cảnh + xoay + thu nhỏ + đổi độ sáng
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[0.10 * w, 0.05 * h], [0.95 * w, 0.0], [0.85 * w, 0.95 * h], [0.0, 0.85 * h]])
    H_persp = cv2.getPerspectiveTransform(src, dst)
    R = np.vstack([cv2.getRotationMatrix2D((w / 2, h / 2), 20, 0.85), [0, 0, 1]])
    H_true = R @ H_persp
    img2 = cv2.warpPerspective(img1, H_true, (w, h), borderMode=cv2.BORDER_CONSTANT)
    img2 = cv2.convertScaleAbs(img2, alpha=1.1, beta=-15)
    cv2.imwrite(os.path.join(OUT, "bai3_anh2_mo_phong.png"), img2)

g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# ================= b. ORB: keypoints + descriptors =================
orb = cv2.ORB_create(nfeatures=2000)
kp1, des1 = orb.detectAndCompute(g1, None)
kp2, des2 = orb.detectAndCompute(g2, None)
print(f"Ảnh 1: {len(kp1)} keypoint, Ảnh 2: {len(kp2)} keypoint")

# ================= c. BFMatcher (Hamming vì descriptor ORB là nhị phân) =================
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
knn = bf.knnMatch(des1, des2, k=2)
all_best = [p[0] for p in knn if len(p) >= 1]
print(f"Số cặp đối sánh ban đầu (láng giềng gần nhất): {len(all_best)}")

# ================= d. Ratio Test của Lowe =================
RATIO = 0.75
good = [m for m, n in (p for p in knn if len(p) == 2) if m.distance < RATIO * n.distance]
print(f"Số cặp tốt sau Ratio Test ({RATIO}): {len(good)}")


# đánh giá độ chính xác bằng RANSAC homography (tỉ lệ inlier)
def inlier_ratio(matches):
    if len(matches) < 4:
        return 0, 0.0
    p1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    p2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    _, mask = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
    n = int(mask.sum())
    return n, n / len(matches)


n_all, r_all = inlier_ratio(all_best)
n_good, r_good = inlier_ratio(good)
print(f"Không lọc   : {n_all}/{len(all_best)} inlier ({r_all * 100:.1f}%)")
print(f"Ratio Test  : {n_good}/{len(good)} inlier ({r_good * 100:.1f}%)")

# ================= e. Hiển thị đường nối các cặp đối sánh =================
good_sorted = sorted(good, key=lambda m: m.distance)
vis_good = cv2.drawMatches(img1, kp1, img2, kp2, good_sorted, None,
                           matchColor=(0, 255, 0), singlePointColor=None,
                           flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
vis_all = cv2.drawMatches(img1, kp1, img2, kp2, all_best, None,
                          matchColor=(0, 0, 255), singlePointColor=None,
                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imwrite(os.path.join(OUT, "bai3_good_matches.png"), vis_good)
cv2.imwrite(os.path.join(OUT, "bai3_all_matches.png"), vis_all)

fig, ax = plt.subplots(2, 1, figsize=(14, 18))
ax[0].imshow(cv2.cvtColor(vis_all, cv2.COLOR_BGR2RGB))
ax[0].set_title(f"BFMatcher không lọc: {len(all_best)} cặp, inlier {r_all * 100:.1f}%")
ax[1].imshow(cv2.cvtColor(vis_good, cv2.COLOR_BGR2RGB))
ax[1].set_title(f"Sau Ratio Test (<{RATIO}): {len(good)} cặp, inlier {r_good * 100:.1f}%")
for a in ax:
    a.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "bai3_so_sanh.png"), dpi=80)
plt.close()
print("Đã lưu kết quả vào", OUT)
