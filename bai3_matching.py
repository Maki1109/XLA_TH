import os
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FOLDER = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(FOLDER, "output")
os.makedirs(OUT, exist_ok=True)

# ================= a. Hai ảnh cùng cảnh, khác góc nhìn =================
img1 = cv2.imread(os.path.join(FOLDER, "IMG_2817.jpg"))
img2 = cv2.imread(os.path.join(FOLDER, "IMG_2818.jpg"))
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


# không có phép biến đổi thật nên đánh giá bằng RANSAC homography (tỉ lệ inlier)
def ransac(matches):
    if len(matches) < 4:
        return None, np.zeros(len(matches), bool)
    p1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    p2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    H, mask = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
    return H, mask.ravel().astype(bool)


H_all, in_all = ransac(all_best)
H_good, in_good = ransac(good)
print(f"Không lọc   : {in_all.sum()}/{len(all_best)} inlier ({in_all.mean() * 100:.1f}%)")
print(f"Ratio Test  : {in_good.sum()}/{len(good)} inlier ({in_good.mean() * 100:.1f}%)")

# phép biến đổi giữa hai ảnh suy ra từ homography của các cặp tốt
A = H_good[:2, :2] / H_good[2, 2]
angle = np.degrees(np.arctan2(A[1, 0], A[0, 0]))
scale = np.sqrt(abs(np.linalg.det(A)))
print(f"Homography ước lượng:\n{np.array2string(H_good / H_good[2, 2], precision=4)}")
print(f"=> xoay {angle:.1f} độ, tỉ lệ {scale:.3f}")

# ================= e. Hiển thị đường nối các cặp đối sánh =================
draw = dict(singlePointColor=None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
vis_all = cv2.drawMatches(img1, kp1, img2, kp2, all_best, None, matchColor=(0, 0, 255), **draw)
vis_good = cv2.drawMatches(img1, kp1, img2, kp2, good, None, matchColor=(0, 255, 0), **draw)
# tô đỏ các cặp tốt nhưng vẫn bị RANSAC loại
outliers = [m for m, ok in zip(good, in_good) if not ok]
vis_good = cv2.drawMatches(img1, kp1, img2, kp2, outliers, vis_good, matchColor=(0, 0, 255),
                           flags=cv2.DrawMatchesFlags_DRAW_OVER_OUTIMG |
                           cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imwrite(os.path.join(OUT, "bai3_all_matches.png"), vis_all)
cv2.imwrite(os.path.join(OUT, "bai3_good_matches.png"), vis_good)

# kiểm tra: chiếu khung ảnh 1 sang ảnh 2 bằng homography ước lượng
h, w = g1.shape
box = cv2.perspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2), H_good)
warp = cv2.warpPerspective(img1, H_good, (img2.shape[1], img2.shape[0]))
overlay = cv2.addWeighted(img2, 0.5, warp, 0.5, 0)
cv2.polylines(overlay, [np.int32(box)], True, (0, 255, 255), 2)
cv2.imwrite(os.path.join(OUT, "bai3_overlay.png"), overlay)

fig, ax = plt.subplots(3, 1, figsize=(12, 18))
ax[0].imshow(cv2.cvtColor(vis_all, cv2.COLOR_BGR2RGB))
ax[0].set_title(f"BFMatcher không lọc: {len(all_best)} cặp, inlier {in_all.mean() * 100:.1f}%")
ax[1].imshow(cv2.cvtColor(vis_good, cv2.COLOR_BGR2RGB))
ax[1].set_title(f"Sau Ratio Test (<{RATIO}): {len(good)} cặp, inlier {in_good.mean() * 100:.1f}% "
                f"(đỏ: bị RANSAC loại)")
ax[2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
ax[2].set_title(f"Ảnh 1 chiếu sang ảnh 2 (xoay {angle:.1f}°, tỉ lệ {scale:.2f}) chồng lên ảnh 2")
for a in ax:
    a.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "bai3_so_sanh.png"), dpi=80)
plt.close()
print("Đã lưu kết quả vào", OUT)
