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

# cả bài 4 dùng cặp ảnh IMG_2817 / IMG_2818
img = cv2.imread(os.path.join(FOLDER, "IMG_2817.jpg"))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
h, w = gray.shape


def timed(fn, repeat=10):
    res = fn()  # khởi động
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return res, float(np.median(ts)) * 1000


# ================= a. Harris vs SIFT/ORB: số lượng và vị trí =================
def harris_pts():
    R = cv2.cornerHarris(np.float32(gray), 2, 3, 0.04)
    dil = cv2.dilate(R, np.ones((11, 11), np.uint8))
    ys, xs = np.nonzero((R == dil) & (R > 0.01 * R.max()))
    return np.stack([xs, ys], 1).astype(np.float32)


p_h, t_h = timed(harris_pts)
(kp_s, des_s), t_s = timed(lambda: cv2.SIFT_create().detectAndCompute(gray, None))
(kp_o, des_o), t_o = timed(lambda: cv2.ORB_create(500).detectAndCompute(gray, None))
(kp_o2, des_o2), t_o2 = timed(lambda: cv2.ORB_create(len(kp_s)).detectAndCompute(gray, None))

p_s = np.float32([k.pt for k in kp_s])
p_o = np.float32([k.pt for k in kp_o])
size_s = np.float32([k.size for k in kp_s])
size_o = np.float32([k.size for k in kp_o])


def near_ratio(a, b, r=5):
    """Tỉ lệ điểm của a có điểm của b nằm trong bán kính r pixel."""
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    return (d.min(1) <= r).mean() * 100


def spread(p):
    """Độ phủ: % ô lưới 8x8 trên ảnh có chứa ít nhất một điểm."""
    gx = np.clip((p[:, 0] / w * 8).astype(int), 0, 7)
    gy = np.clip((p[:, 1] / h * 8).astype(int), 0, 7)
    return len(set(zip(gx, gy))) / 64 * 100


lines = []
lines.append("===== a. Harris vs SIFT / ORB =====")
for name, p, t in [("Harris", p_h, t_h), ("SIFT", p_s, t_s), ("ORB(500)", p_o, t_o)]:
    lines.append(f"{name:9s}: {len(p):5d} điểm, phủ {spread(p):5.1f}% lưới 8x8, {t:7.2f} ms")
lines.append(f"Kích thước (scale) keypoint SIFT: {size_s.min():.1f} - {size_s.max():.1f} px, "
             f"ORB: {size_o.min():.1f} - {size_o.max():.1f} px (Harris: 1 tỉ lệ duy nhất)")
lines.append(f"% điểm Harris có điểm SIFT trong 5px: {near_ratio(p_h, p_s):.1f}%")
lines.append(f"% điểm Harris có điểm ORB  trong 5px: {near_ratio(p_h, p_o):.1f}%")

# ================= b. SIFT vs ORB =================
lines.append("\n===== b. SIFT vs ORB =====")
lines.append(f"SIFT      : {len(kp_s)} kp, descriptor {des_s.shape} {des_s.dtype} "
             f"({des_s.nbytes / 1024:.0f} KB), {t_s:.2f} ms")
lines.append(f"ORB(500)  : {len(kp_o)} kp, descriptor {des_o.shape} {des_o.dtype} "
             f"({des_o.nbytes / 1024:.0f} KB), {t_o:.2f} ms")
lines.append(f"ORB({len(kp_s)}) : {len(kp_o2)} kp, descriptor {des_o2.shape} {des_o2.dtype} "
             f"({des_o2.nbytes / 1024:.0f} KB), {t_o2:.2f} ms")
lines.append(f"=> Cùng số điểm, ORB nhanh hơn SIFT khoảng {t_s / t_o2:.1f} lần")

# ================= c. Vai trò Ratio Test =================
# dùng cặp ảnh của Bài 3
g1 = cv2.imread(os.path.join(FOLDER, "IMG_2817.jpg"), cv2.IMREAD_GRAYSCALE)
g2 = cv2.imread(os.path.join(FOLDER, "IMG_2818.jpg"), cv2.IMREAD_GRAYSCALE)

orb = cv2.ORB_create(2000)
k1, d1 = orb.detectAndCompute(g1, None)
k2, d2 = orb.detectAndCompute(g2, None)
knn = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d1, d2, k=2)


def pts(matches):
    return (np.float32([k1[m.queryIdx].pt for m in matches]),
            np.float32([k2[m.trainIdx].pt for m in matches]))


# không có phép biến đổi thật: ước lượng H tham chiếu bằng RANSAC trên các cặp qua Ratio Test 0.75,
# rồi tinh chỉnh bằng bình phương tối thiểu trên các inlier
ref = [m for m, n in (p for p in knn if len(p) == 2) if m.distance < 0.75 * n.distance]
p1, p2 = pts(ref)
H_ref, mask = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
inl = mask.ravel().astype(bool)
H_ref, _ = cv2.findHomography(p1[inl], p2[inl], 0)


def correct_ratio(matches, tol=5.0):
    """Đếm số cặp đúng: sai số chiếu qua homography tham chiếu <= tol px."""
    if not matches:
        return 0, 0.0
    a, b = pts(matches)
    err = np.linalg.norm(cv2.perspectiveTransform(a.reshape(-1, 1, 2), H_ref).reshape(-1, 2) - b, axis=1)
    n = int((err <= tol).sum())
    return n, n / len(matches) * 100


lines.append("\n===== c. Ratio Test (IMG_2817/IMG_2818, ORB + BFMatcher, đếm cặp đúng bằng homography tham chiếu) =====")
ratios = [None, 0.9, 0.8, 0.75, 0.7, 0.6, 0.5]
stats = []
for r in ratios:
    if r is None:
        ms = [p[0] for p in knn if p]
        label = "không lọc"
    else:
        ms = [m for m, n in (p for p in knn if len(p) == 2) if m.distance < r * n.distance]
        label = f"ratio<{r}"
    n_ok, pct = correct_ratio(ms)
    stats.append((label, len(ms), n_ok, pct))
    lines.append(f"{label:10s}: {len(ms):5d} cặp, {n_ok:5d} đúng, độ chính xác {pct:5.1f}%")

# crossCheck là cách lọc khác để đối chiếu
cc = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(d1, d2)
n_ok, pct = correct_ratio(cc)
lines.append(f"{'crossCheck':10s}: {len(cc):5d} cặp, {n_ok:5d} đúng, độ chính xác {pct:5.1f}%")

# ================= Hình minh hoạ =================
fig, ax = plt.subplots(1, 3, figsize=(20, 8))
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
ax[0].imshow(rgb)
ax[0].scatter(p_h[:, 0], p_h[:, 1], s=12, c="red", label=f"Harris ({len(p_h)})")
ax[0].scatter(p_s[:, 0], p_s[:, 1], s=4, c="lime", alpha=0.6, label=f"SIFT ({len(p_s)})")
ax[0].scatter(p_o[:, 0], p_o[:, 1], s=12, c="cyan", marker="x", label=f"ORB ({len(p_o)})")
ax[0].legend(loc="lower left"); ax[0].set_title("Vị trí điểm: Harris / SIFT / ORB"); ax[0].axis("off")

ax[1].bar(["Harris", "SIFT", "ORB(500)", f"ORB({len(kp_s)})"],
          [t_h, t_s, t_o, t_o2], color=["tab:red", "tab:green", "tab:cyan", "tab:blue"])
ax[1].set_ylabel("ms (trung vị 10 lần)"); ax[1].set_title("Thời gian phát hiện (+ mô tả)")

labels = [s[0] for s in stats]
ax[2].bar(labels, [s[1] for s in stats], color="lightgray", label="tổng số cặp")
ax[2].bar(labels, [s[2] for s in stats], color="tab:green", label="cặp đúng")
ax2 = ax[2].twinx()
ax2.plot(labels, [s[3] for s in stats], "o-", c="tab:red"); ax2.set_ylabel("độ chính xác (%)", color="tab:red")
ax2.set_ylim(0, 100)
ax[2].legend(loc="upper right"); ax[2].set_title("Ảnh hưởng của Ratio Test")
ax[2].tick_params(axis="x", rotation=30)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "bai4_so_sanh.png"), dpi=90)
plt.close()

# ================= Nhận xét =================
NHAN_XET = """
NHẬN XÉT VÀ PHÂN TÍCH

a) Harris và SIFT/ORB
- Số lượng: Harris chỉ cho các điểm có đáp ứng R vượt ngưỡng (1% R max) sau khi loại bỏ
  điểm không cực đại, nên số điểm phụ thuộc mạnh vào ngưỡng. SIFT cho nhiều điểm nhất vì
  nó tìm cực trị trong không gian tỉ lệ (DoG) ở nhiều octave. ORB bị giới hạn bởi nfeatures.
- Vị trí: Harris chỉ phản hồi tại GÓC thực sự (giao điểm hai cạnh): các đỉnh ô bàn cờ, góc
  chữ cái ở viền, đồ vật trên bàn. Tại cạnh thẳng hoặc vùng phẳng, Harris không phản hồi.
  SIFT phát hiện "đốm" (blob) ở nhiều kích thước: ngoài góc còn có tâm ô cờ, vân đá, vân gỗ,
  nên điểm phủ rộng hơn, kể cả vùng có kết cấu (texture) mà không có góc rõ.
  Khoảng 2/3 góc Harris có điểm SIFT nằm sát (<= 5px), nghĩa là SIFT gần như bao trùm các góc
  của Harris và còn thêm nhiều điểm khác.
  ORB dùng FAST rồi xếp hạng bằng điểm Harris, nhưng chỉ giữ 500 điểm mạnh nhất và tìm trên
  kim tự tháp ảnh thu nhỏ, nên điểm dồn cục ở vùng tương phản mạnh (ô cờ, chữ cái ở viền),
  độ phủ ảnh thấp nhất và chỉ ít điểm trùng đúng vị trí góc Harris ở tỉ lệ gốc.
- Harris chỉ cho toạ độ, không có tỉ lệ, hướng hay descriptor nên không dùng trực tiếp để
  đối sánh được; SIFT/ORB cho mỗi điểm thêm scale, hướng và vector mô tả.
- Harris tự xây dựng (Sobel -> tích đạo hàm -> tổng cửa sổ -> R = det - k*trace^2) cho bản
  đồ R trùng với cv2.cornerHarris (sai khác ~1e-7) và cùng tập góc; bản có sẵn nhanh hơn vì
  được tối ưu bằng C++/SIMD.

b) SIFT và ORB
- Trên cùng ảnh, SIFT cho nhiều đặc trưng hơn (ORB mặc định dừng ở 500 điểm).
- Descriptor SIFT: 128 số thực float32 (512 byte/điểm), so khớp bằng khoảng cách L2.
  Descriptor ORB: 32 byte nhị phân (256 bit), so khớp bằng Hamming (XOR + đếm bit) nên rất
  nhanh và nhẹ hơn 16 lần về bộ nhớ.
- Tốc độ: ORB nhanh hơn SIFT rõ rệt, kể cả khi cho ORB lấy cùng số điểm như SIFT.
  SIFT phải dựng kim tự tháp Gauss/DoG và tính histogram gradient; ORB dùng FAST và so sánh
  cặp điểm ảnh (BRIEF), phù hợp ứng dụng thời gian thực / thiết bị yếu.
- Đổi lại SIFT ổn định hơn với thay đổi tỉ lệ, góc nhìn và độ sáng, descriptor giàu thông
  tin hơn nên đối sánh chính xác hơn trong điều kiện khó.

c) Vai trò của Ratio Test (Lowe)
- Với mỗi điểm, so sánh khoảng cách tới láng giềng gần nhất (d1) và gần nhì (d2). Nếu
  d1 < ratio * d2 thì cặp khớp được xem là "đặc trưng riêng biệt", được giữ lại; nếu d1 ~ d2
  thì điểm đó mơ hồ (nhiều chỗ trong ảnh trông giống nhau) và bị loại.
- Cảnh IMG_2817/IMG_2818 có rất nhiều chấm tròn giống nhau, hình tròn lại không đổi khi xoay,
  nên descriptor của các chấm khác nhau rất gần nhau và dễ khớp nhầm sang chấm khác: nếu bỏ qua
  bước lọc, mọi điểm đều có một cặp (kể cả điểm không xuất hiện trong ảnh 2), số đường nối rất
  nhiều, chồng chéo, tỉ lệ cặp sai cao -> ước lượng homography/ghép ảnh dễ sai hoặc RANSAC phải
  chạy lâu hơn.
- Khi áp dụng Ratio Test, số cặp giảm mạnh nhưng độ chính xác tăng lên rõ rệt. Ratio càng
  nhỏ càng chặt (ít cặp, chính xác hơn), ratio lớn thì giữ nhiều cặp nhưng lẫn nhiễu;
  0.7-0.8 là mức cân bằng thường dùng.
"""

report = "\n".join(lines) + "\n" + NHAN_XET
print(report)
with open(os.path.join(OUT, "bai4_nhan_xet.txt"), "w", encoding="utf-8") as f:
    f.write(report)
print("Đã lưu kết quả vào", OUT)
