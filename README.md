# XLA - Bài thực hành 5: Phát hiện, mô tả và đối sánh đặc trưng

Harris (có sẵn và tự cài đặt), SIFT, ORB và đối sánh đặc trưng bằng BFMatcher + Ratio Test của Lowe. Ảnh thử nghiệm: `2026-07-02-18-32-12-576x765.jpg` (bàn cờ) cho Bài 1, 2; cặp ảnh `IMG_2817.jpg`, `IMG_2818.jpg` cho Bài 3, 4.

| File | Nội dung |
|---|---|
| `bai1_harris.py` | Bài 1: Harris của OpenCV và Harris tự xây dựng, so sánh kết quả |
| `bai2_sift_orb.py` | Bài 2: SIFT và ORB, vẽ keypoint, số keypoint và kích thước descriptors |
| `bai3_matching.py` | Bài 3: ORB + BFMatcher + Ratio Test, vẽ đường nối các cặp khớp, đánh giá bằng RANSAC homography |
| `bai4_nhan_xet.py` | Bài 4: số liệu so sánh Harris/SIFT/ORB, ảnh hưởng của Ratio Test, nhận xét |
| `output/` | Hình kết quả và `bai4_nhan_xet.txt` |

Chạy (cần `opencv-python` >= 4.4 hoặc `opencv-contrib-python`, `numpy`, `matplotlib`):

```bash
python bai1_harris.py
python bai2_sift_orb.py
python bai3_matching.py
python bai4_nhan_xet.py
```
