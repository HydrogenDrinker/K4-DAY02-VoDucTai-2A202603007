# Hướng dẫn tái lập thực nghiệm — Lab Day 2 (DeepWeeds)

**Học viên:** Võ Đức Tài  
**MSSV:** 2A202603007  
**Lớp / Khóa:** Track 4 — Day 2: Deeplearning Advance  
**Bài toán:** Phân loại cỏ dại nông nghiệp đa lớp trên bộ dữ liệu DeepWeeds (9 lớp, 17.509 ảnh)  
**GitHub Repository:** [https://github.com/HydrogenDrinker/K4-DAY02-VoDucTai-2A202603007](https://github.com/HydrogenDrinker/K4-DAY02-VoDucTai-2A202603007)  
**Notebook Colab:** [lab_day2.ipynb](https://colab.research.google.com/github/HydrogenDrinker/K4-DAY02-VoDucTai-2A202603007/blob/main/submissions/2A202603007_VoDucTai/code/lab_day2.ipynb)  

---

## 1. Môi trường và Phiên bản Thư viện

- **Hệ điều hành:** Windows 11 (hoặc Linux Ubuntu 22.04 LTS)
- **GPU:** NVIDIA GeForce RTX 4060 Laptop GPU (8GB VRAM) / NVIDIA T4 (Kaggle/Colab)
- **CUDA:** 12.6
- **Python:** 3.10 / 3.12
- **Thư viện chính:**
  - `torch`: 2.12.0+cu126 (hoặc $\ge 2.0.0$)
  - `torchvision`: $\ge 0.15.0$
  - `timm`: 1.0.30
  - `scikit-learn`: $\ge 1.3.0$
  - `pandas`: $\ge 2.0.0$
  - `numpy`: $\ge 1.24.0$
  - `openpyxl`: $\ge 3.1.0$
  - `thop`: $\ge 0.1.1$

---

## 2. Cấu trúc Thư mục Nộp bài

```text
submissions/2A202603007_VoDucTai/
├── README.md               # Hướng dẫn cài đặt và chạy lại (file này)
├── report.md               # Báo cáo khoa học phân tích chuyên sâu
├── results.xlsx            # Bảng tổng hợp số liệu 7 sheets theo chuẩn GUIDE
├── curves/                 # Ảnh đồ thị training (loss/metric) của từng exp_id
│   ├── B01_resnet50.png
│   ├── B02_convnext_tiny.png
│   ├── T04_convnext_tiny.png
│   ├── F01_convnext_tiny.png
│   └── ...
├── predictions/            # File dự đoán test và val cho từng seed (*.csv)
│   ├── F01_seed0_test.csv
│   ├── F01_seed1_test.csv
│   ├── F01_seed2_test.csv
│   ├── T00_seed0_test.csv
│   └── ...
└── code/                   # Toàn bộ mã nguồn hoàn thiện từ starter/
    ├── dataset.py
    ├── model.py
    ├── losses.py
    ├── train.py
    ├── inference.py
    ├── benchmark.py
    └── lab_day2.ipynb      # Notebook thực nghiệm toàn diện
```

---

## 3. Thứ tự Lệnh và Cách Chạy Lại Thực Nghiệm

### Bước 1: Chuẩn bị dữ liệu và thư viện
```bash
# Cài đặt thư viện bổ sung nếu cần
pip install timm openpyxl thop scikit-learn pandas

# Tải dữ liệu ảnh DeepWeeds vào data/images (giải nén từ images.zip)
# Tải nhãn DeepWeeds vào data/labels/
```

### Bước 2: Kiểm tra tính đúng đắn pipeline
```bash
python test_sanity.py
```

### Bước 3: Chạy toàn bộ thí nghiệm (Backbone, Training Recipe, Inference, Final)
```bash
python run_experiments.py
```

### Bước 4: Chấm điểm chính thức qua công cụ `eval.py`
```bash
# Đánh giá chỉ số chi tiết cho cấu hình chung kết F01
python eval.py score --pred "submissions/2A202603007_VoDucTai/predictions/F01_seed*_test.csv" \
    --test-csv data/labels/test_subset0.csv --labels data/labels/labels.csv --tag F01 --out eval_out

# Đánh giá cấu hình mốc T00
python eval.py score --pred "submissions/2A202603007_VoDucTai/predictions/T00_seed*_test.csv" \
    --test-csv data/labels/test_subset0.csv --labels data/labels/labels.csv --tag T00 --out eval_out

# Tự chấm Phần I của RUBRIC (Chung kết so với mốc)
python eval.py grade \
    --final "submissions/2A202603007_VoDucTai/predictions/F01_seed*_test.csv" \
    --baseline "submissions/2A202603007_VoDucTai/predictions/T00_seed*_test.csv" \
    --uncal "submissions/2A202603007_VoDucTai/predictions/F01_uncal_seed*_test.csv" \
    --final-val "submissions/2A202603007_VoDucTai/predictions/F01_seed*_val.csv" \
    --latency-p95-ms 5.99 --latency-method proper \
    --test-csv data/labels/test_subset0.csv --labels data/labels/labels.csv
```

---

## 4. Danh sách Seed Đã Dùng

- **Quét sàng (Bước 1 & Bước 2):** Cố định `seed = 0` cho mọi backbone và các trục công thức huấn luyện nhằm đảm bảo so sánh công bằng.
- **Vòng chung kết (Bước 4):** Sử dụng 3 hạt giống: `seed = 0`, `seed = 1`, `seed = 2` trên cả cấu hình tối ưu (`F01`) và cấu hình mốc (`T00`). Toàn bộ chỉ số được báo cáo dưới dạng **mean ± std** với bậc tự do mẫu (`ddof=1`).
