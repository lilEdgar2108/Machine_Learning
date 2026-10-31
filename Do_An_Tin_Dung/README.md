# Đồ Án Học Máy: Dự Đoán Khả Năng Trả Nợ Của Khách Hàng Vay Vốn (Credit Default Prediction)

- **Môn học:** Học máy (Machine Learning)
- **Giảng viên hướng dẫn:** Thầy Tạ Quang Chiểu
- **Đơn vị:** Khoa Công nghệ thông tin – Trường Đại học Thủy Lợi
- **Nhóm thực hiện:**
  1. **Vũ Đức Hải Nam (`2451261036`)** – Phụ trách Dataset, EDA, Tiền xử lý chống Data Leakage & Quản lý GitHub (Chương 1 & 2)
  2. **Nguyễn Ngọc Thành (`2451261059`)** – Phụ trách Cơ sở lý thuyết & Huấn luyện 4 thuật toán: Logistic Regression, ID3, CART, Random Forest (Chương 3: 3.1 – 3.4)
  3. **Trần Thiên Vũ** – Phụ trách Đánh giá thực nghiệm Mục 3.5, Hàm dự đoán khách hàng mới & Tổng hợp báo cáo (Mục 3.5, Chương 4 & Kết luận)

---

## 1. Cấu trúc thư mục

```text
Do_An_Tin_Dung/
├── data/
│   ├── raw/                          # Dữ liệu gốc tải tự động từ OpenML (ID: 43454)
│   └── processed/                    # Tập Train/Test sau tiền xử lý & tập rời rạc hóa cho ID3
├── outputs/
│   ├── figures/                      # Toàn bộ biểu đồ EDA (Chương I) & Đánh giá (Phần 3.5)
│   ├── models/                       # Bộ tiền xử lý & 4 mô hình đã huấn luyện (.joblib)
│   └── tables/                       # Bảng thống kê mô tả & bảng so sánh kết quả (.csv)
├── src/
│   ├── 01_tai_va_kham_pha_du_lieu.py # Bước 1: Tải dữ liệu tự động & Phân tích khám phá (EDA)
│   ├── 02_tien_xu_ly_du_lieu.py      # Bước 2: Làm sạch, chia Train/Test, Impute, Encode, Scale, Binning
│   ├── id3_tu_cai_dat.py             # Thuật toán ID3 đa nhánh thuần Python (Entropy & Information Gain)
│   ├── 03_huan_luyen_mo_hinh.py      # Bước 3: Huấn luyện Logistic Regression, ID3, CART, Random Forest
│   ├── 04_danh_gia_va_so_sanh.py     # Bước 4: Đánh giá Confusion Matrix, Precision, Recall, F1, ROC-AUC
│   └── 05_du_doan_khach_hang_moi.py  # Bước 5: Hàm predict_loan_status(customer_data) & Demo thực tế
├── report/
│   └── xuat_bao_cao_docx.py          # Script sinh báo cáo Word (.docx) khớp 100% số liệu chạy thật
├── project_docs/                     # 9 tài liệu quản lý dự án, ghi chú học tập & ngân hàng câu hỏi vấn đáp
└── requirements.txt                  # Danh sách thư viện cần thiết
```

---

## 2. Hướng dẫn cài đặt và chạy chương trình (Tái lập kết quả 100%)

Mở terminal tại thư mục gốc workspace và chạy lần lượt theo thứ tự:

```powershell
# Cài đặt thư viện (nếu thiếu)
pip install -r Do_An_Tin_Dung/requirements.txt

# Chặng 1: Tải dữ liệu, EDA & Tiền xử lý dữ liệu (Chương I & II)
python Do_An_Tin_Dung/src/01_tai_va_kham_pha_du_lieu.py
python Do_An_Tin_Dung/src/02_tien_xu_ly_du_lieu.py

# Chặng 2: Huấn luyện 4 mô hình (Chương III: 3.1 - 3.4)
python Do_An_Tin_Dung/src/03_huan_luyen_mo_hinh.py

# Chặng 3: Đánh giá so sánh (Phần 3.5) & Dự đoán khách hàng mới (Chương IV)
python Do_An_Tin_Dung/src/04_danh_gia_va_so_sanh.py
python Do_An_Tin_Dung/src/05_du_doan_khach_hang_moi.py

# Chặng 4: Xuất báo cáo Word (.docx) đầy đủ 4 chương khớp 100% số liệu thật
python Do_An_Tin_Dung/report/xuat_bao_cao_docx.py
```

---

## 3. Tóm tắt kết quả thực nghiệm trên tập Kiểm thử độc lập (`N_test = 6,482`)

| Thuật toán | Train Acc | Test Acc | Precision | Recall | F1-Score | Test ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Logistic Regression** | 0.8546 | 0.8485 | 0.7229 | 0.4986 | 0.5902 | 0.8638 |
| **2. Decision Tree – ID3 (Tự cài đặt)** | 0.8826 | 0.8849 | 0.7847 | 0.6530 | 0.7129 | 0.8944 |
| **3. Decision Tree – CART** | 0.9344 | 0.9320 | 0.9544 | **0.7236** | 0.8231 | 0.9178 |
| **4. Random Forest (200 cây)** | **0.9366** | **0.9352** | **0.9845** | 0.7151 | **0.8284** | **0.9327** |


