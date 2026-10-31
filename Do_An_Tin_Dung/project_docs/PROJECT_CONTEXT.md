# PROJECT CONTEXT – ĐỒ ÁN HỌC MÁY (ĐH THỦY LỢI)

## 1. Thông tin chung
- **Tên đề tài:** Dự đoán khả năng trả nợ của khách hàng vay vốn (*Credit Default Prediction*).
- **Môn học:** Học máy (*Machine Learning*).
- **Giảng viên hướng dẫn & chấm thi:** Thầy Tạ Quang Chiểu – Khoa Công nghệ thông tin, Trường Đại học Thủy Lợi.
- **Repository GitHub:** `https://github.com/lilEdgar2108/Machine_Learning`
- **Thư mục cô lập đồ án:** `Do_An_Tin_Dung/` (tách biệt hoàn toàn với `Btap TH/`, `Btap TrenLop/`, `Tai_Lieu_On_Tap/`).

## 2. Nhóm thực hiện & Phân công chính
1. **Vũ Đức Hải Nam (`2451261036` – GitHub Lead):** Phụ trách Dataset, Phân tích khám phá dữ liệu (EDA), Tiền xử lý dữ liệu chống Data Leakage, Rời rạc hóa cho ID3 (**Chương 1 & Chương 2**).
2. **Nguyễn Ngọc Thành (`2451261059`):** Phụ trách Cơ sở lý thuyết, công thức toán học và huấn luyện 4 thuật toán bắt buộc (**Chương 3: Mục 3.1 – 3.4**).
3. **Trần Thiên Vũ (`2451261069`):** Phụ trách Đánh giá thực nghiệm 8 tiêu chí, xây dựng hàm `predict_loan_status(customer_data)` và tổng hợp báo cáo Word (**Mục 3.5, Chương 4 & Kết luận**).

## 3. Dataset đã chốt
- **Tên Dataset:** `Credit Risk Dataset` (OpenML ID: `43454`).
- **Quy mô gốc:** `32,581` dòng × `12` cột (`11` đặc trưng đầu vào: `7` biến số liên tục + `4` biến phân loại; `1` biến mục tiêu).
- **Quy ước biến mục tiêu (`loan_status`):**
  - `0` = **Trả nợ tốt** (*Non-default* – lớp Negative).
  - `1` = **Vỡ nợ / Không trả được nợ** (*Default* – lớp Positive cần phát hiện rủi ro).

## 4. Bốn thuật toán bắt buộc & Nguyên tắc triển khai
1. **Logistic Regression:** Dùng `LogisticRegression` (`scikit-learn`) trên dữ liệu đã chuẩn hóa `StandardScaler`.
2. **Decision Tree – ID3:** Tự cài đặt class `ID3DecisionTree` thuần Python (tính `Entropy`, `Information Gain`, phân nhánh đa hướng trên tập dữ liệu đã rời rạc hóa `Binning`, có `max_depth` chống Overfitting và tính xác suất nút lá phục vụ vẽ ROC-AUC).
3. **Decision Tree – CART:** Dùng `DecisionTreeClassifier(criterion="gini")` (`scikit-learn`), phân nhánh nhị phân.
4. **Random Forest:** Dùng `RandomForestClassifier` (`scikit-learn`) kết hợp Bootstrap Sampling (`oob_score=True`) và lấy mẫu ngẫu nhiên tập con đặc trưng tại mỗi nút.

