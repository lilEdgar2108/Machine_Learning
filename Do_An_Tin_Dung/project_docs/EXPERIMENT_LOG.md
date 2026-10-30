# EXPERIMENT LOG – NHẬT KÝ SỐ LIỆU THỰC NGHIỆM THẬT 100%

## 1. Số liệu Khám phá dữ liệu thô (Bước 01 – Chương I)
- **Nguồn tải:** `sklearn.datasets.fetch_openml(data_id=43454)` -> `data/raw/credit_risk_raw.csv`.
- **Kích thước gốc:** `32,581` dòng × `12` cột (`7` biến số + `4` biến phân loại + `1` biến mục tiêu `loan_status`).
- **Phân bố nhãn `loan_status` gốc:**
  - Nhãn `0` (*Trả nợ tốt – Non-default*): `25,473` mẫu (`78.18%`).
  - Nhãn `1` (*Vỡ nợ – Default*): `7,108` mẫu (`21.82%`).
  - Tỷ lệ mất cân bằng lớp: `3.58 : 1`.
- **Phát hiện vấn đề dữ liệu thực tế:**
  - Bản ghi trùng lặp hoàn toàn: `165` dòng.
  - Ngoại lai vô lý về tuổi (`person_age > 100`): `5` dòng (giá trị `123` và `144` tuổi).
  - Ngoại lai vô lý về thâm niên làm việc (`person_emp_length > 60` hoặc `> person_age - 14`): `2` dòng (`max = 123.0` năm làm việc).
  - Dữ liệu thiếu (`Missing Values`):
    - `person_emp_length`: thiếu `895` dòng (`2.75%`).
    - `loan_int_rate`: thiếu `3,116` dòng (`9.56%`).
- **Quy luật nghiệp vụ nổi bật từ EDA:**
  - Tỷ lệ vỡ nợ tăng đơn điệu hoàn hảo theo `loan_grade`: Hạng `A` (`10.0%`) -> `B` (`16.3%`) -> `C` (`20.7%`) -> `D` (`59.0%`) -> `E` (`64.4%`) -> `F` (`70.5%`) -> `G` (`98.4%`).
  - Khách thuê nhà `RENT` có tỷ lệ vỡ nợ cao nhất (`31.6%`), khách sở hữu nhà `OWN` thấp nhất (`7.5%`).
  - Khách từng có lịch sử nợ xấu (`Y`) vỡ nợ `37.8%`, cao gấp đôi nhóm chưa từng (`N`: `18.4%`).

## 2. Số liệu Tiền xử lý dữ liệu (Bước 02 – Chương II)
- **Sau xóa trùng lặp (`-165` dòng):** `32,416` dòng.
- **Sau lọc ngoại lai vô lý (`-7` dòng):** `32,409` dòng hợp lệ.
- **Chia Stratified Train / Test (`test_size = 0.20`, `random_state = 42`):**
  - Tập `Train` (`80%`): `25,927` mẫu — Nhãn `1` (Vỡ nợ) = `5,670` mẫu (`21.87%`).
  - Tập `Test` (`20%`): `6,482` mẫu — Nhãn `1` (Vỡ nợ) = `1,418` mẫu (`21.88%`).
- **Tham số điền khuyết `Median` học từ tập `Train`:**
  - `person_emp_length`: `4.0` năm.
  - `loan_int_rate`: `10.99` %/năm.
- **Số chiều đặc trưng đầu ra:**
  - Cho `Logistic Regression` (đã `StandardScaler`) và `CART / Random Forest` (giữ đơn vị gốc): `19` cột.
  - Cho `ID3` (rời rạc hóa 7 biến số theo tứ phân vị tập `Train` + giữ 4 biến phân loại): `11` cột phân loại.

## 3. Số liệu Khảo sát 5-Fold CV & Huấn luyện 4 Mô hình (Bước 03 – Chương III: 3.1 – 3.4)
- **Khảo sát chọn độ sâu cắt tỉa bằng 5-Fold Stratified CV trên tập Train (`25,927` mẫu):**
  - **ID3 tự cài đặt:** Đạt đỉnh tại `max_depth = 4` (`CV_Val_ROC_AUC = 0.8843`, `CV_Val_Acc = 0.8760`). Khi tăng lên `5, 6, 8, 11`, `Train_ROC_AUC` tăng lên `0.9498` nhưng `CV_Val_ROC_AUC` giảm xuống `0.8643` do phân nhánh đa hướng làm mảnh hóa nút lá.
  - **CART:** Đạt đỉnh tại `max_depth = 10` (`CV_Val_ROC_AUC = 0.9140`, `CV_Val_Acc = 0.9263`). Khi bỏ giới hạn (`max_depth = None`), `Train_ROC_AUC = 0.9805` nhưng `CV_Val_ROC_AUC` tụt xuống `0.8936` (Overfitting rõ rệt).
- **Thông số 4 mô hình sau huấn luyện trên toàn bộ tập Train:**
  1. **Logistic Regression (`logistic_regression.joblib`):**
     - Hệ số chặn $b = -1.5989$; thời gian train `0.0231s`.
     - Top trọng số: `person_home_ownership_OWN` ($w = -1.8633$, giảm rủi ro), `loan_percent_income` ($w = +1.4039$, tăng rủi ro), `loan_grade` ($w = +1.1122$, tăng rủi ro).
  2. **Decision Tree – ID3 tự cài đặt (`id3_decision_tree.joblib`):**
     - Entropy ban đầu tập Train: $H(S) = 0.7578\text{ bit}$.
     - Nút gốc: `loan_grade` ($\text{IG} = 0.1078\text{ bit}$), tiếp theo là `loan_percent_income` (`0.0758 bit`), `loan_int_rate` (`0.0616 bit`).
     - Cấu trúc cây (`max_depth = 4`): `477` nút tổng cộng, `370` nút lá; thời gian train `0.3323s`.
  3. **Decision Tree – CART (`cart_decision_tree.joblib`):**
     - Nút gốc: `loan_percent_income <= 0.305` ($\text{Gini} = 0.342$).
     - Cấu trúc cây (`max_depth = 10`, `min_samples_leaf = 10`): `140` nút lá; thời gian train `0.0591s`.
  4. **Random Forest (`random_forest.joblib`):**
     - `200` cây, `max_features = 'sqrt'` ($4$ đặc trưng/nút), `max_depth = 12`, `min_samples_leaf = 5`.
     - **Điểm OOB Accuracy (`36.8%` mẫu ngoài túi Bootstrap): `0.9320` (`93.20%`)**; thời gian train `0.7498s`.
     - Top 5 Feature Importance: `loan_percent_income` (`0.2678`), `loan_grade` (`0.1566`), `person_income` (`0.1345`), `loan_int_rate` (`0.0997` — cao gấp 11 lần so với 1 cây CART đơn lẻ `0.0090` nhờ Feature Subspace!), `person_home_ownership_RENT` (`0.0717`).

## 4. Số liệu Đánh giá trên Tập Test Độc Lập (`N_test = 6,482`: `5,064` Trả tốt, `1,418` Vỡ nợ) – Phần 3.5

### Bảng tổng hợp 4 mô hình (Threshold = 0.5):
| Mô hình | Train Acc | Test Acc | Precision | Recall | F1-Score | Train AUC | Test AUC | TN | FP | FN | TP |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Logistic Regression** | 0.8546 | 0.8485 | 0.7229 | 0.4986 | 0.5902 | 0.8639 | 0.8638 | 4,793 | 271 | 711 | 707 |
| **2. Decision Tree – ID3** | 0.8826 | 0.8849 | 0.7847 | 0.6530 | 0.7129 | 0.9057 | 0.8944 | 4,810 | 254 | 492 | 926 |
| **3. Decision Tree – CART** | 0.9344 | 0.9320 | 0.9544 | **0.7236** | 0.8231 | 0.9350 | 0.9178 | 5,015 | 49 | **392** | **1,026** |
| **4. Random Forest** | 0.9366 | **0.9352** | **0.9845** | 0.7151 | **0.8284** | 0.9632 | **0.9327** | **5,048** | **16** | 404 | 1,014 |

### Điểm nhấn phân tích thực nghiệm:
- **Kiểm chứng Accuracy Paradox:** Mô hình ngây thơ luôn đoán `0` (duyệt hết) vẫn đạt `Accuracy = 78.12%` nhưng `Recall = 0.00%` (`FN = 1,418`).
- **Kiểm chứng lý thuyết OOB Chương 8:** Điểm `OOB Score` trên tập Train của Random Forest (`0.9320`) xấp xỉ cực sát `Test Accuracy` thực tế (`0.9352`, chỉ lệch `0.0032`).
- **So sánh tinh tế CART vs Random Forest:** Ở ngưỡng `0.5`, CART bắt nhiều hơn Random Forest `12` ca vỡ nợ (`Recall 72.36%` vs `71.51%`), nhưng Random Forest giảm số khách tốt bị từ chối nhầm (`FP`) từ `49` xuống chỉ còn **`16` người (`Precision = 98.45%`)** và dẫn đầu toàn diện về **`F1-Score (0.8284)`** và **`ROC-AUC (0.9327)`**.

## 5. Kết quả Chạy thử Hàm `predict_loan_status(customer_data)` (Chương IV)
- **Hồ sơ 1 (Khách tốt – Hạng A, nhà OWN, vay 9% thu nhập):** Cả 4 mô hình đều dự đoán `0 - Trả nợ tốt` với xác suất vỡ nợ cực thấp (`Logistic: 0.21%`, `ID3: 1.31%`, `CART: 0.87%`, `Random Forest: 0.63%`).
- **Hồ sơ 2 (Khách rủi ro cao – Hạng E, nhà RENT, vay 42% thu nhập đảo nợ):** Cả 4 mô hình đều bắt trúng `1 - Nguy cơ vỡ nợ` với xác suất gần tuyệt đối (`Logistic: 99.14%`, `ID3: 100.00%`, `CART: 100.00%`, `Random Forest: 98.71%`).
- **Hồ sơ 3 (Khách khuyết thông tin lãi suất & năm đi làm):** Pipeline tự động điền Median tập Train (`4.0` năm, `10.99%`) và dự đoán không lỗi (`Random Forest: 9.86%` -> `0 - Trả nợ tốt`).

