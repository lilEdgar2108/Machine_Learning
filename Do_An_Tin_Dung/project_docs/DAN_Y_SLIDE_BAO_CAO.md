# DÀN Ý CHI TIẾT SLIDE BÁO CÁO ĐỒ ÁN HỌC MÁY (20 SLIDES CHUẨN BẢO VỆ)

- **Đề tài:** Dự đoán khả năng trả nợ của khách hàng vay vốn (*Credit Default Prediction*)
- **Giảng viên hướng dẫn & chấm thi:** Thầy Tạ Quang Chiểu – Khoa CNTT, Trường Đại học Thủy Lợi
- **Thời lượng thuyết trình chuẩn:** 15 – 18 phút (+ 10 – 15 phút vấn đáp)
- **Phân bổ 3 thành viên (mỗi người ~6–7 slides, nắm trọn 1 phần xuyên suốt):**
  1. **Vũ Đức Hải Nam (`2451261036`):** **Slide 1 – Slide 7** (Mở đầu, Bài toán, Dataset, EDA, Quy trình Tiền xử lý chống Data Leakage & Rời rạc hóa cho ID3).
  2. **Nguyễn Ngọc Thành (`2451261059`):** **Slide 8 – Slide 13** (Cơ sở toán học, Chọn siêu tham số bằng 5-Fold CV, Kết quả huấn luyện 4 thuật toán: Logistic Regression, ID3 tự cài đặt, CART, Random Forest).
  3. **Trần Thiên Vũ (`2451261069`):** **Slide 14 – Slide 20** (Đánh giá & So sánh thực nghiệm trên tập Test độc lập, Confusion Matrix, ROC-AUC, Demo hàm `predict_loan_status`, Kết luận & Hướng phát triển).

---

## TỔNG QUAN BỐ CỤC 20 SLIDES & PHÂN CÔNG

| STT | Tiêu đề Slide | Phụ trách | Chương tương ứng | File Hình / Bảng chèn từ `outputs/` |
| :-: | :--- | :---: | :---: | :--- |
| **01** | Trang bìa (Title Slide) | **Hải Nam** | Mở đầu | Logo ĐH Thủy Lợi (`report/template/logo_dhtl.png`) |
| **02** | Nội dung trình bày & Phân công nhóm | **Hải Nam** | Mở đầu | Sơ đồ 4 chương + Bảng phân công 3 thành viên |
| **03** | Đặt vấn đề & Mục tiêu bài toán tín dụng | **Hải Nam** | Chương 1 | Minh họa bài toán phân lớp nhị phân ($y \in \{0, 1\}$) |
| **04** | Tổng quan bộ dữ liệu `Credit Risk Dataset` | **Hải Nam** | Chương 1 | `tables/01_kiem_tra_chat_luong_du_lieu.csv` |
| **05** | Phân tích khám phá dữ liệu (EDA) & Phát hiện bất thường | **Hải Nam** | Chương 1 | `figures/01_phan_bo_nhan_loan_status.png` + `figures/01_ty_le_vo_no_theo_nhom.png` |
| **06** | Quy trình Tiền xử lý chống Rò rỉ dữ liệu (*Anti-Data Leakage*) | **Hải Nam** | Chương 2 | `tables/02_tong_ket_tien_xu_ly.csv` + `figures/02_minh_hoa_tien_xu_ly.png` |
| **07** | Thiết kế 3 nhánh dữ liệu đặc thù cho 4 thuật toán | **Hải Nam** | Chương 2 | `tables/02_tham_so_imputation_va_scaler.csv` + `tables/02_nguong_roi_rac_hoa_id3.csv` |
| **08** | Thuật toán 1: Hồi quy Logistic (*Logistic Regression*) | **Thành** | Chương 3 (3.1) | `tables/03_he_so_logistic_regression.csv` |
| **09** | Thuật toán 2: Cây quyết định đa nhánh ID3 (Tự cài đặt Python) | **Thành** | Chương 3 (3.2) | `figures/03_information_gain_nut_goc_id3.png` + `tables/03_ig_nut_goc_id3.csv` |
| **10** | Cấu trúc nhánh đầu tiên của Cây quyết định ID3 | **Thành** | Chương 3 (3.2) | `tables/03_so_do_nhanh_dau_cay_id3.txt` |
| **11** | Thuật toán 3: Cây quyết định nhị phân CART (*Gini Index*) | **Thành** | Chương 3 (3.3) | `figures/03_cay_quyet_dinh_cart_minh_hoa.png` |
| **12** | Kiểm soát Overfitting cho ID3 & CART bằng 5-Fold Stratified CV | **Thành** | Chương 3 (3.2–3.3) | `figures/03_chon_do_sau_chong_overfitting.png` + `tables/03_khao_sat_do_sau_cv.csv` |
| **13** | Thuật toán 4: Rừng ngẫu nhiên (*Random Forest*) & Kiểm chứng OOB | **Thành** | Chương 3 (3.4) | `figures/03_top10_feature_importance_rf.png` + `tables/03_feature_importance_cart_rf.csv` |
| **14** | Thiết lập thực nghiệm & Hệ thống 6 độ đo đánh giá (Tại sao cần Recall/F1?) | **Vũ** | Chương 3 (3.5) | Giải thích *Accuracy Paradox* trên tập Test ($N = 6.482$) |
| **15** | Bảng tổng hợp kết quả thực nghiệm 4 mô hình trên tập Test | **Vũ** | Chương 3 (3.5) | `tables/04_bang_so_sanh_4_mo_hinh.csv` + `figures/04_so_sanh_5_chi_so_test.png` |
| **16** | Phân tích Ma trận nhầm lẫn (*Confusion Matrix*) dưới góc độ Ngân hàng | **Vũ** | Chương 3 (3.5) | `figures/04_confusion_matrix_4_mo_hinh.png` + `tables/04_chi_tiet_confusion_matrix_test.csv` |
| **17** | Phân tích Đường cong ROC & Chỉ số AUC của 4 mô hình | **Vũ** | Chương 3 (3.5) | `figures/04_roc_curve_so_sanh_4_mo_hinh.png` |
| **18** | Xây dựng Pipeline đóng gói & Hàm dự đoán `predict_loan_status` | **Vũ** | Chương 4 | Sơ đồ luồng `preprocessor_bundle.joblib` + 4 model `.joblib` |
| **19** | Kết quả chạy thử nghiệm 3 kịch bản hồ sơ khách hàng mới | **Vũ** | Chương 4 | `tables/05_ket_qua_du_doan_ho_so_mau.csv` |
| **20** | Kết luận, Khuyến nghị chọn mô hình & Hướng phát triển | **Vũ** | Kết luận | Tóm tắt 4 kết luận cốt lõi |

---

## CHI TIẾT NỘI DUNG TỪNG SLIDE (KÈM SỐ LIỆU THẬT 100% & GỢI Ý THUYẾT TRÌNH)

---

### PHẦN 1: MỞ ĐẦU, KHÁM PHÁ DỮ LIỆU & TIỀN XỬ LÝ (PHỤ TRÁCH: VŨ ĐỨC HẢI NAM – `2451261036`)

#### SLIDE 01: TRANG BÌA (TITLE SLIDE)
- **Nội dung hiển thị trên Slide:**
  - **TRƯỜNG ĐẠI HỌC THỦY LỢI – KHOA CÔNG NGHỆ THÔNG TIN**
  - **BÁO CÁO ĐỒ ÁN MÔN HỌC: HỌC MÁY (*MACHINE LEARNING*)**
  - **Đề tài:** **DỰ ĐOÁN KHẢ NĂNG TRẢ NỢ CỦA KHÁCH HÀNG VAY VỐN (*CREDIT DEFAULT PREDICTION*)**
  - **Giảng viên hướng dẫn:** TS. Tạ Quang Chiểu
  - **Nhóm sinh viên thực hiện:**
    1. Vũ Đức Hải Nam – MSV: `2451261036`
    2. Nguyễn Ngọc Thành – MSV: `2451261059`
    3. Trần Thiên Vũ – MSV: `2451261069`
  - **Hà Nội, Năm 2026**
- **Hình chèn:** Logo Trường ĐH Thủy Lợi (`Do_An_Tin_Dung/report/template/logo_dhtl.png`).

---

#### SLIDE 02: NỘI DUNG TRÌNH BÀY & PHÂN CÔNG NHIỆM VỤ
- **Nội dung hiển thị trên Slide:**
  - **Cấu trúc báo cáo 4 chương:**
    1. **Chương 1:** Tổng quan bài toán & Phân tích khám phá dữ liệu (EDA).
    2. **Chương 2:** Tiền xử lý dữ liệu chuẩn học thuật (Chống *Data Leakage*).
    3. **Chương 3:** Huấn luyện & Đánh giá thực nghiệm 4 thuật toán (*Logistic Regression, ID3, CART, Random Forest*).
    4. **Chương 4:** Xây dựng chương trình dự đoán khách hàng mới & Kết luận.
  - **Phân công thực hiện:**
    - **Vũ Đức Hải Nam (`2451261036`):** Chương 1 & Chương 2 (`01_tai_va_kham_pha_du_lieu.py`, `02_tien_xu_ly_du_lieu.py`, quản lý GitHub).
    - **Nguyễn Ngọc Thành (`2451261059`):** Chương 3 – Mục 3.1 đến 3.4 (`id3_tu_cai_dat.py`, `03_huan_luyen_mo_hinh.py`).
    - **Trần Thiên Vũ (`2451261069`):** Chương 3 – Mục 3.5, Chương 4 & Tổng hợp Báo cáo (`04_danh_gia_va_so_sanh.py`, `05_du_doan_khach_hang_moi.py`).

---

#### SLIDE 03: ĐẶT VẤN ĐỀ & MỤC TIÊU BÀI TOÁN TÍN DỤNG
- **Nội dung hiển thị trên Slide:**
  - **Bài toán thực tế tại Ngân hàng:**
    - Khi khách hàng nộp hồ sơ vay vốn, ngân hàng đối mặt với **2 loại rủi ro đối nghịch**:
      1. **Bỏ lọt nợ xấu (*False Negative – FN*):** Duyệt nhầm khách hàng mất khả năng trả nợ $\rightarrow$ **Mất vốn gốc + lãi**.
      2. **Từ chối nhầm khách tốt (*False Positive – FP*):** Quá khắt khe làm mất khách hàng uy tín $\rightarrow$ **Mất chi phí cơ hội kinh doanh**.
  - **Phát biểu bài toán Học máy có giám sát (*Supervised Learning*):**
    - **Bài toán:** Phân lớp nhị phân (*Binary Classification*).
    - **Đầu vào ($\mathbf{x}$):** Vector đặc trưng nhân khẩu học, tài chính và lịch sử tín dụng của khách hàng.
    - **Đầu ra ($y = \text{loan\_status}$):**
      - $y = 0$: **Trả nợ tốt** (*Non-default* – lớp Negative).
      - $y = 1$: **Vỡ nợ / Rủi ro cao** (*Default* – lớp Positive cần ưu tiên phát hiện).

---

#### SLIDE 04: TỔNG QUAN BỘ DỮ LIỆU `CREDIT RISK DATASET`
- **Nội dung hiển thị trên Slide:**
  - **Nguồn dữ liệu:** `Credit Risk Dataset` (OpenML ID: `43454`).
  - **Quy mô ban đầu:** **`32.581` bản ghi** $\times$ **`12` cột** (`11` đặc trưng đầu vào + `1` biến mục tiêu `loan_status`).
  - **Cấu trúc 11 đặc trưng đầu vào:**
    - **7 biến số liên tục (*Numerical*):**
      1. `person_age`: Tuổi khách hàng (năm).
      2. `person_income`: Thu nhập hàng năm (USD).
      3. `person_emp_length`: Thâm niên làm việc (năm) — *Thiếu `895` dòng (`2,75%`)*.
      4. `loan_amnt`: Số tiền đăng ký vay (USD).
      5. `loan_int_rate`: Lãi suất vay (%/năm) — *Thiếu `3.116` dòng (`9,56%`)*.
      6. `loan_percent_income`: Tỷ lệ tiền vay / thu nhập hàng năm.
      7. `cb_person_cred_hist_length`: Số năm lịch sử tín dụng.
    - **4 biến phân loại (*Categorical*):**
      8. `person_home_ownership`: Tình trạng nhà ở (`RENT`, `MORTGAGE`, `OWN`, `OTHER` – Định danh).
      9. `loan_intent`: Mục đích vay (`EDUCATION`, `MEDICAL`, `VENTURE`, `PERSONAL`, `DEBTCONSOLIDATION`, `HOMEIMPROVEMENT` – Định danh).
      10. `loan_grade`: Hạng tín dụng từ `A` (tốt nhất) $\rightarrow$ `G` (rủi ro nhất) – **Biến thứ tự (*Ordinal*)**.
      11. `cb_person_default_on_file`: Lịch sử từng vỡ nợ (`N` / `Y` – Nhị phân).
- **Bảng tham chiếu:** [`01_kiem_tra_chat_luong_du_lieu.csv`](file:///c:/Users/vuhai/Machine_Learning_By_Hnam/Do_An_Tin_Dung/outputs/tables/01_kiem_tra_chat_luong_du_lieu.csv).

---

#### SLIDE 05: PHÂN TÍCH KHÁM PHÁ DỮ LIỆU (EDA) & PHÁT HIỆN BẤT THƯỜNG
- **Nội dung hiển thị trên Slide:**
  - **1. Mất cân bằng lớp (*Class Imbalance*):**
    - Nhãn `0` (Trả nợ tốt): **`25.473` mẫu (`78,18%`)**.
    - Nhãn `1` (Vỡ nợ): **`7.108` mẫu (`21,82%`)** $\rightarrow$ Tỷ lệ mất cân bằng **`3,58 : 1`**.
  - **2. Ba vấn đề chất lượng dữ liệu phát hiện qua EDA:**
    - **Trùng lặp:** Có **`165` bản ghi** trùng lặp hoàn toàn.
    - **Ngoại lai sai logic (*Impossible Outliers*):** **`7` bản ghi** vô lý (`5` dòng có tuổi `123`–`144`; `2` dòng có thâm niên làm việc `123` năm).
    - **Khuyết dữ liệu (*Missing Values*):** Chỉ rơi vào `person_emp_length` (`2,75%`) và `loan_int_rate` (`9,56%`).
  - **3. Quy luật nghiệp vụ nổi bật:**
    - Tỷ lệ vỡ nợ tăng đơn điệu theo hạng `loan_grade`: Hạng `A` (**`10,0%`**) $\rightarrow$ `B` (**`16,3%`**) $\rightarrow$ `C` (**`20,7%`**) $\rightarrow$ `D` (**`59,0%`**) $\rightarrow$ `E` (**`64,4%`**) $\rightarrow$ `F` (**`70,5%`**) $\rightarrow$ `G` (**`98,4%`**).
    - Khách thuê nhà (`RENT`) vỡ nợ **`31,6%`** (cao gấp 4 lần khách sở hữu nhà `OWN`: **`7,5%`**).
- **Hình chèn lên Slide (Đặt 2 ảnh cạnh nhau):**
  - Trái: [`01_phan_bo_nhan_loan_status.png`](file:///c:/Users/vuhai/Machine_Learning_By_Hnam/Do_An_Tin_Dung/outputs/figures/01_phan_bo_nhan_loan_status.png)
  - Phải: [`01_ty_le_vo_no_theo_nhom.png`](file:///c:/Users/vuhai/Machine_Learning_By_Hnam/Do_An_Tin_Dung/outputs/figures/01_ty_le_vo_no_theo_nhom.png)

---

#### SLIDE 06: QUY TRÌNH TIỀN XỬ LÝ CHỐNG RÒ RỈ DỮ LIỆU (*ANTI-DATA LEAKAGE*)
- **Nội dung hiển thị trên Slide:**
  - **Bước 1 – Làm sạch dữ liệu (*Data Cleaning*):**
    - Xóa `165` dòng trùng lặp $\rightarrow$ còn `32.416` dòng.
    - Lọc `7` dòng ngoại lai sai logic (`person_age > 100` hoặc `person_emp_length > 60`) $\rightarrow$ còn **`32.409` bản ghi sạch**.
  - **Bước 2 – Chia tập Train/Test phân tầng (*Stratified Split 80/20*) TRƯỚC KHI tính thống kê:**
    - **Tập Huấn luyện (`Train` – 80%):** **`25.927` mẫu** (Nhãn `1` = `5.670` mẫu, chiếm **`21,87%`**).
    - **Tập Kiểm tra (`Test` – 20%):** **`6.482` mẫu** (Nhãn `1` = `1.418` mẫu, chiếm **`21,88%`**).
  - **Bước 3 – Điền giá trị khuyết bằng Trung vị (*Median*) chỉ học từ tập `Train`:**
    - Tại sao dùng **Median** thay vì **Mean**? Vì thu nhập, lãi suất và thâm niên bị **lệch phải (*Right-skewed*)**, Median không bị kéo lệch bởi các giá trị cực đại.
    - `Median(person_emp_length)_Train` = **`4,0` năm**; `Median(loan_int_rate)_Train` = **`10,99` %/năm**.
- **Hình & Bảng chèn lên Slide:**
  - Bảng: [`02_tong_ket_tien_xu_ly.csv`](file:///c:/Users/vuhai/Machine_Learning_By_Hnam/Do_An_Tin_Dung/outputs/tables/02_tong_ket_tien_xu_ly.csv)
  - Hình: [`02_minh_hoa_tien_xu_ly.png`](file:///c:/Users/vuhai/Machine_Learning_By_Hnam/Do_An_Tin_Dung/outputs/figures/02_minh_hoa_tien_xu_ly.png)
- **Điểm nhấn trả lời thầy Chiểu:** *"Nếu nhóm gọi `fillna()` hoặc `StandardScaler.fit_transform()` trên toàn bộ 32.409 dòng trước rồi mới chia Train/Test thì thông tin phân phối của tập Test đã bị lộ vào tập Train (Data Leakage). Vì vậy nhóm bắt buộc `split` trước, mọi tham số `Median, Mean, Std, Quartile` chỉ được `.fit()` trên `25.927` dòng Train."*

---

#### SLIDE 07: THIẾT KẾ 3 NHÁNH DỮ LIỆU ĐẶC THÙ CHO 4 THUẬT TOÁN
- **Nội dung hiển thị trên Slide:**
  - **Mã hóa biến phân loại (*Categorical Encoding*):**
    - **Ordinal Encoding:** `loan_grade` (`A` $\rightarrow 0$, `B` $\rightarrow 1$, ..., `G` $\rightarrow 6$) để bảo toàn thứ tự rủi ro; `cb_person_default_on_file` (`N` $\rightarrow 0$, `Y` $\rightarrow 1`).
    - **One-Hot Encoding:** `person_home_ownership` (4 cột) và `loan_intent` (6 cột) vì không có thứ tự lớn nhỏ $\rightarrow$ Tổng cộng **`19` đặc trưng số**.
  - **Tách 3 định dạng đầu vào khớp bản chất toán học từng mô hình:**
    1. **Nhánh 1 – Cho `Logistic Regression` (`19` cột chuẩn hóa Z-score):**
       - Áp dụng `StandardScaler`: $z = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}}}$ để đưa `person_income` (hàng chục nghìn USD) và `loan_percent_income` (`0.0`–`0.8`) về cùng thang đo $\rightarrow$ Hội tụ nhanh và công bằng khi phạt L2.
    2. **Nhánh 2 – Cho `CART` & `Random Forest` (`19` cột giữ nguyên đơn vị gốc):**
       - Cây nhị phân xét ngưỡng bất biến với phép biến đổi đơn điệu, giữ nguyên đơn vị gốc để đọc trực tiếp ngưỡng trên cây (ví dụ: `loan_percent_income <= 0.305`).
    3. **Nhánh 3 – Cho `ID3 tự cài đặt` (`11` cột rời rạc hóa *Quantile Binning* 4 khoảng):**
       - Thuật toán ID3 gốc chỉ hoạt động trên biến rời rạc $\rightarrow$ Chia 7 biến số liên tục thành 4 khoảng tứ phân vị ($Q_1, Q_2, Q_3$ học từ tập `Train`): `Thap`, `Trung_binh`, `Kha_cao`, `Cao`.
- **Bảng tham chiếu:** [`02_nguong_roi_rac_hoa_id3.csv`](file:///c:/Users/vuhai/Machine_Learning_By_Hnam/Do_An_Tin_Dung/outputs/tables/02_nguong_roi_rac_hoa_id3.csv).

---

### PHẦN 2: CƠ SỞ LÝ THUYẾT & HUẤN LUYỆN 4 THUẬT TOÁN (PHỤ TRÁCH: NGUYỄN NGỌC THÀNH – `2451261059`)

#### SLIDE 08: THUẬT TOÁN 1 – HỒI QUY LOGISTIC (*LOGISTIC REGRESSION*)
- **Nội dung hiển thị trên Slide:**
  - **Cơ sở toán học (Chương 6 bài giảng):**
    - Kết hợp tuyến tính: $z = \mathbf{w}^T \mathbf{x} + b = \sum_{j=1}^{19} w_j x_j + b$.
    - Hàm Sigmoid ánh xạ sang xác suất vỡ nợ:
      $$\hat{p} = P(y = 1 \mid \mathbf{x}) = \sigma(z) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}$$
    - Tối ưu hàm mất mát **Binary Cross-Entropy + điều chuẩn L2 ($C = 1.0$, solver `lbfgs`)**.
  - **Kết quả huấn luyện trên tập Train (`25.927` mẫu, thời gian `0,0231s`):**
    - Hệ số chặn: $b = -1,5989$.
    - **Top đặc trưng làm TĂNG nguy cơ vỡ nợ ($w > 0$):**
      - `loan_percent_income`: $w = +1,4039$ ($\text{Odds Ratio } e^w = 4,07$).
      - `loan_grade`: $w = +1,1122$ ($\text{Odds Ratio } e^w = 3,04$).
      - `person_home_ownership_RENT`: $w = +0,5253$ ($\text{Odds Ratio } e^w = 1,69$).
    - **Top đặc trưng làm GIẢM nguy cơ vỡ nợ ($w < 0$):**
      - `person_home_ownership_OWN`: $w = -1,8633$ ($\text{Odds Ratio } e^w = 0,16$).
- **Bảng chèn lên Slide:** Top 8 dòng từ [`03_he_so_logistic_regression.csv`](file:///c:/Users/vuhai/Machine_Learning_By_Hnam/Do_An_Tin_Dung/outputs/tables/03_he_so_logistic_regression.csv).

---

#### SLIDE 09: THUẬT TOÁN 2 – CÂY QUYẾT ĐỊNH ĐA NHÁNH ID3 (TỰ CÀI ĐẶT PYTHON THUẦN)
- **Nội dung hiển thị trên Slide:**
  - **Tại sao nhóm tự cài đặt class `ID3DecisionTree` (`id3_tu_cai_dat.py`)?**
    - `DecisionTreeClassifier` của `scikit-learn` chỉ cài đặt thuật toán **CART chia nhị phân (2 nhánh)**, không phải ID3 đa nhánh của Quinlan (1986).
    - Nhóm tự cài đặt **ID3 đa nhánh thuần Python** theo đúng công thức Chương 5 của thầy Chiểu:
  - **Công thức Entropy & Information Gain (log cơ số 2 – đơn vị bit):**
    $$H(S) = -\sum_{c \in \{0, 1\}} p_c \log_2(p_c), \qquad IG(S, A) = H(S) - \sum_{v \in \text{Values}(A)} \frac{|S_v|}{|S|} H(S_v)$$
  - **Kết quả tính toán tại Nút gốc tập Train ($N = 25.927$):**
    - Entropy gốc: **$H(S) = 0,7578\text{ bit}$** (với $p_1 = 21,87\%$, $p_0 = 78,13\%$).
    - **Top 5 thuộc tính có Information Gain lớn nhất tại Nút gốc:**
      1. **`loan_grade`:** $\text{IG} = \mathbf{0,1078\text{ bit}}$ ($H_{\text{còn lại}} = 0,6500\text{ bit}$) $\rightarrow$ **Được chọn làm Nút gốc!**
      2. `loan_percent_income`: $\text{IG} = 0,0758\text{ bit}$.
      3. `loan_int_rate`: $\text{IG} = 0,0616\text{ bit}$.
      4. `person_income`: $\text{IG} = 0,0529\text{ bit}$.
      5. `person_home_ownership`: $\text{IG} = 0,0448\text{ bit}$.
- **Hình chèn lên Slide:** [`03_information_gain_nut_goc_id3.png`](file:///c:/Users/vuhai/Machine_Learning_By_Hnam/Do_An_Tin_Dung/outputs/figures/03_information_gain_nut_goc_id3.png).

---

#### SLIDE 10: CẤU TRÚC NHÁNH ĐẦU TIÊN CỦA CÂY QUYẾT ĐỊNH ID3
- **Nội dung hiển thị trên Slide:**
  - **Cấu trúc phân nhánh đa hướng từ Nút gốc `loan_grade` (`7` nhánh A $\rightarrow$ G):**
    ```text
    [Nút gốc: loan_grade | H = 0.7578 bit | IG = 0.1078 bit | n = 25,927]
      ├── Nhánh 'A' (n=8,530 | P(Vỡ nợ)=10.0%) ──> Hỏi tiếp: loan_percent_income (IG=0.0832)
      ├── Nhánh 'B' (n=8,281 | P(Vỡ nợ)=16.4%) ──> Hỏi tiếp: loan_percent_income (IG=0.0878)
      ├── Nhánh 'C' (n=5,221 | P(Vỡ nợ)=20.3%) ──> Hỏi tiếp: loan_percent_income (IG=0.0774)
      ├── Nhánh 'D' (n=2,890 | P(Vỡ nợ)=59.1%) ──> Hỏi tiếp: loan_intent         (IG=0.1873)
      ├── Nhánh 'E' (n=  755 | P(Vỡ nợ)=65.4%) ──> Hỏi tiếp: loan_intent         (IG=0.2240)
      ├── Nhánh 'F' (n=  196 | P(Vỡ nợ)=70.4%) ──> Hỏi tiếp: loan_intent         (IG=0.2176)
      └── Nhánh 'G' (n=   54 | P(Vỡ nợ)=98.1%) ──> Hỏi tiếp: loan_percent_income (IG=0.0729)
    ```
  - **Ý nghĩa nghiệp vụ rất thú vị mà cây ID3 tự phát hiện:**
    - Với khách hàng **hạng tốt (`A, B, C`)**: Câu hỏi quan trọng tiếp theo là **Tỷ lệ vay trên thu nhập (`loan_percent_income`)**.
    - Với khách hàng **hạng rủi ro (`D, E, F`)**: Câu hỏi quan trọng tiếp theo chuyển sang **Mục đích vay vốn (`loan_intent`)** (vay đảo nợ/y tế rủi ro hơn vay giáo dục/kinh doanh).
  - **Thông số cây ID3 hoàn chỉnh (`max_depth = 4`, `min_samples_split = 25`):** Tổng **`477` nút**, **`370` nút lá**, thời gian huấn luyện `0,3323s`.

---

#### SLIDE 11: THUẬT TOÁN 3 – CÂY QUYẾT ĐỊNH NHỊ PHÂN CART (*GINI INDEX*)
- **Nội dung hiển thị trên Slide:**
  - **Sự khác biệt cốt lõi giữa CART và ID3:**
    1. **Độ đo tạp chất:** Dùng **Chỉ số Gini (*Gini Impurity*)** thay vì Entropy (không tốn phép tính $\log_2$):
       $$\text{Gini}(S) = 1 - \sum_{c \in \{0, 1\}} p_c^2, \qquad \text{Gini}_{\text{split}}(S, A, t) = \frac{|S_L|}{|S|}\text{Gini}(S_L) + \frac{|S_R|}{|S|}\text{Gini}(S_R)$$
    2. **Cách chia nhánh:** Luôn **chia nhị phân (Đúng / Sai)** tại ngưỡng tối ưu $x_j \le t$, xử lý trực tiếp biến số liên tục và cho phép **1 thuộc tính xuất hiện nhiều lần** ở các tầng khác nhau.
  - **Kết quả huấn luyện cây CART (`max_depth = 10`, `min_samples_leaf = 10`):**
    - **Nút gốc:** `loan_percent_income <= 0.305` ($\text{Gini} = 0,342$) $\rightarrow$ Nếu vay vượt quá `30,5%` thu nhập năm, nguy cơ vỡ nợ tăng vọt.
    - **Quy mô cây:** `140` nút lá (gọn hơn ID3 dù sâu hơn vì chỉ tách 2 nhánh/nút), thời gian train `0,0591s`.
- **Hình chèn lên Slide:** [`03_cay_quyet_dinh_cart_minh_hoa.png`](file:///c:/Users/vuhai/Machine_Learning_By_Hnam/Do_An_Tin_Dung/outputs/figures/03_cay_quyet_dinh_cart_minh_hoa.png) (3 tầng đầu tiên của cây CART).

---

#### SLIDE 12: KIỂM SOÁT OVERFITTING CHO ID3 & CART BẰNG 5-FOLD STRATIFIED CV
- **Nội dung hiển thị trên Slide:**
  - **Vấn đề:** Nếu để cây phát triển tự do (`max_depth = None`), cây sẽ học thuộc lòng nhiễu trên tập Train (*Overfitting*).
  - **Giải pháp chuẩn học thuật:** Dùng **5-Fold Stratified Cross-Validation ngay trên tập Train (`25.927` mẫu)** (tuyệt đối không chạm vào tập Test) để chọn `max_depth` tối ưu:
    - **Đối với ID3 đa nhánh:**
      - Đạt đỉnh Validation tại **`max_depth = 4`** (`CV Val ROC-AUC = 0,8843`, `CV Val Acc = 87,60%`).
      - Khi tăng độ sâu lên `6, 8, 11`: `Train AUC` vọt lên `0,9498` nhưng `Val AUC` tụt xuống `0,8643` do phân nhánh đa hướng làm mảnh hóa nhanh các nút lá.
    - **Đối với CART nhị phân:**
      - Đạt đỉnh Validation tại **`max_depth = 10`** (`CV Val ROC-AUC = 0,9140`, `CV Val Acc = 92,63%`).
      - Nếu không giới hạn (`None`): `Train AUC = 0,9805` nhưng `Val AUC` giảm còn `0,8936`.
- **Hình chèn lên Slide:** [`03_chon_do_sau_chong_overfitting.png`](file:///c:/Users/vuhai/Machine_Learning_By_Hnam/Do_An_Tin_Dung/outputs/figures/03_chon_do_sau_chong_overfitting.png).

---

#### SLIDE 13: THUẬT TOÁN 4 – RỪNG NGẪU NHIÊN (*RANDOM FOREST*) & KIỂM CHỨNG OOB
- **Nội dung hiển thị trên Slide:**
  - **Bản chất Học kết hợp (*Ensemble Learning – Bagging*, Chương 8):**
    - Huấn luyện **`B = 200` cây quyết định CART độc lập** (`max_depth = 12`, `min_samples_leaf = 5`).
    - **Hai tầng ngẫu nhiên hóa giúp giảm Phương sai (*Variance*) và phá vỡ tương quan giữa các cây:**
      1. **Bootstrap Sampling:** Mỗi cây lấy mẫu có hoàn lại $N$ bản ghi từ tập Train ($\approx 63,2\%$ mẫu nằm trong túi, **$36,8\%$ mẫu nằm ngoài túi – *Out-Of-Bag / OOB***).
      2. **Feature Subspace (`max_features = 'sqrt'`):** Tại mỗi nút chỉ xét ngẫu nhiên $\lfloor\sqrt{19}\rfloor = \mathbf{4}$ đặc trưng.
  - **Hai phát hiện thực nghiệm đắt giá:**
    1. **Kiểm chứng lý thuyết OOB:** Điểm **`OOB Accuracy = 93,20%`** xấp xỉ cực sát với **`Test Accuracy = 93,52%`** (chỉ lệch `0,32%`).
    2. **Phá vỡ sự thống trị của biến mạnh:** Ở 1 cây CART đơn lẻ, `loan_int_rate` bị che khuất (Importance chỉ `0,0090`), nhưng trong Random Forest nhờ chỉ bốc 4 biến/nút, `loan_int_rate` vươn lên Top 4 với Importance **`0,0997` (tăng gấp 11 lần)**!
- **Hình chèn lên Slide:** [`03_top10_feature_importance_rf.png`](file:///c:/Users/vuhai/Machine_Learning_By_Hnam/Do_An_Tin_Dung/outputs/figures/03_top10_feature_importance_rf.png).

---

### PHẦN 3: ĐÁNH GIÁ THỰC NGHIỆM, CHƯƠNG TRÌNH DỰ ĐOÁN & KẾT LUẬN (PHỤ TRÁCH: TRẦN THIÊN VŨ – `2451261069`)

#### SLIDE 14: THIẾT LẬP ĐÁNH GIÁ & TẠI SAO KHÔNG CHỈ DÙNG ACCURACY?
- **Nội dung hiển thị trên Slide:**
  - **Quy mô tập Kiểm tra độc lập (`Test Set` – 20%):** **`N = 6.482` khách hàng** (`5.064` Trả nợ tốt – `78,12%`; `1.418` Vỡ nợ – `21,88%`).
  - **Nghịch lý Độ chính xác (*Accuracy Paradox*):**
    - Nếu một mô hình "ngây thơ" dự đoán **100% khách hàng đều trả nợ tốt ($\hat{y} = 0$)**:
    - Nó vẫn đạt **`Accuracy = 5.064 / 6.482 = 78,12%`**, nhưng **`Recall = 0%`** (bỏ lọt toàn bộ `1.418` khoản nợ xấu $\rightarrow$ Ngân hàng phá sản!).
  - **Hệ thống 5 chỉ số đánh giá toàn diện:**
    - **Accuracy:** Tỷ lệ dự đoán đúng tổng thể.
    - **Precision ($\frac{TP}{TP + FP}$):** Trong số các hồ sơ bị mô hình cảnh báo vỡ nợ, bao nhiêu % thực sự vỡ nợ? (*Cao $\rightarrow$ Ít từ chối nhầm khách tốt*).
    - **Recall ($\frac{TP}{TP + FN}$):** Trong tổng số `1.418` khách vỡ nợ thực tế, mô hình bắt trúng được bao nhiêu %? (*Cao $\rightarrow$ Ít bỏ lọt nợ xấu*).
    - **F1-Score:** Trung bình điều hòa cân bằng giữa Precision và Recall: $F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$.
    - **ROC-AUC:** Khả năng phân tách hai lớp trên mọi ngưỡng xác suất $T \in [0, 1]$.

---

#### SLIDE 15: BẢNG TỔNG HỢP KẾT QUẢ 4 MÔ HÌNH TRÊN TẬP TEST (`N = 6.482`)
- **Nội dung hiển thị trên Slide:**
  - **Bảng so sánh thực nghiệm (Ngưỡng phân lớp $T = 0,5$):**

| Mô hình | Train Acc | Test Acc | Precision | Recall | F1-Score | Train AUC | Test AUC | Độ lệch AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Logistic Regression** | 85,46% | 84,85% | 72,29% | 49,86% | 0,5902 | 0,8639 | 0,8638 | 0,0001 |
| **2. Decision Tree – ID3** | 88,26% | 88,49% | 78,47% | 65,30% | 0,7129 | 0,9057 | 0,8944 | 0,0113 |
| **3. Decision Tree – CART** | 93,44% | 93,20% | 95,44% | **72,36%** | 0,8231 | 0,9350 | 0,9178 | 0,0172 |
| **4. Random Forest** | 93,66% | **93,52%** | **98,45%** | 71,51% | **0,8284** | 0,9632 | **0,9327** | 0,0305 |

  - **Nhận xét cốt lõi:**
    - Cả 4 mô hình đều vượt xa mức cơ sở `78,12%` và **không bị Overfitting** (độ lệch Train – Test AUC chỉ từ `0,0001` đến `0,0305`).
    - Các mô hình phi tuyến dựa trên cây (`ID3, CART, Random Forest`) vượt trội hơn hẳn ranh giới siêu phẳng tuyến tính của `Logistic Regression`.
    - **`Random Forest` dẫn đầu toàn diện** trên 3 chỉ số tổng hợp quan trọng nhất: **Test Accuracy (`93,52%`)**, **Precision (`98,45%`)**, **F1-Score (`0,8284`)** và **ROC-AUC (`0,9327`)**.
- **Hình chèn lên Slide:** [`04_so_sanh_5_chi_so_test.png`](file:///c:/Users/vuhai/Machine_Learning_By_Hnam/Do_An_Tin_Dung/outputs/figures/04_so_sanh_5_chi_so_test.png).

---

#### SLIDE 16: PHÂN TÍCH MA TRẬN NHẦM LẪN (*CONFUSION MATRIX*) DƯỚI GÓC ĐỘ NGÂN HÀNG
- **Nội dung hiển thị trên Slide:**
  - **Chi tiết 4 ô `TN, FP, FN, TP` trên `6.482` khách hàng tập Test (`5.064` Khách tốt, `1.418` Nợ xấu):**

| Mô hình | Duyệt đúng khách tốt (`TN`) | Từ chối nhầm khách tốt (`FP`) | Bỏ lọt nợ xấu (`FN`) | Bắt trúng nợ xấu (`TP`) | Tỷ lệ từ chối nhầm (`FPR`) | Tỷ lệ bỏ lọt nợ xấu (`FNR`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Logistic Regression** | 4.793 | 271 | 711 | 707 | 5,35% | 50,14% |
| **2. Decision Tree – ID3** | 4.810 | 254 | 492 | 926 | 5,02% | 34,70% |
| **3. Decision Tree – CART** | 5.015 | 49 | **392** | **1.026** | 0,97% | **27,64%** |
| **4. Random Forest** | **5.048** | **16** | 404 | 1.014 | **0,32%** | 28,49% |

  - **Ý nghĩa kinh tế thực tiễn:**
    - `Logistic Regression` bỏ lọt tới `711/1.418` ca nợ xấu (`50,14%`) do ranh giới tuyến tính không bắt được các cụm tương tác phi tuyến.
    - `ID3 tự cài đặt` giảm số ca bỏ lọt xuống còn `492` ca (bắt thêm được `219` khoản nợ xấu so với Logistic).
    - **So sánh tinh tế `CART` vs `Random Forest`:** Ở ngưỡng `0,5`, `CART` bắt nhiều hơn `Random Forest` `12` ca nợ xấu (`1.026` vs `1.014`), nhưng **`Random Forest` chỉ từ chối nhầm đúng `16` khách hàng tốt trên tổng số `5.064` người (`FPR = 0,32%`, giảm 3 lần sai lầm loại I so với CART)**!
- **Hình chèn lên Slide:** [`04_confusion_matrix_4_mo_hinh.png`](file:///c:/Users/vuhai/Machine_Learning_By_Hnam/Do_An_Tin_Dung/outputs/figures/04_confusion_matrix_4_mo_hinh.png).

---

#### SLIDE 17: PHÂN TÍCH ĐƯỜNG CONG ROC & CHỈ SỐ AUC CỦA 4 MÔ HÌNH
- **Nội dung hiển thị trên Slide:**
  - **Ý nghĩa Đường cong ROC (*Receiver Operating Characteristic*):**
    - Biểu diễn sự đánh đổi giữa **True Positive Rate ($\text{TPR} = \text{Recall}$)** và **False Positive Rate ($\text{FPR}$)** khi quét ngưỡng xác suất $T$ từ $0 \rightarrow 1$.
  - **Thứ tự diện tích dưới đường cong (ROC-AUC trên tập Test):**
    1. **Random Forest:** **`AUC = 0,9327`** (Đường cong bao trùm cao nhất ở góc trên bên trái).
    2. **Decision Tree – CART:** **`AUC = 0,9178`**.
    3. **Decision Tree – ID3:** **`AUC = 0,8944`** (Tính xác suất từ tỷ lệ mẫu lớp `1` tại nút lá: $\hat{P}(y=1|L) = \frac{n_{1,L}}{n_L}$).
    4. **Logistic Regression:** **`AUC = 0,8638`**.
  - **Ứng dụng điều chỉnh ngưỡng ($T$) cho Ngân hàng:**
    - Với `Random Forest` (`AUC = 0,9327`), nếu ngân hàng muốn ưu tiên an toàn vốn trong giai đoạn kinh tế biến động, chỉ cần hạ ngưỡng quyết định từ $T = 0,50$ xuống $T = 0,35$ để đẩy `Recall` vượt trên `80%` mà vẫn giữ `Precision` ở mức rất cao.
- **Hình chèn lên Slide:** [`04_roc_curve_so_sanh_4_mo_hinh.png`](file:///c:/Users/vuhai/Machine_Learning_By_Hnam/Do_An_Tin_Dung/outputs/figures/04_roc_curve_so_sanh_4_mo_hinh.png).

---

#### SLIDE 18: XÂY DỰNG PIPELINE ĐÓNG GÓI & HÀM DỰ ĐOÁN `predict_loan_status`
- **Nội dung hiển thị trên Slide:**
  - **Mục tiêu Chương 4 (`05_du_doan_khach_hang_moi.py`):** Đưa mô hình đã huấn luyện vào thẩm định tự động hồ sơ khách hàng mới trong thực tế mà không cần huấn luyện lại.
  - **Kiến trúc đóng gói `.joblib` (`outputs/models/`):**
    - `preprocessor_bundle.joblib`: Lưu trữ toàn bộ tham số học từ tập Train (`median_values`, `ordinal_maps`, `scaler`, `id3_bin_edges`, danh sách 19 cột).
    - 4 file mô hình: `logistic_regression.joblib`, `id3_decision_tree.joblib`, `cart_decision_tree.joblib`, `random_forest.joblib`.
  - **Luồng xử lý của hàm `predict_loan_status(customer_data)`:**
    1. **Nhận đầu vào:** Dictionary thông tin thô của 1 khách hàng (`11` thuộc tính).
    2. **Tự động tiền xử lý đồng bộ tập Train:** Điền khuyết bằng `Median_Train` (nếu thiếu) $\rightarrow$ Mã hóa Ordinal & One-Hot (`reindex` đúng 19 cột) $\rightarrow$ Chia 3 nhánh (`StandardScaler` cho Logistic / Giữ gốc cho CART & RF / Rời rạc hóa theo `id3_bin_edges` cho ID3).
    3. **Xuất kết quả thẩm định:** Nhãn dự đoán (`0`/`1`), Xác suất vỡ nợ (%), Mức độ rủi ro (`THẤP` / `TRUNG BÌNH` / `CAO`) và Khuyến nghị tín dụng.

---

#### SLIDE 19: KẾT QUẢ CHẠY THỬ NGHIỆM 3 KỊCH BẢN HỒ SƠ KHÁCH HÀNG MỚI
- **Nội dung hiển thị trên Slide:**
  - **Bảng kết quả thực thi hàm `predict_loan_status(customer_data)`:**

| Kịch bản Hồ sơ kiểm thử | Logistic Regression | Decision Tree – ID3 | Decision Tree – CART | Random Forest (Khuyên dùng) | Quyết định thẩm định |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Hồ sơ 1 – Khách hàng tốt:**<br>30 tuổi, thu nhập `85.000 USD`, nhà `OWN`, vay `8.000 USD` (`9%` thu nhập) cho `EDUCATION`, hạng `A`, chưa từng vỡ nợ | Nhãn `0`<br>(`0,21%`) | Nhãn `0`<br>(`1,31%`) | Nhãn `0`<br>(`0,87%`) | **Nhãn `0`**<br>(**`0,63%`** – Rủi ro THẤP) | **0 – TRẢ NỢ TỐT**<br>*(Đề xuất: DUYỆT KHOẢN VAY)* |
| **Hồ sơ 2 – Khách hàng rủi ro cao:**<br>22 tuổi, thu nhập `24.000 USD`, nhà `RENT`, vay `10.000 USD` (`42%` thu nhập) để `DEBTCONSOLIDATION`, hạng `E`, từng vỡ nợ `Y` | Nhãn `1`<br>(`99,14%`) | Nhãn `1`<br>(`100,00%`) | Nhãn `1`<br>(`100,00%`) | **Nhãn `1`**<br>(**`98,71%`** – Rủi ro CAO) | **1 – NGUY CƠ VỠ NỢ**<br>*(Đề xuất: TỪ CHỐI / YÊU CẦU TSĐB)* |
| **Hồ sơ 3 – Hồ sơ khuyết thông tin:**<br>26 tuổi, thu nhập `48.000 USD`, nhà `MORTGAGE`, vay `9.600 USD` (`20%` thu nhập), hạng `B`, **khuyết `loan_int_rate` & `person_emp_length`** | Nhãn `0`<br>(`15,12%`) | Nhãn `0`<br>(`5,36%`) | Nhãn `0`<br>(`5,50%`) | **Nhãn `0`**<br>(**`9,86%`** – Rủi ro THẤP) | **0 – TRẢ NỢ TỐT**<br>*(Tự động Impute Median: `4,0` năm & `10,99%`)* |

- **Bảng tham chiếu:** [`05_ket_qua_du_doan_ho_so_mau.csv`](file:///c:/Users/vuhai/Machine_Learning_By_Hnam/Do_An_Tin_Dung/outputs/tables/05_ket_qua_du_doan_ho_so_mau.csv).

---

#### SLIDE 20: KẾT LUẬN, KHUYẾN NGHỊ & HƯỚNG PHÁT TRIỂN
- **Nội dung hiển thị trên Slide:**
  - **1. Kết luận tổng kết đồ án:**
    - Đã xây dựng thành công quy trình Học máy khép kín, **chống tuyệt đối rò rỉ dữ liệu (*Anti-Data Leakage*)** trên bộ dữ liệu `32.409` hồ sơ tín dụng sạch.
    - Đã cài đặt và huấn luyện đúng bản chất toán học cả **4 thuật toán bắt buộc**, trong đó tự cài đặt thành công thuật toán **ID3 đa nhánh thuần Python** (`Test Accuracy = 88,49%`, `ROC-AUC = 0,8944`).
  - **2. Khuyến nghị lựa chọn mô hình triển khai:**
    - **Mô hình chính thức:** **Random Forest (`Test Acc = 93,52%`, `Precision = 98,45%`, `F1 = 0,8284`, `ROC-AUC = 0,9327`)** — Hạn chế tối đa việc từ chối nhầm khách hàng tốt (`FP = 16/5.064`).
    - **Mô hình giải thích luật minh bạch:** Kết hợp **CART (`Recall = 72,36%`)** và **ID3** để cung cấp lý do từ chối rõ ràng cho chuyên viên tín dụng.
  - **3. Hướng phát triển tiếp theo:**
    - Tối ưu ngưỡng quyết định theo ma trận chi phí tài chính (*Cost-sensitive Threshold Tuning*).
    - Thử nghiệm các thuật toán *Gradient Boosting* (`XGBoost`, `LightGBM`) và tích hợp giải thích cục bộ `SHAP Values`.
  - **Lời cảm ơn:** *"Nhóm chúng em xin chân thành cảm ơn TS. Tạ Quang Chiểu đã tận tình giảng dạy và hướng dẫn! Nhóm xin phép lắng nghe câu hỏi nhận xét từ Thầy ạ."*

---

## MẸO THIẾT KẾ SLIDE ĐỂ GHI ĐIỂM TUYỆT ĐỐI VỚI THẦY CHIỂU
1. **Không copy nguyên đoạn văn dài lên Slide:** Mỗi slide chỉ để tối đa **4–6 gạch đầu dòng**, chữ cỡ **18–24 pt**, công thức toán viết to, rõ ràng.
2. **Tận dụng tối đa 12 biểu đồ `.png` (`dpi=180`) trong `Do_An_Tin_Dung/outputs/figures/`:** Toàn bộ biểu đồ đã có sẵn tiêu đề tiếng Việt, nhãn trục rõ nét và màu sắc đồng bộ.
3. **Thống nhất màu sắc nhấn mạnh trên bảng số liệu:** Tô đậm màu xanh lá hoặc xanh dương đậm cho dòng **Random Forest** và các con số tốt nhất (`93,52%`, `98,45%`, `0,8284`, `0,9327`, `FP = 16`).

