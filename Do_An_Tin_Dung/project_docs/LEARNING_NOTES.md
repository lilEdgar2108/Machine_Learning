# LEARNING NOTES – SỔ TAY KIẾN THỨC 6 TẦNG CHO NHÓM

## Chủ đề 1 (Chặng 1): Chống Rò Rỉ Dữ Liệu (Data Leakage) & Lấy Mẫu Phân Tầng (Stratified Split)
- **Tầng 1 (Đời thường):** Giống như ôn thi đại học, đề thi thật (tập Test) phải được niêm phong cất kín trước khi mở sách ôn bài (tập Train). Nếu lấy cả đề thi thật ra tính điểm trung bình hay tìm quy luật trước khi học, kết quả thi thử sẽ cao ảo nhưng ra đời thật sẽ hỏng.
- **Tầng 2 (Ví dụ tín dụng):** Khi điền lãi suất bị khuyết cho khách hàng, ngân hàng chỉ được tính mức lãi suất trung vị từ **25.927 hồ sơ quá khứ** (`10.99%`), tuyệt đối không được lấy thông tin của **6.482 khách hàng tương lai** (tập Test) gộp vào tính chung.
- **Tầng 3 (Bản chất kỹ thuật):** Chia tập dữ liệu sạch $D$ thành $D_{\text{train}}$ ($80\%$) và $D_{\text{test}}$ ($20\%$) sao cho $P(y=1 \mid D_{\text{train}}) \approx P(y=1 \mid D_{\text{test}}) \approx 21.87\%$ (`stratify=y`). Mọi phép biến đổi có tham số $\theta$ (Median của `SimpleImputer`, $\mu, \sigma$ của `StandardScaler`, phân vị $Q_1, Q_2, Q_3$ của Binning) chỉ ước lượng $\hat{\theta} = \text{fit}(X_{\text{train}})$, rồi áp dụng $\text{transform}(X_{\text{train}}; \hat{\theta})$ và $\text{transform}(X_{\text{test}}; \hat{\theta})$.
- **Tầng 4 (Liên hệ code):** Hàm `chia_tap_train_test_phan_tang` được gọi ở bước số 2 trong `main()` của `src/02_tien_xu_ly_du_lieu.py`, đứng trước tất cả các hàm `dien_khuyet_du_lieu`, `ma_hoa_va_chuan_hoa_cho_logreg_cart_rf` và `roi_rac_hoa_cho_id3`.

## Chủ đề 2 (Chặng 1): Rời Rạc Hóa Biến Liên Tục (Discretization / Binning) Cho Thuật Toán ID3
- **Tầng 1 (Đời thường):** Thay vì hỏi chính xác lương khách hàng lẻ từng đồng (`54,321 USD`), cán bộ tín dụng gom thành 4 mức dễ phân loại: `Thấp`, `Trung bình`, `Khá cao`, `Cao`.
- **Tầng 2 (Ví dụ tín dụng):** Biến `person_income` trên tập Train có hàng nghìn con số khác nhau. Gom theo tứ phân vị tập Train thành 4 nhóm (`<= 38,888`, `38,888 - 55,000`, `55,000 - 79,000`, `> 79,000` USD) giúp cây ID3 chia đúng 4 nhánh rõ nghĩa.
- **Tầng 3 (Bản chất kỹ thuật):** ID3 gốc tính $\text{Gain}(S, A) = H(S) - \sum_{k=1}^K \frac{|S_k|}{|S|} H(S_k)$ bằng cách tạo $K$ nhánh ứng với $K$ giá trị khác nhau của $A$. Nếu $A$ là số thực liên tục có $K \approx N$ giá trị riêng biệt, mỗi nhánh con $S_k$ chỉ có 1–2 mẫu nên $H(S_k) \approx 0 \implies \text{Gain}(S, A) \approx H(S)$ đạt cực đại giả tạo nhưng không tổng quát hóa được (nhược điểm thiên vị thuộc tính nhiều giá trị của ID3 trong Chương 5).
- **Tầng 4 (Liên hệ code):** Hàm `roi_rac_hoa_cho_id3` trong `src/02_tien_xu_ly_du_lieu.py` và bảng `outputs/tables/02_nguong_roi_rac_hoa_id3.csv`.

---

## Chủ đề 3 (Chặng 2): Bản Chất 4 Thuật Toán Học Máy Ở Chương III (Phần Của Thành)

### 3.1. Logistic Regression (Hồi quy Logistic)
- **Tầng 1 (Đời thường):** Cán bộ ngân hàng cộng điểm từng tiêu chí hồ sơ (có tiêu chí cộng điểm rủi ro, có tiêu chí trừ điểm rủi ro), sau đó đưa tổng điểm qua một "bộ nén chữ S" (hàm Sigmoid) để ép về thang xác suất từ `0%` đến `100%`. Quá `50%` thì kết luận có nguy cơ vỡ nợ.
- **Tầng 2 (Ví dụ tín dụng):** Hồ sơ có `loan_percent_income` cao ($w = +1.4039$) và `loan_grade` xấu ($w = +1.1122$) sẽ đẩy điểm $z$ tăng mạnh về phía dương ($\sigma(z) \to 1$, dễ vỡ nợ); ngược lại khách tự sở hữu nhà `person_home_ownership_OWN` ($w = -1.8633$) kéo $z$ về âm ($\sigma(z) \to 0$, an toàn).
- **Tầng 3 (Bản chất kỹ thuật):**
  - Tổ hợp tuyến tính: $z = w^T x + b$.
  - Xác suất dự đoán lớp $1$ (Vỡ nợ): $\hat{y} = P(y=1 \mid x) = \sigma(z) = \frac{1}{1 + e^{-z}}$.
  - Hàm mất mát Binary Cross-Entropy (Log Loss) có điều chuẩn L2:
    $$J(w, b) = -\frac{1}{N} \sum_{i=1}^N \left[y_i \ln \hat{y}_i + (1 - y_i) \ln(1 - \hat{y}_i)\right] + \frac{\lambda}{2} \|w\|_2^2$$
    *(Trong `scikit-learn`, tham số $C = \frac{1}{\lambda} = 1.0$; dùng BCE thay vì MSE vì BCE kết hợp Sigmoid tạo hàm lồi - convex, đảm bảo hội tụ tới cực tiểu toàn cục).*
- **Tầng 4 (Liên hệ code):** Hàm `huan_luyen_logistic_regression` trong `src/03_huan_luyen_mo_hinh.py` và bảng `outputs/tables/03_he_so_logistic_regression.csv`.

### 3.2. Decision Tree – ID3 (Tự cài đặt thuần Python)
- **Tầng 1 (Đời thường):** Giống trò chơi "20 câu hỏi", mỗi lượt mô hình chọn ra câu hỏi giúp phân loại đám đông lộn xộn thành các nhóm rõ trắng đen nhanh nhất, và chia ra tất cả các phương án trả lời cùng lúc.
- **Tầng 2 (Ví dụ tín dụng):** Ban đầu tập Train có `21.87%` vỡ nợ nên độ hỗn loạn $H(S) = 0.7578\text{ bit}$. Khi xét cả 11 thuộc tính, `loan_grade` đạt Độ lợi thông tin cao nhất ($\text{IG} = 0.1078\text{ bit}$) nên được chọn làm **Nút gốc** và chia ngay thành **7 nhánh** (`A` đến `G`).
- **Tầng 3 (Bản chất kỹ thuật):**
  - Entropy: $H(S) = -\sum_{c \in \{0, 1\}} p_c \log_2(p_c)$.
  - Độ lợi thông tin: $\text{Gain}(S, A) = H(S) - \sum_{k=1}^K \frac{|S_k|}{|S|} H(S_k)$.
  - Phân nhánh đa hướng (*Multi-way split*): Mỗi thuộc tính rời rạc chỉ xuất hiện tối đa 1 lần trên mỗi đường đi từ gốc tới lá. Cắt tỉa trước với `max_depth = 4` vì chia đa nhánh làm số lá tăng rất nhanh ($370$ lá ở độ sâu $4$).
- **Tầng 4 (Liên hệ code):** Class `ID3DecisionTree` trong `src/id3_tu_cai_dat.py` (`tinh_entropy`, `tinh_information_gain`, `_xay_dung_cay_de_quy`, `predict_proba`).

### 3.3. Decision Tree – CART
- **Tầng 1 (Đời thường):** Vẫn là cây hỏi đáp nhưng **chỉ hỏi câu hỏi Có/Không (nhị phân 2 nhánh)** dựa trên các ngưỡng số cụ thể (ví dụ: `Lãi suất <= 11.0%?`).
- **Tầng 2 (Ví dụ tín dụng):** CART cắt trực tiếp trên số thực như `loan_percent_income <= 0.305` hoặc `person_income <= 45,000 USD` mà không cần gom khoảng trước như ID3.
- **Tầng 3 (Bản chất kỹ thuật):**
  - Chỉ số tạp chất Gini: $\text{Gini}(S) = 1 - \sum_{c \in \{0,1\}} p_c^2 = 2p_1(1 - p_1)$.
  - Chọn cặp thuộc tính và ngưỡng cắt $(j, t)$ làm cực tiểu hóa $\text{Gini}_{\text{split}}(S) = \frac{|S_{\text{trái}}|}{|S|}\text{Gini}(S_{\text{trái}}) + \frac{|S_{\text{phải}}|}{|S|}\text{Gini}(S_{\text{phải}})$.
- **Tầng 4 (Liên hệ code):** Hàm `huan_luyen_cart_va_random_forest` trong `src/03_huan_luyen_mo_hinh.py`.

### 3.4. Random Forest (Rừng ngẫu nhiên)
- **Tầng 1 (Đời thường):** Thay vì tin lời phán xét của 1 thẩm định viên duy nhất (dễ chủ quan), ngân hàng lập **Hội đồng 200 thẩm định viên độc lập**, mỗi người học trên một phần hồ sơ hơi khác nhau và nhìn vào các khía cạnh khác nhau, rồi lấy trung bình phán quyết của cả 200 người.
- **Tầng 2 (Ví dụ tín dụng):** 200 cây CART cùng chấm 1 hồ sơ vay: nếu 150 cây báo vỡ nợ, xác suất rủi ro tổng hợp xoay quanh `75%`, ổn định hơn hẳn 1 cây đơn lẻ.
- **Tầng 3 (Bản chất kỹ thuật):**
  - **Bootstrap Sampling:** Rút ngẫu nhiên có hoàn lại $N$ mẫu cho mỗi cây (khoảng $1/e \approx 36.8\%$ mẫu rơi ra ngoài làm tập **Out-of-Bag – OOB**).
  - **Feature Subspace:** Tại mỗi nút chia chỉ xét ngẫu nhiên $m = \lfloor\sqrt{19}\rfloor = 4$ đặc trưng để **phá vỡ sự tương quan giữa các cây**, từ đó giảm mạnh **Phương sai (Variance)** khi lấy trung bình xác suất $\hat{P}_{\text{RF}}(y=1 \mid x) = \frac{1}{B}\sum_{b=1}^B \hat{P}_b(y=1 \mid x)$.
- **Tầng 4 (Liên hệ code):** `RandomForestClassifier(n_estimators=200, max_features="sqrt", oob_score=True, ...)` trong `src/03_huan_luyen_mo_hinh.py`.

---

## Chủ đề 4 (Chặng 3): Trọng Tâm Phần 3.5 (8 Tiêu Chí Đánh Giá) & Chương IV (Phần Của Vũ)

### 4.1. Confusion Matrix & Sự bất đối xứng chi phí FP vs FN trong Tín dụng
- **Lớp Dương tính (`1`) = Vỡ nợ**, **Lớp Âm tính (`0`) = Trả nợ tốt** (`N_test = 6,482`: `5,064` tốt, `1,418` xấu).
- **TP (True Positive):** Bắt đúng khách sẽ vỡ nợ -> Ngân hàng từ chối kịp thời, bảo toàn tiền gốc.
- **TN (True Negative):** Duyệt đúng khách trả tốt -> Ngân hàng giải ngân an toàn, thu lợi nhuận lãi vay.
- **FP (False Positive – Sai lầm loại I):** Đánh nhầm khách tốt thành vỡ nợ -> Mất **chi phí cơ hội thu lãi** (Random Forest chỉ mắc `16` ca / `5,064` khách tốt = `0.32%`).
- **FN (False Negative – Sai lầm loại II):** Bỏ lọt khách vỡ nợ thành khách tốt -> Giải ngân tiền thật và **mất trắng cả gốc lẫn lãi** (`CART` thấp nhất với `392` ca, `Random Forest` `404` ca, `ID3` `492` ca, `Logistic` bỏ lọt tới `711` ca!).

### 4.2. Accuracy Paradox, Precision – Recall Trade-off, F1-Score & ROC-AUC
- **Accuracy Paradox:** Duyệt bừa 100% hồ sơ vẫn đạt `Accuracy = 78.12%` nhưng `Recall = 0%` (phá sản!).
- **Precision = $\frac{TP}{TP+FP}$:** Trong những người bị phán vỡ nợ, bao nhiêu % vỡ nợ thật (`Random Forest` đạt `98.45%`).
- **Recall = $\frac{TP}{TP+FN}$:** Trong tổng `1,418` người vỡ nợ thật ngoài đời, bắt trúng bao nhiêu % (`CART` đạt `72.36%`, `Random Forest` đạt `71.51%` ở ngưỡng mặc định `0.5`).
- **F1-Score = $\frac{2PR}{P+R}$:** Trung bình điều hòa phạt nặng nếu 1 trong 2 chỉ số quá thấp (`Random Forest` cao nhất `0.8284`).
- **ROC-AUC:** Quét mọi ngưỡng phân loại từ `0` đến `1` (trục tung `TPR = Recall`, trục hoành `FPR`). `Random Forest` đạt **`AUC = 0.9327`** cao nhất toàn diện, chứng tỏ khả năng xếp hạng rủi ro tốt nhất (nếu ngân hàng muốn bắt thêm nợ xấu mùa khủng hoảng, chỉ cần hạ ngưỡng từ `0.50` xuống `0.35` trên `predict_loan_status`).



