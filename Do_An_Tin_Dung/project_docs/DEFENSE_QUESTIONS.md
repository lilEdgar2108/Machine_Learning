# DEFENSE QUESTIONS – NGÂN HÀNG CÂU HỎI VẤN ĐÁP THẦY CHIỂU

## Phần A: Câu hỏi Chương I & Chương II (Trọng tâm của Hải Nam – Thành & Vũ nắm chéo)

### Câu 1: Tại sao nhóm điền dữ liệu thiếu cho `person_emp_length` và `loan_int_rate` bằng Trung vị (Median) mà không dùng Trung bình cộng (Mean) hay xóa bỏ các dòng bị thiếu?
- **Đáp án chuẩn ngắn gọn:**
  1. Không xóa bỏ vì riêng `loan_int_rate` thiếu `3.116` dòng (`9.56%`), nếu xóa sẽ làm mất gần `10%` dữ liệu quý giá.
  2. Không dùng `Mean` vì dữ liệu tài chính/tín dụng có phân phối lệch phải mạnh (ví dụ `person_income` có độ lệch `skewness = 32.87`, `person_emp_length` có `skewness = 2.61`). Giá trị `Mean` dễ bị kéo lệch bởi một số ít khách hàng thu nhập hoặc thâm niên rất lớn, còn `Median` nằm đúng vị trí chính giữa (phân vị 50%) nên phản ánh khách hàng điển hình bền vững hơn nhiều.

### Câu 2: Tại sao bước lọc ngoại lai (`tuổi <= 100`, `thâm niên <= 60`) nhóm làm TRƯỚC khi chia Train/Test, nhưng bước điền khuyết `Median` và chuẩn hóa `StandardScaler` lại làm SAU khi chia Train/Test? Có bị Data Leakage ở bước lọc ngoại lai không?
- **Đáp án chuẩn ăn điểm tuyệt đối:**
  - **Không hề bị Data Leakage!** Vì quy tắc lọc `person_age <= 100` và `person_emp_length <= 60` là **ràng buộc logic thực tế cố định trên từng dòng độc lập** (không ai vay tiêu dùng ở tuổi 144 hay đi làm 123 năm), hoàn toàn **không tính toán bất kỳ tham số thống kê nào từ toàn bộ tập dữ liệu**.
  - Ngược lại, nếu lọc ngoại lai bằng quy tắc thống kê như $Z\text{-score} > 3$ hay $IQR$, hoặc tính `Median`, `Mean`, `Std`, `Quantile` thì các con số đó phụ thuộc vào tập mẫu, nên bắt buộc phải chia `Train/Test` trước rồi chỉ `fit` trên `Train`!

### Câu 3: Tại sao cùng là biến phân loại nhưng `loan_grade` lại dùng Ordinal Encoding (`0..6`), còn `person_home_ownership` và `loan_intent` lại dùng One-Hot Encoding?
- **Đáp án chuẩn:**
  - `loan_grade` (`A` đến `G`) là biến phân loại **có thứ tự tự nhiên**: thực tế trên EDA, tỷ lệ vỡ nợ tăng liên tục từ hạng `A` (`10.0%`) lên hạng `G` (`98.4%`). Mã hóa thành `0..6` bảo toàn đúng thứ tự rủi ro này cho cả Logistic Regression lẫn Cây quyết định.
  - Còn `loan_intent` (Giáo dục, Y tế, Sửa nhà...) là biến **định danh không có thứ tự**. Nếu gán `0, 1, 2, 3...` rồi đưa vào tổ hợp tuyến tính $z = w^T x + b$ của Logistic Regression, mô hình sẽ hiểu sai về mặt toán học rằng "Mục đích Sửa nhà gấp 3 lần Giáo dục". Dùng One-Hot Encoding tách thành các cột nhị phân `0/1` độc lập giúp loại bỏ thứ tự giả tạo đó.

### Câu 4: Tại sao nhóm chuẩn hóa `StandardScaler` cho Logistic Regression nhưng lại giữ nguyên đơn vị gốc cho CART và Random Forest, và rời rạc hóa cho ID3?
- **Đáp án chuẩn:**
  - **Logistic Regression:** Tính $z = w^T x + b$ và tối ưu hàm mất mát kèm điều chuẩn L2 ($\lambda \|w\|_2^2$). Thu nhập hàng chục nghìn USD lớn gấp hàng vạn lần tỷ lệ vay/thu nhập (`0.15`), nếu không đưa về cùng thang đo Z-score ($\mu=0, \sigma=1$), thuật toán tối ưu hội tụ chậm và L2 phạt không công bằng giữa các trọng số.
  - **CART & Random Forest:** Xét các phép chia nhị phân $x_j \le t$ chỉ phụ thuộc vào **thứ tự sắp xếp** của biến. Phép biến đổi tuyến tính dương $(x_j - \mu)/\sigma$ giữ nguyên 100% thứ tự và chỉ số Gini, nên không cần chuẩn hóa mà giữ đơn vị gốc (USD, năm, %) giúp đọc nút chia trên cây trực quan hơn nhiều.
  - **ID3:** Thuật toán ID3 nguyên bản chỉ phân nhánh đa hướng trên các giá trị rời rạc, nên bắt buộc phải rời rạc hóa (Binning) 7 biến số liên tục thành 4 khoảng phân vị (`Thấp`, `Trung bình`, `Khá cao`, `Cao`).

---

## Phần B: Câu hỏi Chương III – 4 Thuật Toán (Trọng tâm của Thành – Hải Nam & Vũ nắm chéo)

### Câu 5: Tại sao trong Logistic Regression phải dùng hàm mất mát Binary Cross-Entropy (Log Loss) mà không dùng hàm MSE (Trung bình bình phương sai số) như Hồi quy tuyến tính?
- **Đáp án chuẩn theo tài liệu thầy Chiểu:**
  - Nếu thay hàm phi tuyến Sigmoid $\hat{y} = \frac{1}{1 + e^{-(w^T x + b)}}$ vào hàm MSE $\frac{1}{2}(\hat{y} - y)^2$, hàm mục tiêu trở thành **hàm không lồi (non-convex)** có nhiều vùng đạo hàm bão hòa bằng 0 và cực tiểu địa phương, làm thuật toán tối ưu dễ bị kẹt.
  - Ngược lại, hàm **Binary Cross-Entropy** xuất phát từ Nguyên lý Hợp lý Cực đại (*Maximum Likelihood Estimation – MLE*) là một **hàm lồi nghiêm ngặt (convex)** theo $(w, b)$, đảm bảo luôn hội tụ về cực tiểu toàn cục duy nhất.

### Câu 6: Trong slide trên lớp thầy tính Entropy theo logarit tự nhiên $\ln$, còn trong code `id3_tu_cai_dat.py` nhóm lại dùng `np.log2`. Cây quyết định ID3 sinh ra có bị khác nhau không?
- **Đáp án chuẩn 10/10:**
  - **Cấu trúc cây giống hệt nhau 100%, không khác dù chỉ một nhánh!**
  - Vì theo công thức đổi cơ số logarit: $\log_2(p) = \frac{\ln(p)}{\ln 2}$. Mọi giá trị Entropy $H(S)$ và Độ lợi thông tin $\text{Gain}(S, A)$ tính theo $\log_2$ (đơn vị *bit*) chỉ bằng đúng giá trị tính theo $\ln$ (đơn vị *nat*) nhân với hằng số dương $\frac{1}{\ln 2} \approx 1.4427$. Do đó thứ tự lớn nhỏ giữa các thuộc tính khi tìm $\arg\max_A \text{Gain}(S, A)$ hoàn toàn bất biến!

### Câu 7: Phân biệt cốt lõi giữa cây ID3 tự cài đặt của nhóm và cây CART của `scikit-learn`? Tại sao qua 5-Fold CV, ID3 đạt đỉnh ở `max_depth = 4` trong khi CART lại đạt đỉnh ở `max_depth = 10`?
- **Đáp án chuẩn gắn chặt thực nghiệm:**
  1. **Tiêu chuẩn đo độ tạp:** ID3 dùng `Entropy & Information Gain`; CART dùng `Gini Impurity`.
  2. **Cách chia nhánh:** ID3 chia **đa nhánh (multi-way split)** theo toàn bộ giá trị rời rạc của thuộc tính (ví dụ nút gốc `loan_grade` chia một lúc thành 7 nhánh `A..G`); còn CART luôn chia **nhị phân (binary split – đúng 2 nhánh)** với cả biến số lẫn biến phân loại.
  3. **Lý do chênh lệch `max_depth` tối ưu:** Vì ID3 chia 4–7 nhánh mỗi tầng nên số nút lá bùng nổ theo hàm mũ cơ số lớn — ngay ở `max_depth = 4`, cây ID3 đã có tới **370 nút lá** (`CV ROC-AUC = 0.8843`), nếu tăng lên 5–8 các nhánh con bị xé quá nhỏ gây Overfitting. Ngược lại, cây nhị phân CART ở độ sâu 4 tối đa mới có $2^4 = 16$ lá, nên cần độ sâu `max_depth = 10` mới đủ chi tiết để phân tách tốt các vùng rủi ro (`CV ROC-AUC = 0.9140`).

### Câu 8: Tại sao Random Forest phải lấy ngẫu nhiên tập con đặc trưng (`max_features = 'sqrt'`) tại mỗi nút chia? Nếu bỏ bước đó (xét toàn bộ 19 đặc trưng ở mọi nút) thì chuyện gì xảy ra?
- **Đáp án chuẩn:**
  - Nếu ở mọi nút đều xét cả 19 đặc trưng (tức chỉ làm Bagging thông thường), vài đặc trưng cực mạnh như `loan_percent_income`, `loan_grade` và `person_income` sẽ luôn chiến thắng ở các tầng trên cùng của hầu hết cả 200 cây. Khi đó 200 cây sẽ có cấu trúc **rất giống nhau (tương quan cao)**, việc lấy trung bình không giảm được bao nhiêu phương sai (Variance).
  - Việc chỉ bốc ngẫu nhiên $\lfloor\sqrt{19}\rfloor = 4$ đặc trưng ứng viên tại mỗi nút buộc các cây phải khai thác cả những đặc trưng thứ cấp, làm **giảm tương quan giữa các cây** và giúp Random Forest chống Overfitting vượt trội.

---

## Phần C: Câu hỏi Phần 3.5 & Chương IV (Trọng tâm của Vũ – Hải Nam & Thành nắm chéo)

### Câu 9: Trong bài toán xét duyệt vay vốn ngân hàng, sai lầm FP (False Positive) và FN (False Negative) cái nào gây thiệt hại nặng nề hơn? Giải thích trên con số thực tế của nhóm!
- **Đáp án chuẩn 10/10:**
  - Quy ước lớp Dương tính (`1`) = **Vỡ nợ**, lớp Âm tính (`0`) = **Trả nợ tốt**.
  - **FP (Báo động giả):** Khách hàng thực chất trả nợ tốt nhưng bị mô hình phán nhầm là vỡ nợ $\implies$ Ngân hàng từ chối cho vay $\implies$ Chỉ **mất chi phí cơ hội thu tiền lãi** (và giảm trải nghiệm khách hàng). Ví dụ trên tập Test, Random Forest chỉ mắc **16 ca FP** trên 5.064 khách tốt (`0.32%`).
  - **FN (Bỏ lọt nợ xấu – Nguy hiểm chí mạng!):** Khách hàng thực chất sẽ vỡ nợ nhưng mô hình phán nhầm là trả tốt $\implies$ Ngân hàng giải ngân tiền thật $\implies$ **Mất trắng cả tiền gốc lẫn tiền lãi** và phát sinh chi phí thu hồi nợ! Một ca mất gốc (FN) thường phải cần lợi nhuận lãi vay của 10–15 khách tốt mới bù đắp nổi.

### Câu 10: Nếu FN nguy hiểm như vậy, tại sao ngân hàng không hạ ngưỡng Threshold từ `0.5` xuống `0.01` để bắt hết 100% nợ xấu (`Recall = 100%`)?
- **Đáp án chuẩn (Precision – Recall Trade-off):**
  - Nếu hạ ngưỡng xuống `0.01`, hầu như mọi hồ sơ đều bị phán là "Vỡ nợ". Khi đó dù `Recall = 100%` (`FN = 0`), chỉ số `Precision` sẽ tụt dốc thảm hại (`FP` tăng vọt lên gần 5.000 người): ngân hàng từ chối gần hết khách hàng tốt, không giải ngân được khoản vay nào thì cũng phá sản vì không có doanh thu lãi vay!
  - Đó là lý do phải dùng **`F1-Score`** (trung bình điều hòa phạt nặng nếu 1 trong 2 chỉ số quá thấp) và **`ROC-AUC`** để cân bằng giữa an toàn vốn (Recall) và tăng trưởng tín dụng (Precision).

### Câu 11: Tại sao tính `F1-Score` người ta dùng Trung bình điều hòa (Harmonic Mean) mà không lấy Trung bình cộng `(Precision + Recall) / 2`?
- **Đáp án chuẩn:**
  - Giả sử một mô hình phán bừa mọi khách hàng đều vỡ nợ: `Recall = 1.00` (`100%`), còn `Precision = 0.2188` (`21.88%`).
  - Nếu tính **trung bình cộng**, điểm ra `(1.0 + 0.2188)/2 = 0.6094` (trông có vẻ khá ổn dù mô hình vô dụng!).
  - Nhưng với **trung bình điều hòa** $F_1 = \frac{2PR}{P+R}$, điểm chỉ còn **`0.3590`** — vì trung bình điều hòa luôn bị kéo mạnh về phía con số nhỏ hơn, buộc mô hình phải cao đồng thời cả Precision lẫn Recall mới đạt F1 cao!

### Câu 12: Khi một khách hàng mới nộp hồ sơ vào hàm `predict_loan_status(customer_data)`, tại sao nhóm nạp `preprocessor_bundle.joblib` để `.transform()` mà không gọi `.fit_transform()` trực tiếp trên hồ sơ đó?
- **Đáp án chuẩn:**
  1. Về toán học, 1 dòng dữ liệu duy nhất có độ lệch chuẩn $\sigma = 0$ (không thể tính Z-score hay chia tứ phân vị).
  2. Quan trọng hơn, nếu dùng `OneHotEncoder().fit_transform()` trên 1 khách hàng thuê nhà (`RENT`), nó chỉ sinh ra 1 cột thay vì đủ 4 cột nhà ở như lúc huấn luyện 19 đặc trưng, làm mô hình báo lỗi lệch số chiều ngay lập tức. Nạp `preprocessor_bundle.joblib` đảm bảo khách hàng mới đi qua đúng 100% hệ quy chiếu của tập Train!



