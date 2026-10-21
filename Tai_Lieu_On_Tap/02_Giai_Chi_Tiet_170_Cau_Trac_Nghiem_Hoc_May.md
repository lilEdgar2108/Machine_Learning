# Giải Chi Tiết 170 Câu Hỏi Trắc Nghiệm Học Máy (Chương 1 – Chương 8)
**Tài liệu giải chi tiết 100% câu hỏi trắc nghiệm trong cuốn `Bai_tap_Hoc_may_v6_Image.pdf` (ĐH Thủy Lợi) kèm đối chiếu bảng đáp án gốc.**

---

## Chương 1: Giới Thiệu Về Học Máy (20 Câu: `1.1` – `1.20`)

| Câu | 1.1 | 1.2 | 1.3 | 1.4 | 1.5 | 1.6 | 1.7 | 1.8 | 1.9 | 1.10 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Đáp án** | **C** | **C** | **B** | **C** | **C** | **B** | **C** | **C** | **B** | **A** |
| **Câu** | **1.11** | **1.12** | **1.13** | **1.14** | **1.15** | **1.16** | **1.17** | **1.18** | **1.19** | **1.20** |
| **Đáp án** | **B** | **C** | **B** | **C** | **B** | **C** | **C** | **B** | **B** | **C** |

### Giải thích chi tiết từng câu Chương 1:
* **1.1 → C:** Theo định nghĩa kinh điển của Tom Mitchell (1997), một chương trình máy tính được xem là "học" nếu hiệu năng $P$ của nó trên tác vụ $T$ được cải thiện nhờ **kinh nghiệm $E$ thu thập từ dữ liệu**.
* **1.2 → C:** Giá cổ phiếu là một **con số thực liên tục** $\implies$ bài toán học có giám sát thuộc nhóm **Hồi quy (Regression)**.
* **1.3 → B:** Trong học có giám sát (*Supervised Learning*), tập dữ liệu huấn luyện luôn bao gồm các cặp **(vector đặc trưng $x_i$, nhãn đích $y_i$)**.
* **1.4 → C:** Học không giám sát (*Unsupervised Learning*) làm việc trên dữ liệu **không có nhãn**; máy tự khám phá cấu trúc ẩn (như cụm, thành phần chính) bên trong dữ liệu.
* **1.5 → C:** K-Means gom các điểm dữ liệu không nhãn thành $K$ cụm dựa trên khoảng cách $\implies$ thuộc nhóm **Học không giám sát - Phân cụm (Clustering)**.
* **1.6 → B:** Định lý Bayes $P(A\mid B) = \frac{P(B\mid A)P(A)}{P(B)}$ được dùng để **ước lượng xác suất hậu nghiệm dựa trên bằng chứng mới**.
* **1.7 → C:** Trong công thức cập nhật Gradient Descent $w := w - \eta L'(w)$, hệ số $\eta$ là **tốc độ học (learning rate)**.
* **1.8 → C:** Bài toán hồi quy dự đoán giá trị liên tục nên hàm mất mát phổ biến nhất là **Mean Squared Error (MSE)** (trong khi Cross-entropy và Hinge loss dùng cho phân loại).
* **1.9 → B:** Làm mượt Laplace (*Laplace smoothing*) cộng thêm $1$ vào tử số tần suất để **tránh xác suất bằng $0$ cho các từ chưa từng xuất hiện trong một lớp** khi nhân xác suất trong Naive Bayes.
* **1.10 → A:** Theo quy ước ma trận dữ liệu $X \in \mathbb{R}^{m \times n}$: $m$ hàng tương ứng với **$m$ mẫu (samples)**, $n$ cột tương ứng với **$n$ đặc trưng (features)** của mỗi mẫu.
* **1.11 → B:** **Overfitting (Quá khớp)** xảy ra khi mô hình quá phức tạp, **học thuộc cả nhiễu trong tập huấn luyện** (training error rất thấp) nhưng **dự đoán kém trên dữ liệu mới** (test error cao).
* **1.12 → C:** **Cây quyết định (Decision Tree)** là thuật toán **học có giám sát** (cần nhãn để tính Entropy/Gini), còn K-means, PCA, DBSCAN đều là học không giám sát.
* **1.13 → B:** **Học bán giám sát (*Semi-supervised learning*)** phù hợp nhất khi **chỉ một phần nhỏ dữ liệu có nhãn, phần lớn dữ liệu không có nhãn** (vì việc gán nhãn đắt đỏ).
* **1.14 → C:** Công thức tích vô hướng $a \cdot b = \|a\| \|b\| \cos\theta \implies$ càng lớn khi hai vector **càng cùng hướng ($\theta \approx 0^\circ$) và có độ lớn lớn**.
* **1.15 → B:** Chuẩn $L_2$ của vector $\|x\|_2 = \sqrt{\sum x_i^2}$ dùng để đo **độ dài Euclid (Euclidean length)** của vector.
* **1.16 → C:** Ký hiệu xác suất có điều kiện $P(\text{"mua"} \mid \text{Spam})$ biểu thị **xác suất từ "mua" xuất hiện với điều kiện email đó là spam**.
* **1.17 → C:** "Biên dịch tài liệu bằng cách tra từ điển thủ công" là thao tác thủ công/quy tắc cứng, **không phải** ứng dụng của học máy.
* **1.18 → B:** Nếu tốc độ học $\eta$ quá lớn, bước nhảy của Gradient Descent sẽ vượt qua đáy cực tiểu, khiến **hàm mất mát dao động mạnh hoặc phân kỳ**.
* **1.19 → B:** Mục tiêu chính của giảm chiều dữ liệu (*Dimensionality Reduction* như PCA) là **giảm số chiều (số đặc trưng) nhưng vẫn giữ được thông tin quan trọng nhất và giảm nhiễu**.
* **1.20 → C:** Naive Bayes dựa trên **Định lý Bayes** và giả định "ngây thơ" rằng **các đặc trưng độc lập có điều kiện với nhau khi biết nhãn lớp**.

---

## Chương 2: Hồi Quy Tuyến Tính Và Quá Khớp (20 Câu: `2.1` – `2.20`)

| Câu | 2.1 | 2.2 | 2.3 | 2.4 | 2.5 | 2.6 | 2.7 | 2.8 | 2.9 | 2.10 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Đáp án** | **B** | **B** | **B** | **C** | **B** | **A** | **A** | **C** | **B** | **B** |
| **Câu** | **2.11** | **2.12** | **2.13** | **2.14** | **2.15** | **2.16** | **2.17** | **2.18** | **2.19** | **2.20** |
| **Đáp án** | **D** | **D** | **C** | **C** | **C** | **B** | **B** | **B** | **D** | **C** |

### Giải thích chi tiết từng câu Chương 2:
* **2.1 → B:** Mục tiêu của Linear Regression là **dự báo giá trị liên tục** (số thực).
* **2.2 → B:** Phương trình hồi quy tuyến tính đơn (1 biến đầu vào) có dạng đường thẳng **$y = wx + b$**.
* **2.3 → B:** Trong Linear Regression, $w$ biểu diễn **trọng số của các đặc trưng (weights)**.
* **2.4 → C:** Tham số tự do $b$ (hoặc $w_0$) được gọi là **Bias (hoặc intercept - hệ số chặn)**.
* **2.5 → B:** Hàm mất mát phổ biến nhất trong Linear Regression là **Mean Squared Error (MSE)**.
* **2.6 → A:** Quá trình huấn luyện Linear Regression tìm bộ tham số $(w, b)$ nhằm **tối thiểu hóa hàm mất mát MSE**.
* **2.7 → A:** `Training error` là **sai số được tính ngay trên tập dữ liệu huấn luyện (training set)**.
* **2.8 → C:** `Test error` được tính trên **tập kiểm tra (Test set)** (dữ liệu hoàn toàn mới chưa dùng để huấn luyện).
* **2.9 → B:** Mô hình bị **Overfitting (quá khớp)** khi học quá sát tập train nhưng tổng quát hóa kém: **Training error thấp, Test error cao**.
* **2.10 → B:** Mô hình bị **Underfitting (chưa khớp)** khi quá đơn giản không học được cả tập train lẫn tập test: **Training error cao, Test error cao**.
* **2.11 → D:** Khi độ phức tạp của mô hình tăng lên (ví dụ tăng bậc đa thức), mô hình càng khớp sát các điểm huấn luyện nên **training error thường giảm**.
* **2.12 → D:** Tập Validation được sử dụng trong quá trình phát triển/lựa chọn mô hình để **đánh giá khả năng tổng quát hóa của mô hình** (và tinh chỉnh siêu tham số).
* **2.13 → C:** Tập Validation thường được **tách ra từ tập huấn luyện ban đầu (Training set)** (để giữ nguyên Test set độc lập đến cuối cùng).
* **2.14 → C:** Khi lựa chọn mô hình (Model Selection), mô hình tốt nhất là mô hình có **Validation error nhỏ nhất**.
* **2.15 → C:** Trong K-Fold Cross Validation, dữ liệu được chia thành **$K$ phần bằng nhau (folds)**.
* **2.16 → B:** Ở mỗi lần lặp của K-Fold Cross Validation, **$K-1$ folds dùng để train và 1 fold dùng để validation**.
* **2.17 → B:** Cross-validation đặc biệt hữu ích khi **dữ liệu ít**, vì nó tận dụng tối đa mọi mẫu cho cả việc học và kiểm định.
* **2.18 → B:** Chính quy hóa (*Regularization*) thêm thành phần phạt độ lớn trọng số nhằm **giảm độ phức tạp của mô hình**, chống overfitting.
* **2.19 → D:** Trong kỹ thuật **Early Stopping**, ta dừng huấn luyện ngay khi **Validation error bắt đầu có xu hướng tăng lên** (dù training error vẫn giảm).
* **2.20 → C:** Hàm mất mát có chính quy hóa $J(w) = L(w) + \lambda R(w)$ nhằm **đồng thời tối thiểu hóa cả sai số huấn luyện (loss) và thành phần chính quy hóa (regularization term)**.

---

## Chương 3: Gradient Descent Và Thuật Toán Perceptron (20 Câu: `3.1` – `3.20`)

| Câu | 3.1 | 3.2 | 3.3 | 3.4 | 3.5 | 3.6 | 3.7 | 3.8 | 3.9 | 3.10 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Đáp án** | **B** | **C** | **C** | **A** | **B** | **C** | **C** | **C** | **B** | **B** |
| **Câu** | **3.11** | **3.12** | **3.13** | **3.14** | **3.15** | **3.16** | **3.17** | **3.18** | **3.19** | **3.20** |
| **Đáp án** | **C** | **C** | **B** | **B** | **D** | **A** | **B** | **A** | **D** | **D** |

### Giải thích chi tiết từng câu Chương 3:
* **3.1 → B:** Gradient Descent là thuật toán tối ưu dùng chủ yếu để **tối ưu hàm mục tiêu (hàm mất mát) bằng cách tìm cực trị (cực tiểu)**.
* **3.2 → C:** Tham số $\eta$ (*learning rate*) quyết định **kích thước bước cập nhật tham số** ở mỗi vòng lặp.
* **3.3 → C:** Công thức cập nhật chuẩn của Gradient Descent là đi ngược hướng gradient: **$w \leftarrow w - \eta \nabla L(w)$**.
* **3.4 → A:** Điều kiện dừng thường dùng của Gradient Descent là khi **đạo hàm (gradient) xấp xỉ bằng 0** ($\nabla L(w) \approx 0$).
* **3.5 → B:** Nếu learning rate $\eta$ quá nhỏ, mỗi bước cập nhật đi được rất ngắn nên thuật toán **hội tụ chậm**.
* **3.6 → C:** Nếu learning rate $\eta$ quá lớn, thuật toán **có thể dao động lớn và bỏ qua điểm tối ưu** (thậm chí phân kỳ).
* **3.7 → C:** **Batch Gradient Descent (BGD)** tính đạo hàm trên **toàn bộ tập huấn luyện** cho mỗi một lần cập nhật trọng số.
* **3.8 → C:** Với $N$ mẫu dữ liệu (số chiều cố định), mỗi bước cập nhật của Batch Gradient Descent phải duyệt qua cả $N$ mẫu nên có độ phức tạp là **$\mathcal{O}(N)$**.
* **3.9 → B:** **Stochastic Gradient Descent (SGD)** cập nhật trọng số tức thời dựa trên **một mẫu dữ liệu ngẫu nhiên** tại mỗi bước.
* **3.10 → B:** Ưu điểm lớn nhất của SGD so với BGD là **chi phí tính toán cho mỗi lần cập nhật rất thấp** (chỉ tính trên 1 mẫu).
* **3.11 → C:** **Mini-batch Gradient Descent** là sự kết hợp hài hòa ưu điểm của **Batch GD** (ổn định, vector hóa) và **SGD** (cập nhật nhanh, thoát cực tiểu địa phương).
* **3.12 → C:** Thuật toán Perceptron được đề xuất bởi nhà tâm lý học/khoa học máy tính **Frank Rosenblatt** vào năm 1957.
* **3.13 → B:** Perceptron (PLA) được thiết kế cho bài toán **Phân lớp nhị phân (Binary Classification)** với hai nhãn $\{-1, +1\}$.
* **3.14 → B:** Trong Perceptron, nhãn dự đoán $\hat{y} = \text{sgn}(w^T x)$ được xác định dựa trên **dấu của tích vô hướng $w^T x$** ($+$ hoặc $-$).
* **3.15 → D:** Khi một mẫu $(x, y)$ bị phân lớp sai, quy tắc cập nhật trọng số của Perceptron là **$w \leftarrow w + yx$** (với $\eta = 1$).
* **3.16 → A:** Theo Định lý hội tụ Perceptron, PLA đảm bảo hội tụ sau hữu hạn bước khi và chỉ khi **dữ liệu phân tách tuyến tính được (linearly separable)**.
* **3.17 → B:** Độ đo **Precision** được tính bằng công thức **$\frac{TP}{TP + FP}$** (tỷ lệ mẫu đúng thực sự trong số các mẫu được dự đoán là Positive).
* **3.18 → A:** **$F_1$-score** là **trung bình điều hòa (harmonic mean)** của Precision và Recall: $F_1 = \frac{2PR}{P+R}$.
* **3.19 → D:** Để huấn luyện mô hình theo thời gian thực (**online learning** — dữ liệu đến từng mẫu liên tục và có 50.000 mẫu), phương pháp phù hợp nhất là **Stochastic Gradient Descent (SGD)**.
* **3.20 → D:** Với tập dữ liệu $N = 1000$ mẫu và kích thước mini-batch $k = 50$, một epoch sẽ thực hiện $\frac{N}{k} = \frac{1000}{50} = \mathbf{20}$ lần cập nhật trọng số.

---

## Chương 4: Mạng Nơ-ron Nhân Tạo (20 Câu: `4.1` – `4.20`)

| Câu | 4.1 | 4.2 | 4.3 | 4.4 | 4.5 | 4.6 | 4.7 | 4.8 | 4.9 | 4.10 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Đáp án** | **B** | **C** | **B** | **C** | **C** | **A** | **B** | **B** | **D** | **A** |
| **Câu** | **4.11** | **4.12** | **4.13** | **4.14** | **4.15** | **4.16** | **4.17** | **4.18** | **4.19** | **4.20** |
| **Đáp án** | **D** | **C** | **B** | **A** | **B** | **C** | **A** | **C** | **C** | **D** |

### Giải thích chi tiết từng câu Chương 4:
* **4.1 → B:** Các giá trị $x_1, x_2, \dots, x_n$ đưa vào nơ-ron được gọi là các **đầu vào (inputs / features)**.
* **4.2 → C:** Các tham số $w_i$ biểu diễn **trọng số kết nối (connection weights)** giữa đầu vào thứ $i$ và nơ-ron.
* **4.3 → B:** Tham số **Bias** ($b$) cộng thêm vào tổng có trọng số giúp **dịch chuyển hàm kích hoạt** sang trái/phải để khớp dữ liệu linh hoạt hơn.
* **4.4 → C:** Tầng nhận dữ liệu trực tiếp từ bên ngoài là **Input Layer (Tầng đầu vào)**.
* **4.5 → C:** Tầng tạo ra kết quả dự đoán cuối cùng của mạng là **Output Layer (Tầng đầu ra)**.
* **4.6 → A:** Tầng ẩn (**Hidden Layer**) là các tầng **nằm giữa tầng đầu vào (Input Layer) và tầng đầu ra (Output Layer)**.
* **4.7 → B:** Gọi là "tầng ẩn" vì **người dùng không quan sát trực tiếp các giá trị của tầng này từ dữ liệu đầu vào hoặc đầu ra**.
* **4.8 → B:** Vai trò chính của tầng ẩn là **trích xuất và học các đặc trưng (features/representations) bậc cao từ dữ liệu**.
* **4.9 → D:** Trong `scikit-learn` (`MLPClassifier`/`MLPRegressor`), khai báo `hidden_layer_sizes=(3, 2)` có nghĩa là mạng có **hai tầng ẩn, lần lượt có 3 và 2 nơ-ron**.
* **4.10 → A:** Việc sử dụng nhiều tầng ẩn (kết hợp hàm kích hoạt phi tuyến) giúp mạng **học được các mối quan hệ phi tuyến phức tạp hơn**.
* **4.11 → D:** Nếu mọi nơ-ron tầng ẩn đều dùng hàm kích hoạt tuyến tính $f(x) = x$, toàn bộ mạng nhiều lớp suy biến thành một phép nhân ma trận tuyến tính duy nhất $\implies$ **không làm tăng khả năng biểu diễn phi tuyến của mạng**.
* **4.12 → C:** Một tầng ẩn có $5$ nơ-ron, mỗi nơ-ron kết nối với $3$ đầu vào (không tính bias) $\implies$ số trọng số cần học là $3 \times 5 = \mathbf{15}$.
* **4.13 → B:** **Lan truyền xuôi (Forward Propagation)** là quá trình **tính đầu ra dự đoán của mạng từ dữ liệu đầu vào**.
* **4.14 → A:** **Hàm mất mát (Loss Function)** dùng để **đo mức độ sai khác giữa giá trị dự đoán $\hat{y}$ và giá trị thực tế $y$**.
* **4.15 → B:** Thuật toán **Backpropagation (Lan truyền ngược)** được sử dụng để **tính đạo hàm (gradient) của hàm mất mát** theo từng trọng số trong mạng.
* **4.16 → C:** Trong quá trình lan truyền ngược, tín hiệu sai số được truyền ngược từ tầng đầu ra về tầng đầu vào: **$\text{Output} \to \text{Input}$** (qua các tầng ẩn).
* **4.17 → A:** Sau khi Backpropagation tính xong gradient, **Gradient Descent** được dùng để **cập nhật trọng số nhằm giảm hàm mất mát**.
* **4.18 → C:** Mục tiêu cuối cùng của quá trình huấn luyện mạng nơ-ron là tìm bộ trọng số giúp **giảm giá trị của hàm mất mát** (tối ưu hóa hiệu năng dự đoán).
* **4.19 → C (Bài toán tính tay Forward):**
  * Ta có $x = \begin{bmatrix} 1 \\ 2 \end{bmatrix}$, $W^{(1)} = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}$ (ma trận đơn vị $I_2$), $W^{(2)} = \begin{bmatrix} 1 & 1 \end{bmatrix}$, hàm kích hoạt ReLU, không có bias.
  * Tầng ẩn: $z^{(1)} = W^{(1)} x = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} \begin{bmatrix} 1 \\ 2 \end{bmatrix} = \begin{bmatrix} 1 \\ 2 \end{bmatrix} \implies a^{(1)} = \text{ReLU}(z^{(1)}) = \begin{bmatrix} 1 \\ 2 \end{bmatrix}$.
  * Tầng đầu ra: $z^{(2)} = W^{(2)} a^{(1)} = \begin{bmatrix} 1 & 1 \end{bmatrix} \begin{bmatrix} 1 \\ 2 \end{bmatrix} = 1 \times 1 + 1 \times 2 = 3 \implies \hat{y} = \text{ReLU}(3) = \mathbf{3}$. Chọn **C**.
* **4.20 → D (Bài toán tính tay Backprop):**
  * Từ câu `4.19`, ta có dự đoán $\hat{y} = 3$, nhãn thật $y = 5$, và $a^{(1)} = \begin{bmatrix} 1 \\ 2 \end{bmatrix}$.
  * Hàm mất mát $J = \frac{1}{2}(y - \hat{y})^2 = \frac{1}{2}(\hat{y} - y)^2$.
  * Đạo hàm của $J$ theo $\hat{y}$: $\frac{\partial J}{\partial \hat{y}} = \hat{y} - y = 3 - 5 = -2$ (vì $z^{(2)} = 3 > 0$ nên $\text{ReLU}'(z^{(2)}) = 1$).
  * Do $z^{(2)} = W^{(2)} a^{(1)}$, ta có:
    $$\frac{\partial J}{\partial W^{(2)}} = (\hat{y} - y) \cdot (a^{(1)})^T = (-2) \cdot \begin{bmatrix} 1 & 2 \end{bmatrix} = \mathbf{[-2, -4]}$$
    Chọn **D**.

---

## Chương 5: Phương Pháp Học Theo Mô Hình Cây (20 Câu: `5.1` – `5.20`)

| Câu | 5.1 | 5.2 | 5.3 | 5.4 | 5.5 | 5.6 | 5.7 | 5.8 | 5.9 | 5.10 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Đáp án** | **C** | **C** | **A** | **B** | **C** | **C** | **D** | **B** | **C** | **C** |
| **Câu** | **5.11** | **5.12** | **5.13** | **5.14** | **5.15** | **5.16** | **5.17** | **5.18** | **5.19** | **5.20** |
| **Đáp án** | **B** | **A** | **B** | **B** | **B** | **A** | **B** | **C** | **C** | **B** |

### Giải thích chi tiết từng câu Chương 5:
* **5.1 → C:** Tập dữ liệu có 2 lớp với tỷ lệ $p_1 = 0.7$ và $p_2 = 0.3$. Entropy (tính theo đơn vị bit, cơ số 2) là:
  $$H(S) = -0.7 \log_2(0.7) - 0.3 \log_2(0.3) \approx -0.7(-0.5146) - 0.3(-1.7370) \approx 0.3602 + 0.5211 = \mathbf{0.8813\text{ bit}}$$
* **5.2 → C:** Đề bài hỏi phát biểu **SAI**: ID3 luôn chọn thuộc tính có Information Gain **lớn nhất** (chứ không phải nhỏ nhất) để làm nút phân nhánh $\implies$ **C** hiển nhiên sai (đúng theo bảng đáp án gốc). *(Lưu ý thêm: câu B viết "entropy sau khi chia trừ entropy trước khi chia" cũng bị ngược dấu so với công thức $\text{Gain} = H_{\text{trước}} - H_{\text{sau}}$, nhưng trong đáp án chính thức của giáo trình chọn **C** vì C sai trực diện về nguyên lý chọn nút của ID3).*
* **5.3 → A:** Trong CART, chỉ số $\text{Gini} = 1 - \sum p_i^2 = 0$ khi nút hoàn toàn thuần khiết, tức **tất cả các mẫu thuộc cùng một lớp** ($p_c = 1$).
* **5.4 → B:** Cây quyết định thường cần **cắt tỉa (pruning)** để **giảm độ sâu của cây, tránh quá khớp (overfitting) và cải thiện khả năng tổng quát hóa**.
* **5.5 → C:** Trong **cây hồi quy (Regression Tree)**, biến mục tiêu là số thực nên dự đoán tại nút lá là **trung bình cộng (mean) của các giá trị mục tiêu** trong nút lá đó.
* **5.6 → C:** Điểm yếu cố hữu của cây quyết định là dự đoán dạng bậc thang hằng số từng khúc nên **kém khả năng ngoại suy (extrapolation) ngoài phạm vi dữ liệu huấn luyện**.
* **5.7 → D:** Phát biểu **ĐÚNG** về ID3: **ID3 chọn thuộc tính có Information Gain lớn nhất để làm nút phân nhánh** (A, B là đặc điểm của CART; C sai vì ID3 không trực tiếp xử lý số liên tục).
* **5.8 → B:** Nhược điểm nổi tiếng của ID3 là **dễ bị ảnh hưởng (thiên vị) bởi các thuộc tính có nhiều giá trị** (ví dụ thuộc tính mã số ID có Gain lớn nhất nhưng không có giá trị tổng quát hóa).
* **5.9 → C:** Phát biểu **SAI** về CART là: **"CART không thể áp dụng cho bài toán hồi quy"** (vì chính chữ R trong CART là *Regression* — CART giải được cả phân lớp và hồi quy!).
* **5.10 → C:** Khi xây dựng **cây hồi quy**, CART chọn phép phân nhánh dựa trên tiêu chí **Reduction in Variance (Giảm phương sai / giảm MSE)**.
* **5.11 → B:** Tập dữ liệu chứa đúng một lớp duy nhất ($p_1 = 1$) hoàn toàn không có độ hỗn loạn nên $\text{Entropy} = -1 \log_2(1) = \mathbf{0}$.
* **5.12 → A:** Trong ID3, quá trình phân nhánh dừng lại khi **tất cả các mẫu trong nút thuộc cùng một lớp (thuần khiết), hoặc không còn thuộc tính nào để phân chia**.
* **5.13 → B:** Công thức $\text{Gain}(S, A) = H(S) - \sum \frac{|S_v|}{|S|} H(S_v)$ thể hiện **mức giảm Entropy trung bình sau khi phân nhánh theo thuộc tính $A$**.
* **5.14 → B:** Tập dữ liệu có $10$ mẫu gồm $6$ mẫu "Có" ($p_1 = 0.6$) và $4$ mẫu "Không" ($p_2 = 0.4$). Entropy là:
  $$H(S) = -0.6 \log_2(0.6) - 0.4 \log_2(0.4) \approx 0.4422 + 0.5288 = \mathbf{0.971\text{ bit}}$$
* **5.15 → B:** ID3 nguyên bản **không xử lý được thuộc tính số liên tục một cách trực tiếp** (phải rời rạc hóa trước hoặc nâng cấp lên C4.5 / CART).
* **5.16 → A:** Tập có $4$ mẫu gồm $3$ mẫu lớp A ($p_A = \frac{3}{4} = 0.75$) và $1$ mẫu lớp B ($p_B = \frac{1}{4} = 0.25$). Chỉ số Gini là:
  $$\text{Gini} = 1 - \left(\left(\frac{3}{4}\right)^2 + \left(\frac{1}{4}\right)^2\right) = 1 - \left(\frac{9}{16} + \frac{1}{16}\right) = \frac{6}{16} = \mathbf{0.375}$$
* **5.17 → B:** Trong CART phân lớp, một phép chia làm tăng độ thuần khiết nên tốt hơn nếu **tổng Gini có trọng số của hai tập con nhỏ hơn Gini của nút cha**.
* **5.18 → C:** Vì CART luôn tạo cây nhị phân, khi gặp thuộc tính hạng mục có 3 giá trị $\{X, Y, Z\}$, CART sẽ **duyệt các cách gộp thành hai nhóm (ví dụ $\{X\}$ vs $\{Y, Z\}$) để tìm phép chia nhị phân tốt nhất**.
* **5.19 → C:** Tương tự câu `5.5`, trong cây hồi quy, giá trị dự đoán tại nút lá là **trung bình cộng của các giá trị mục tiêu trong nút**.
* **5.20 → B:** Độ giảm phương sai ($\text{Reduction in Variance} = \text{Var}(S) - \sum \frac{|S_k|}{|S|}\text{Var}(S_k)$) đạt giá trị lớn nhất khi **phương sai của cả hai tập con sau khi chia đều nhỏ** (các điểm trong mỗi nút con có giá trị $y$ rất gần nhau).

---

## Chương 6: Hồi Quy Logistic & Máy Vector Hỗ Trợ - SVM (20 Câu: `6.1` – `6.20`)

| Câu | 6.1 | 6.2 | 6.3 | 6.4 | 6.5 | 6.6 | 6.7 | 6.8 | 6.9 | 6.10 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Đáp án** | **C** | **B** | **B** | **C** | **B** | **C** | **C** | **B** | **B** | **B** |
| **Câu** | **6.11** | **6.12** | **6.13** | **6.14** | **6.15** | **6.16** | **6.17** | **6.18** | **6.19** | **6.20** |
| **Đáp án** | **B** | **A** | **C** | **A** | **B** | **C** | **D** | **B** | **C** | **C** |

### Giải thích chi tiết từng câu Chương 6:
* **6.1 → C:** Hàm Sigmoid $\sigma(z) = \frac{1}{1 + e^{-z}}$ luôn nhận giá trị trong khoảng mở **$(0, 1)$** với mọi $z \in \mathbb{R}$ (vì $e^{-z} > 0$).
* **6.2 → B:** Trong Logistic Regression, hàm mất mát chuẩn thường dùng là **Binary Cross-Entropy (Log Loss)**.
* **6.3 → B:** Hàm mất mát Cross-Entropy trong Logistic Regression **là hàm lồi (convex) theo tham số mô hình**, giúp Gradient Descent luôn hội tụ về cực tiểu toàn cục (nó không có nghiệm dạng đóng nên A sai).
* **6.4 → C:** Tại điểm dữ liệu được dự đoán đúng với xác suất rất cao ($\hat{y}_i \approx 1$ và $y_i = 1$), sai số $(\hat{y}_i - y_i) \approx 0 \implies$ gradient đóng góp $(\hat{y}_i - y_i)x_{ij}$ **gần bằng 0**.
* **6.5 → B:** Mục tiêu của nguyên lý Hợp lý Cực đại (**MLE - Maximum Likelihood Estimation**) là tìm tham số mô hình nhằm **cực đại hóa xác suất quan sát được tập dữ liệu huấn luyện**.
* **6.6 → C:** Trong SVM tuyến tính, khoảng cách từ siêu phẳng đến mỗi đường biên lề là $\frac{1}{\|w\|}$, nên tổng độ rộng lề (**margin**) giữa hai lớp được tính bằng **$\frac{2}{\|w\|}$**.
* **6.7 → C:** Theo điều kiện KKT trong bài toán đối ngẫu của SVM, **Support vectors** là những điểm **có nhân tử Lagrange $\alpha_i > 0$ trong bài toán đối ngẫu** (nằm trên biên lề ở hard-margin, hoặc nằm trên/trong/vượt lề ở soft-margin).
* **6.8 → B:** Trong SVM lề mềm (*Soft-margin SVM*), hàm mục tiêu là $\frac{1}{2}\|w\|^2 + C\sum \xi_i$. Khi tham số phạt **$C$ lớn**, mô hình phạt rất nặng các lỗi vi phạm lề $\implies$ dẫn đến **Margin hẹp hơn, hạn chế điểm vi phạm**.
* **6.9 → B:** Trong bài toán đối ngẫu Lagrange của SVM, mỗi mẫu huấn luyện $(x_i, y_i)$ gắn với đúng 1 biến nhân tử Lagrange $\alpha_i \implies$ số biến của bài toán đối ngẫu **bằng số mẫu huấn luyện $N$**.
* **6.10 → B (Sai):** Khi dùng **Kernel trick**, dữ liệu **KHÔNG** được ánh xạ tường minh lên không gian đặc trưng mà chỉ tính ngầm tích vô hướng thông qua hàm nhân $K(x_i, x_j) \implies$ phát biểu "ánh xạ tường minh" là **Sai (B)**.
* **6.11 → B:** Logistic Regression dùng hàm Sigmoid để **biến đổi tổ hợp tuyến tính $z = w^T x + b$ thành một giá trị trong khoảng $(0, 1)$ để biểu diễn xác suất**.
* **6.12 → A:** Công thức hàm mất mát Binary Cross-Entropy cho 1 mẫu $(x_i, y_i)$ là **$-\left[y_i \log \hat{y}_i + (1 - y_i)\log(1 - \hat{y}_i)\right]$**.
* **6.13 → C:** Gradient đối với 1 mẫu $(\hat{y}_i - y_i)x_i$ gần bằng $0$ khi **dự đoán $\hat{y}_i$ rất gần với nhãn thật $y_i$ (mô hình tự tin và đúng)**.
* **6.14 → A:** Nếu tất cả $x_{ij} = 0$ với mọi mẫu $i$, thì $\frac{\partial J}{\partial w_j} = \frac{1}{N}\sum (\hat{y}_i - y_i)\cdot 0 = 0 \implies$ **Gradient bằng 0, trọng số $w_j$ không thay đổi**.
* **6.15 → B:** MSE ít được dùng cho Logistic Regression vì khi ghép với hàm Sigmoid, **MSE làm cho bài toán tối ưu trở nên không lồi (non-convex), dễ bị kẹt ở cực tiểu địa phương**.
* **6.16 → C:** Trong SVM tuyến tính lề cứng, việc cực tiểu hóa $\frac{1}{2}\|w\|^2$ (tức cực tiểu hóa $\|w\|$) tương đương với việc **cực đại hóa độ rộng lề (margin) $\frac{2}{\|w\|}$ giữa hai lớp**.
* **6.17 → D:** Những mẫu có $\alpha_i = 0$ trong bài toán đối ngẫu của SVM là **các điểm nằm ngoài margin ($y_i(w^T x_i + b) > 1$), đã được phân lớp đúng và không ảnh hưởng đến siêu phẳng phân tách**.
* **6.18 → B:** Vai trò của tham số $C$ trong SVM lề mềm là **đánh đổi (trade-off) giữa độ rộng margin và mức phạt đối với các điểm vi phạm margin**.
* **6.19 → C:** Trong SVM lề mềm, điều kiện là $y_i(w^T x_i + b) \ge 1 - \xi_i$. Một điểm **bị phân lớp sai** có $y_i(w^T x_i + b) < 0$, suy ra biến bù phải thỏa mãn **$\xi_i > 1$**.
* **6.20 → C:** **Kernel trick** trong SVM cho phép **ngầm ánh xạ dữ liệu lên không gian đặc trưng nhiều chiều hơn (thậm chí vô hạn chiều) mà không cần tính tường minh ánh xạ $\Phi(x)$**.

---

## Chương 7: Học Không Giám Sát - K-Means & DBSCAN (30 Câu: `7.1` – `7.30`)

| Câu | 7.1 | 7.2 | 7.3 | 7.4 | 7.5 | 7.6 | 7.7 | 7.8 | 7.9 | 7.10 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Đáp án** | **B** | **C** | **B** | **C** | **C** | **B** | **C** | **B** | **B** | **B** |
| **Câu** | **7.11** | **7.12** | **7.13** | **7.14** | **7.15** | **7.16** | **7.17** | **7.18** | **7.19** | **7.20** |
| **Đáp án** | **B** | **A** | **A** | **B** | **B** | **C** | **B** | **C** | **B** | **B** |
| **Câu** | **7.21** | **7.22** | **7.23** | **7.24** | **7.25** | **7.26** | **7.27** | **7.28** | **7.29** | **7.30** |
| **Đáp án** | **A** | **C** | **B** | **B** | **B** | **A** | **B** | **B** | **B** | **B** |

### Giải thích chi tiết từng câu Chương 7:
* **7.1 → B:** K-Means làm việc trên dữ liệu không có nhãn $\implies$ thuộc loại **Học không giám sát (Unsupervised Learning)**.
* **7.2 → C:** Trong K-Means, mỗi cụm được đại diện bởi tâm cụm (centroid) là **trung bình cộng (mean) của các điểm trong cụm**.
* **7.3 → B:** Hàm mục tiêu mà K-Means cố gắng tối thiểu hóa là **Tổng bình phương khoảng cách nội cụm (WCSS - Within-Cluster Sum of Squares)**.
* **7.4 → C:** K-Means luôn giảm WCSS sau mỗi bước và chỉ có hữu hạn cách phân hoạch nên **luôn hội tụ đến một cực tiểu cục bộ (local minimum) sau số hữu hạn bước**.
* **7.5 → C:** Ở bước cập nhật tâm, tâm cụm mới được tính **bằng trung bình cộng các điểm được gán vào cụm đó**.
* **7.6 → B:** Vì K-Means chỉ hội tụ về cực tiểu cục bộ nên việc khởi tạo tâm ban đầu **ảnh hưởng trực tiếp đến tốc độ hội tụ và chất lượng phân cụm cuối cùng**.
* **7.7 → C:** Thuật toán khởi tạo **K-Means++** cải tiến khởi tạo ngẫu nhiên bằng cách **chọn các tâm ban đầu sao cho chúng ở xa nhau với xác suất cao** (tỷ lệ với bình phương khoảng cách đến tâm gần nhất đã chọn).
* **7.8 → B:** Ở mỗi vòng lặp K-Means, ta cần tính khoảng cách từ $N$ điểm đến $K$ tâm cụm trong không gian $d$ chiều $\implies$ độ phức tạp là **$\mathcal{O}(N \cdot K \cdot d)$**.
* **7.9 → B:** Khi số cụm $K$ tăng lên, các điểm càng gần tâm cụm hơn nên giá trị **WCSS (inertia) luôn giảm**.
* **7.10 → B:** Nhược điểm cơ bản của K-Means là **cần phải biết trước số cụm $K$ và giả định các cụm có dạng hình cầu (lồi, kích thước tương đương)**.
* **7.11 → B:** Trong bước gán nhãn của K-Means, mỗi điểm được gán vào cụm có **tâm gần nhất theo khoảng cách Euclid**.
* **7.12 → A:** Thuật toán K-Means chuẩn (với bước cập nhật tâm là trung bình cộng để tối thiểu hóa phương sai) dựa trên **khoảng cách Euclidean** (nếu dùng Manhattan sẽ dẫn tới biến thể K-Medians/K-Medoids).
* **7.13 → A:** Phương pháp **Elbow (Khuỷu tay)** chọn $K$ bằng cách vẽ **đồ thị WCSS theo $K$ và tìm điểm gập khuỷu tay** (nơi WCSS bắt đầu giảm chậm lại).
* **7.14 → B:** Khi các cụm thực sự có hình dạng **không lồi** (ví dụ 2 vòng tròn đồng tâm), K-Means dùng đường phân giác phẳng nên **gặp khó khăn và thường phân cụm sai**.
* **7.15 → B:** Trong `scikit-learn`, thuộc tính `inertia_` của K-Means chính là **tổng bình phương khoảng cách từ các điểm đến tâm cụm của chúng (WCSS)**.
* **7.16 → C:** Nếu trong quá trình chạy K-Means có một cụm trở thành rỗng (không có điểm nào gần tâm đó nhất), **tâm cụm đó có thể được khởi tạo lại (ví dụ lấy điểm xa nhất) hoặc bị loại bỏ**.
* **7.17 → B:** So với K-Medoids (PAM - chọn 1 điểm dữ liệu thực tế làm tâm), K-Means khác ở chỗ **dùng trung bình cộng (mean) làm tâm, nên tâm có thể không trùng với bất kỳ điểm dữ liệu thực nào**.
* **7.18 → C:** Sau khi K-Means đã hội tụ, nếu chạy thêm vòng lặp thì **các tâm và nhãn không thay đổi nữa (thuật toán dừng)**.
* **7.19 → B:** Ý tưởng của K-Means có thể xem như trường hợp đặc biệt của thuật toán **Kỳ vọng - Cực đại hóa (EM) cho mô hình Gaussian Mixture (GMM) với ma trận hiệp phương sai cầu**.
* **7.20 → B:** Khi các đặc trưng có thang đo rất khác biệt (tuổi $0-100$ vs thu nhập $0-100$ triệu), đặc trưng có giá trị lớn sẽ lấn át hoàn toàn khoảng cách Euclid $\implies$ bắt buộc phải **chuẩn hóa dữ liệu (standardize / normalize)** trước khi áp dụng K-Means.
* **7.21 → A:** DBSCAN là viết tắt của **Density-Based Spatial Clustering of Applications with Noise**.
* **7.22 → C:** DBSCAN chỉ cần 2 tham số chính là bán kính $\varepsilon$ (`eps`), ngưỡng số điểm $\text{minPts}$ (và hàm khoảng cách), hoàn toàn **không có tham số số cụm $K$**.
* **7.23 → B:** Một điểm được gọi là **Core point (điểm lõi)** nếu **nó có ít nhất $\text{minPts}$ điểm nằm trong bán kính $\varepsilon$-lân cận của nó**.
* **7.24 → B:** **Border point (điểm biên)** là điểm **không phải là core point, nhưng nằm trong vùng $\varepsilon$-lân cận của một core point**.
* **7.25 → B:** Trong DBSCAN, một cụm được định nghĩa chặt chẽ là **tập tối đại các điểm được kết nối mật độ (density-connected)**.
* **7.26 → A:** **Điểm nhiễu (Noise)** trong DBSCAN là **điểm không thuộc bất kỳ cụm nào** (không phải core point và cũng không nằm trong $\varepsilon$-lân cận của bất kỳ core point nào).
* **7.27 → B:** Ưu điểm vượt trội của DBSCAN so với K-Means là **không cần chỉ định trước số cụm $K$, phát hiện được điểm nhiễu và tìm được cụm có hình dạng bất kỳ**.
* **7.28 → B:** Hạn chế chính của DBSCAN là **rất nhạy cảm với hai tham số $\varepsilon$ và $\text{minPts}$, đồng thời gặp khó khăn khi mật độ giữa các cụm khác nhau lớn**.
* **7.29 → B:** Khi bán kính $\varepsilon$ được đặt **quá lớn**, vùng lân cận mở rộng bao trùm các cụm $\implies$ DBSCAN có xu hướng **hợp nhất tất cả các điểm thành một cụm duy nhất hoặc rất ít cụm**.
* **7.30 → B:** Khi ngưỡng $\text{minPts}$ tăng lên, điều kiện để đạt mật độ lõi khó hơn $\implies$ **số điểm được coi là Core point sẽ giảm đi**.

---

## Chương 8: Học Kết Hợp - Ensemble Learning (20 Câu: `8.1` – `8.20`)

| Câu | 8.1 | 8.2 | 8.3 | 8.4 | 8.5 | 8.6 | 8.7 | 8.8 | 8.9 | 8.10 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Đáp án** | **B** | **B** | **C** | **C** | **B** | **B** | **B** | **C** | **B** | **C** |
| **Câu** | **8.11** | **8.12** | **8.13** | **8.14** | **8.15** | **8.16** | **8.17** | **8.18** | **8.19** | **8.20** |
| **Đáp án** | **B** | **B** | **B** | **B** | **C** | **B** | **C** | **B** | **C** | **C** |

### Giải thích chi tiết từng câu Chương 8:
* **8.1 → B:** Ý tưởng chính của Học kết hợp (*Ensemble Learning*) là **kết hợp nhiều mô hình con (base learners) để đạt hiệu quả dự đoán và tổng quát hóa cao hơn**.
* **8.2 → B:** **Bootstrapping** là kỹ thuật thống kê **lấy mẫu ngẫu nhiên có hoàn lại (with replacement) có cùng kích thước với mẫu gốc**.
* **8.3 → C:** Xác suất một quan sát không được chọn trong $n$ lần rút có hoàn lại là $\left(1 - \frac{1}{n}\right)^n \approx e^{-1} \approx \mathbf{0.368}$ ($36.8\%$).
* **8.4 → C:** Trong **Bagging**, các mô hình con được huấn luyện **độc lập (song song) trên các mẫu bootstrap khác nhau**.
* **8.5 → B:** Bằng cách lấy trung bình/biểu quyết nhiều mô hình độc lập, **Bagging giúp giảm Phương sai (Variance)** hiệu quả nhất (chống overfitting).
* **8.6 → B:** Trong **Random Forest**, việc chọn ngẫu nhiên một tập con đặc trưng tại mỗi nút phân nhánh nhằm **làm giảm sự tương quan (correlation) giữa các cây**, giúp giảm phương sai của toàn rừng tốt hơn nữa.
* **8.7 → B:** Với tổng số $p$ đặc trưng, giá trị mặc định của số đặc trưng xét tại mỗi nút cho bài toán **phân loại** trong Random Forest thường là **$\sqrt{p}$** (`max_features='sqrt'`).
* **8.8 → C:** **Out-of-Bag (OOB)** là **phương pháp đánh giá mô hình dùng chính các mẫu không nằm trong mẫu bootstrap ($\approx 36.8\%$) để kiểm thử trên những cây không được huấn luyện bằng quan sát đó**.
* **8.9 → B:** Trong Bagging cho bài toán **Hồi quy (Regression)**, dự đoán cuối cùng được tính bằng **trung bình cộng (mean) các dự đoán thành phần** (còn Phân loại mới dùng biểu quyết đa số).
* **8.10 → C:** Trong **Boosting**, các mô hình con được huấn luyện **tuần tự (sequentially), mô hình sau tập trung sửa lỗi của mô hình trước**.
* **8.11 → B:** Sau mỗi vòng lặp, thuật toán **AdaBoost cập nhật trọng số của các mẫu huấn luyện** (tăng trọng số mẫu bị phân lớp sai, giảm trọng số mẫu phân lớp đúng).
* **8.12 → B:** Thay $\epsilon_t = 0.1$ vào công thức trọng số mô hình của AdaBoost:
  $$\alpha_t = \frac{1}{2}\ln\left(\frac{1 - \epsilon_t}{\epsilon_t}\right) = \frac{1}{2}\ln\left(\frac{1 - 0.1}{0.1}\right) = \frac{1}{2}\ln(9) \approx \frac{2.1972}{2} \approx \mathbf{1.1}$$
* **8.13 → B:** Để một weak learner có ích trong AdaBoost ($\alpha_t > 0$), sai số có trọng số của nó phải thỏa mãn **$\epsilon_t < 0.5$ (tốt hơn đoán ngẫu nhiên)**.
* **8.14 → B:** **Gradient Boosting** khác AdaBoost cơ bản ở chỗ thay vì cập nhật trọng số mẫu, nó **tối ưu hóa trực tiếp hàm mất mát khả vi bằng cách khớp cây mới vào phần dư (negative gradient / residuals)**.
* **8.15 → C:** Với hàm mất mát bình phương (*Squared Loss* $\frac{1}{2}(y - F(x))^2$), đạo hàm âm chính là **phần dư $y - F_{m-1}(x)$**, nên mỗi cây mới sẽ khớp trực tiếp với $y - F_{m-1}(x)$.
* **8.16 → B:** Random Forest ít bị overfitting hơn một cây quyết định đơn lẻ vì **nó lấy trung bình của nhiều cây kết hợp với việc ngẫu nhiên hóa (bootstrap + chọn ngẫu nhiên đặc trưng) giúp giảm mạnh phương sai**.
* **8.17 → C:** **XGBoost** (*Extreme Gradient Boosting*) là một bản cải tiến tối ưu hóa cả về giải thuật lẫn hệ thống của **Gradient Boosting**.
* **8.18 → B:** Lỗi **OOB (Out-of-Bag error)** trong Random Forest là ước lượng không chệch của lỗi tổng quát hóa nên **có thể thay thế trực tiếp cho tập kiểm tra (validation/test set) để ước lượng sai số tổng quát**.
* **8.19 → C:** Theo Luật số lớn, khi tăng số lượng cây trong Random Forest lên rất lớn, **lỗi tổng quát sẽ giảm dần rồi bão hòa (tiệm cận một giới hạn), chứ không tăng lên (không bị overfit vì thêm cây)**.
* **8.20 → C:** Khi base learner có độ lệch (bias) thấp và phương sai (variance) cao (như cây quyết định không cắt tỉa), **Bagging có tác dụng giảm phương sai và giữ nguyên độ lệch**.
