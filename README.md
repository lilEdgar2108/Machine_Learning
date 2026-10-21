# Machine Learning - Đại học Thủy lợi

## Giới thiệu

Đây là repository phục vụ việc học tập môn Máy học tại Đại học Thủy lợi.

Repository được sử dụng để lưu trữ các bài tập, bài thực hành, tài liệu ghi chú và các project được thực hiện trong quá trình học. Nội dung trong repository sẽ được cập nhật theo tiến độ của môn học.

## Mục tiêu

- Hoàn thành đầy đủ các bài tập và yêu cầu do giảng viên giao.
- Củng cố kiến thức nền tảng về máy học và quy trình xây dựng một mô hình máy học.
- Rèn luyện kỹ năng xử lý, phân tích và trực quan hóa dữ liệu.
- Tìm hiểu, triển khai và đánh giá các thuật toán máy học phù hợp với từng bài toán.
- Xây dựng các project có tính ứng dụng thực tế thông qua việc áp dụng kiến thức máy học.
- Hình thành quy trình làm việc hoàn chỉnh từ thu thập dữ liệu, tiền xử lý dữ liệu, huấn luyện mô hình đến đánh giá kết quả.

## Nội dung học tập

Các nội dung chính được thực hiện trong repository gồm:

1. Tổng quan về máy học.
2. Thu thập, làm sạch và tiền xử lý dữ liệu.
3. Phân tích và trực quan hóa dữ liệu.
4. Học có giám sát và học không giám sát.
5. Các bài toán hồi quy và phân loại.
6. Lựa chọn đặc trưng và xây dựng mô hình.
7. Đánh giá, tối ưu hóa và cải thiện mô hình.
8. Ứng dụng máy học vào các bài toán thực tế.

## Định hướng project

Bên cạnh việc hoàn thành các bài tập trên lớp, repository hướng đến việc phát triển những project có khả năng giải quyết các vấn đề thực tế. Mỗi project sẽ tập trung vào việc áp dụng máy học để phân tích dữ liệu, dự đoán kết quả hoặc hỗ trợ ra quyết định.


## Công nghệ sử dụng

- Python
- Jupyter Notebook
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

Danh sách công nghệ có thể được bổ sung tùy theo yêu cầu của từng bài tập hoặc project.

## Cấu trúc repository

Cấu trúc repository có thể được tổ chức theo từng bài tập và project. Mỗi phần nên bao gồm mã nguồn, dữ liệu sử dụng, kết quả thực hiện và tài liệu mô tả khi cần thiết.

Ví dụ:

```text
Machine_Learning/
├── assignments/     Các bài tập được giao
├── projects/        Các project có tính ứng dụng
├── notebooks/       Các notebook thực hành và phân tích dữ liệu
├── data/            Dữ liệu sử dụng trong quá trình học
└── README.md        Tài liệu giới thiệu repository
```

## Cách sử dụng

Để chạy các bài thực hành hoặc project, người dùng cần cài đặt Python và các thư viện cần thiết. Nếu project có tệp `requirements.txt`, có thể cài đặt các thư viện bằng lệnh:

```bash
pip install -r requirements.txt
```

Hướng dẫn chạy cụ thể sẽ được bổ sung trong thư mục tương ứng với từng bài tập hoặc project.

## Tiến độ thực hiện

Repository sẽ được cập nhật thường xuyên trong suốt quá trình học. Tiến độ bao gồm:

- Hoàn thành các bài tập do giảng viên giao.
- Hoàn thành các bài thực hành về xử lý và phân tích dữ liệu.
- Hoàn thành các bài thực hành xây dựng mô hình máy học.
- Xây dựng và hoàn thiện các project có tính ứng dụng.
- Tổng hợp kiến thức và đánh giá kết quả học tập.

## Ghi chú

Repository được xây dựng với mục đích học tập và thực hành. Các nội dung sẽ tiếp tục được chỉnh sửa, bổ sung và cải thiện dựa trên kiến thức tiếp thu trong môn học cũng như kết quả thực hiện các bài tập và project.


---

## Danh mục Bài tập Thực hành & Tài liệu Ôn tập (Chương 1 – Chương 8)

### 1. Tài liệu Ôn tập & Lời giải Chi tiết (`Tai_Lieu_On_Tap/`)
- [`01_Tai_lieu_On_tap_Ly_thuyet_Hoc_May.md`](./Tai_Lieu_On_Tap/01_Tai_lieu_On_tap_Ly_thuyet_Hoc_May.md): Tổng hợp lý thuyết, công thức toán học và phương pháp giải bài tập tính tay toàn bộ 8 chương.
- [`02_Giai_Chi_Tiet_170_Cau_Trac_Nghiem_Hoc_May.md`](./Tai_Lieu_On_Tap/02_Giai_Chi_Tiet_170_Cau_Trac_Nghiem_Hoc_May.md): Bảng đáp án chuẩn và giải thích chi tiết cho toàn bộ 170 câu hỏi trắc nghiệm.
- [`03_Giai_Chi_Tiet_Tu_Luan_Tinh_Tay_va_Code_Hoc_May.md`](./Tai_Lieu_On_Tap/03_Giai_Chi_Tiet_Tu_Luan_Tinh_Tay_va_Code_Hoc_May.md): Lời giải tự luận lý thuyết, bài tập tính tay từng bước và code mẫu Python.

### 2. Mã nguồn Giải Bài tập Thực hành theo Chương (`Btap TH/`)
- [`BT_Chuong_2_Linear_Regression.py`](./Btap%20TH/BT_Chuong_2_Linear_Regression.py): Hồi quy tuyến tính, Overfitting/Underfitting, K-Fold CV, Early Stopping, LASSO vs Ridge, đánh giá $R^2$, NSE, MAE, RMSE (Bài 2.21 – 2.25).
- [`BT_Chuong_3_GD_Perceptron.py`](./Btap%20TH/BT_Chuong_3_GD_Perceptron.py): Mô phỏng Gradient Descent, tính toán & cập nhật Perceptron (PLA), cài đặt lớp `Perceptron` từ đầu (Bài 3.26 – 3.30).
- [`BT_Chuong_4_Neural_Network.py`](./Btap%20TH/BT_Chuong_4_Neural_Network.py): Tính toán Lan truyền xuôi & Lan truyền ngược, cài đặt mạng Nơ-ron đa tầng `CustomMLP` từ đầu cho Hồi quy & Phân lớp (Bài 4.19 – 4.28).
- [`BT_Chuong_5_Decision_Tree_ID3_CART.py`](./Btap%20TH/BT_Chuong_5_Decision_Tree_ID3_CART.py): Tính tay & cài đặt thuật toán ID3 từ đầu (`build_tree`, `predict`), thực hành cây phân lớp & cây hồi quy CART (Bài 5.21 – 5.25).
- [`BT_Chuong_6_Logistic_SVM.py`](./Btap%20TH/BT_Chuong_6_Logistic_SVM.py): Tính toán Hồi quy Logistic, cài đặt Gradient Descent cho Logistic Regression trên tập Iris, tính toán & thực hành SVM lề cứng / lề mềm / RBF Kernel (Bài 1 – Bài 5 Chương 6).
- [`BT_Chuong_7_KMeans_DBSCAN.py`](./Btap%20TH/BT_Chuong_7_KMeans_DBSCAN.py): Tính tay K-Means & DBSCAN, tự cài đặt `MyKMeans` & `MyDBSCAN`, phương pháp Elbow, nén ảnh bằng K-Means ($K=16$), so sánh K-Means++ (Bài 1 – Bài 8 Chương 7).
- [`BT_Chuong_8_Ensemble_Learning.py`](./Btap%20TH/BT_Chuong_8_Ensemble_Learning.py): Ước lượng khoảng tin cậy Bootstrap 95%, tính tay Bagging với Decision Stumps, Random Forest OOB vs CV, tự cài đặt AdaBoost từ đầu và so sánh các mô hình Ensemble (Bài 1 – Bài 5 Chương 8).
