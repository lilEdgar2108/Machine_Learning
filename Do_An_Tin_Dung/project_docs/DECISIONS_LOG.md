# DECISIONS LOG – NHẬT KÝ QUYẾT ĐỊNH KỸ THUẬT

| Mã QĐ | Ngày | Nội dung quyết định | Lý do học thuật & thực tiễn | Trạng thái |
| :--- | :--- | :--- | :--- | :--- |
| **QĐ-01** | 2026-10-31 | Cô lập toàn bộ đồ án vào thư mục `Do_An_Tin_Dung/` | Bảo vệ tuyệt đối các bài tập cũ trong `Btap TH/`, `Btap TrenLop/`, `Tai_Lieu_On_Tap/` và dễ quản lý trên GitHub. | Đã chốt |
| **QĐ-02** | 2026-10-31 | Chọn `Credit Risk Dataset` (`OpenML ID = 43454`, 32.581 dòng × 12 cột) làm Dataset chính; giữ toàn bộ bản ghi hợp lệ sau làm sạch | Đạt 9.45/10 điểm trọng số: biến đời thường dễ giải thích, có đủ dữ liệu trùng lặp, ngoại lai vô lý (tuổi > 100, năm làm việc > 100), dữ liệu thiếu (`person_emp_length`, `loan_int_rate`), có cả biến số, biến định danh và biến thứ tự (`loan_grade`). | Đã chốt |
| **QĐ-03** | 2026-10-31 | Chia `train_test_split(..., test_size=0.2, stratify=y, random_state=42)` **TRƯỚC KHI** Impute, Encode, Scale và Discretize | Chống tuyệt đối **Data Leakage** (rò rỉ thông tin từ tập Test sang tập Train) và giữ nguyên tỷ lệ vỡ nợ ~21.8% ở cả hai tập. | Đã chốt |
| **QĐ-04** | 2026-10-31 | Tự cài đặt `ID3DecisionTree` đa nhánh bằng Python thuần trên dữ liệu đã rời rạc hóa thay vì gọi `DecisionTreeClassifier(criterion='entropy')` là ID3 | `scikit-learn` cài đặt thuật toán CART chia nhánh nhị phân. Tự viết ID3 đa nhánh khớp 100% với bài giảng `bt_id3_4.1.py` và sách bài tập Chương 5 của thầy Chiểu. | Đã chốt |
| **QĐ-05** | 2026-10-31 | Dùng `matplotlib` chuẩn để xuất toàn bộ biểu đồ `.png` (`dpi=180`) | Không phụ thuộc thư viện phụ ngoài hệ thống, đồng bộ phong cách các file trong `Btap TH/` của thầy Chiểu. | Đã chốt |

