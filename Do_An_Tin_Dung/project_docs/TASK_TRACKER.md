# TASK TRACKER – BẢNG THEO DÕI NHIỆM VỤ 3 THÀNH VIÊN

## Chặng 1: Khởi tạo, Khám phá dữ liệu (EDA) & Tiền xử lý (Chương I & II)
- [x] Khảo sát phong cách bài giảng & bài tập thực hành của thầy Chiểu (`Btap TH/`, `Btap TrenLop/`, `Tai_Lieu_On_Tap/`).
- [x] Đánh giá và chốt Dataset chính: `Credit Risk Dataset` (`OpenML ID = 43454`).
- [x] Tạo thư mục cô lập `Do_An_Tin_Dung/` và hệ thống 9 file quản lý dự án.
- [x] **[Hải Nam]** Xây dựng và chạy thành công `src/01_tai_va_kham_pha_du_lieu.py` (tải dữ liệu, xuất 3 bảng CSV và 4 biểu đồ EDA Chương I).
- [x] **[Hải Nam]** Xây dựng và chạy thành công `src/02_tien_xu_ly_du_lieu.py` (xóa 165 dòng trùng, lọc 7 dòng ngoại lai vô lý, chia Stratified 80/20 trước tiên, Impute Median/Mode, Encode, StandardScaler, Rời rạc hóa tứ phân vị cho ID3, lưu `preprocessor_bundle.joblib`).

## Chặng 2: Cài đặt & Huấn luyện 4 Thuật toán (Chương III: 3.1 – 3.4)
- [x] **[Thành]** Xây dựng `src/id3_tu_cai_dat.py` (`ID3DecisionTree` thuần Python: Entropy, Information Gain, phân nhánh đa hướng, `max_depth`, xác suất nút lá `predict_proba`).
- [x] **[Thành]** Xây dựng và chạy `src/03_huan_luyen_mo_hinh.py` khảo sát 5-Fold Stratified CV chọn độ sâu cắt tỉa, huấn luyện Logistic Regression, ID3, CART, Random Forest (`oob_score=True`) và lưu đủ 4 model `.joblib`.

## Chặng 3: Đánh giá thực nghiệm Phần 3.5 & Chương trình dự đoán (Chương IV)
- [x] **[Vũ]** Xây dựng và chạy `src/04_danh_gia_va_so_sanh.py` xuất bảng so sánh 4 thuật toán (Train & Test) kèm Confusion Matrix, ROC-AUC Curve, biểu đồ cột 5 chỉ số.
- [x] **[Vũ]** Xây dựng và chạy `src/05_du_doan_khach_hang_moi.py` với hàm `predict_loan_status(customer_data)` và 3 kịch bản khách hàng thực tế (Khách tốt, Khách rủi ro cao, Khách khuyết dữ liệu).

## Chặng 4 & 5: Tổng hợp Báo cáo 4 Chương & Luyện Vấn Đáp
- [x] **[Vũ + Nhóm]** Tạo và chạy thành công script `report/xuat_bao_cao_docx.py` xuất báo cáo Word `Bao_Cao_Do_An_Hoc_May_Nhom_HaiNam_Thanh_Vu.docx` (`1.79 MB`, 13 bảng dữ liệu thật, 12 biểu đồ PNG sắc nét) khớp 100% số liệu thực nghiệm.
- [ ] **[Hải Nam + Thành + Vũ]** Luyện vấn đáp giả lập (`ACTIVATE DEFENSE MODE` / `MEMBER HAI NAM` / `MEMBER THANH` / `MEMBER VU`) theo từng thành viên và hỏi chéo trước khi bảo vệ.


