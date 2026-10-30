# CHANGELOG – LỊCH SỬ CẬP NHẬT DỰ ÁN

## [v0.1.0] – 2026-10-31 (Hoàn thành Chặng 1)
- Khởi tạo thư mục cô lập `Do_An_Tin_Dung/`, `requirements.txt`, `README.md` và trọn bộ 9 file quản lý dự án trong `project_docs/`.
- Xây dựng và chạy thành công `src/01_tai_va_kham_pha_du_lieu.py`:
  - Tự động tải và lưu `data/raw/credit_risk_raw.csv` (`32,581 × 12`).
  - Xuất 3 bảng thống kê Chương I (`01_kiem_tra_chat_luong_du_lieu.csv`, `01_thong_ke_mo_ta_so.csv`, `01_thong_ke_phan_loai.csv`).
  - Xuất 4 biểu đồ EDA sắc nét (`01_phan_bo_nhan_loan_status.png`, `01_phan_bo_bien_so_va_ngoai_lai.png`, `01_ty_le_vo_no_theo_nhom.png`, `01_ma_tran_tuong_quan.png`).
- Xây dựng và chạy thành công `src/02_tien_xu_ly_du_lieu.py`:
  - Làm sạch 165 dòng trùng & 7 dòng ngoại lai vô lý -> còn `32,409` dòng sạch.
  - Tách Stratified Train (`25,927`) / Test (`6,482`) trước tiên.
  - Điền khuyết, mã hóa, chuẩn hóa và rời rạc hóa 3 nhánh dữ liệu phục vụ đúng bản chất 4 thuật toán.
  - Lưu `outputs/models/preprocessor_bundle.joblib`, 3 bảng thông số Chương II và biểu đồ `02_minh_hoa_tien_xu_ly.png`.

## [v0.2.0] – 2026-10-31 (Hoàn thành Chặng 2)
- Xây dựng module `src/id3_tu_cai_dat.py`: cài đặt hoàn chỉnh class `ID3DecisionTree` thuần Python (Entropy cơ số 2, Information Gain, phân nhánh đa hướng, cắt tỉa trước `max_depth` & `min_samples_split`, xuất xác suất `predict_proba` và sơ đồ chữ `xuat_chuoi_cay`).
- Xây dựng và chạy thành công `src/03_huan_luyen_mo_hinh.py`:
  - Khảo sát 5-Fold Stratified CV trên tập Train tìm độ sâu tối ưu chống Overfitting (`ID3 max_depth = 4`, `CART max_depth = 10`).
  - Huấn luyện và lưu đủ 4 mô hình `.joblib` trong `outputs/models/`.
  - Xuất 5 bảng CSV/TXT (`03_khao_sat_do_sau_cv.csv`, `03_he_so_logistic_regression.csv`, `03_ig_nut_goc_id3.csv`, `03_so_do_nhanh_dau_cay_id3.txt`, `03_feature_importance_cart_rf.csv`, `03_cau_hinh_4_mo_hinh.csv`) và 4 biểu đồ PNG Chương III.

## [v0.3.0] – 2026-10-31 (Hoàn thành Chặng 3 – Toàn bộ Code Đồ Án Hoàn Tất!)
- Xây dựng và chạy thành công `src/04_danh_gia_va_so_sanh.py` (Phần 3.5):
  - Đánh giá độc lập 4 mô hình trên tập Test (`6,482` mẫu), xuất bảng tổng hợp `04_bang_so_sanh_4_mo_hinh.csv` và `04_chi_tiet_confusion_matrix_test.csv`.
  - Xuất 3 biểu đồ trọng điểm: `04_confusion_matrix_4_mo_hinh.png`, `04_roc_curve_so_sanh_4_mo_hinh.png`, `04_so_sanh_5_chi_so_test.png`.
- Xây dựng và chạy thành công `src/05_du_doan_khach_hang_moi.py` (Chương IV):
  - Hoàn thiện hàm `predict_loan_status(customer_data)` kết nối đồng bộ với `preprocessor_bundle.joblib`.
  - Kiểm thử thành công trên 3 kịch bản khách hàng thực tế (khách tốt, khách rủi ro cao, khách khuyết thông tin) và lưu bảng `05_ket_qua_du_doan_ho_so_mau.csv`.

## [v1.0.0] – 2026-10-31 (Hoàn thành Chặng 4 – Xuất Báo Cáo Word 4 Chương Chuẩn Học Thuật)
- Xây dựng và chạy thành công `report/xuat_bao_cao_docx.py` sinh file `report/Bao_Cao_Do_An_Hoc_May_Nhom_HaiNam_Thanh_Vu.docx` (`1.79 MB`).
- Báo cáo định dạng chuẩn đại học (Times New Roman 13pt, giãn dòng 1.3), tự động nạp trực tiếp 13 bảng dữ liệu CSV thật và nhúng toàn bộ 12 biểu đồ PNG độ nét cao, đảm bảo độ khớp 100% giữa Code – Bảng số liệu – Báo cáo.
- Sẵn sàng 100% cho Chặng 5 (`ACTIVATE DEFENSE MODE`).


