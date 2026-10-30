"""
SCRIPT TỰ ĐỘNG SINH BÁO CÁO ĐỒ ÁN HỌC MÁY CHUẨN HỌC THUẬT (.DOCX)
Đọc trực tiếp 100% bảng số liệu thật từ `outputs/tables/` và chèn toàn bộ biểu đồ PNG từ
`outputs/figures/` để đảm bảo báo cáo phản ánh chính xác tuyệt đối những gì nhóm chạy trong code.

Đồ án Học máy - Khoa CNTT, Trường Đại học Thủy Lợi
Giảng viên hướng dẫn: Thầy Chiểu
Nhóm thực hiện: Hải Nam - Thành - Vũ
"""

import os
import sys
import pandas as pd
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

REPORT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(REPORT_DIR)
TABLES_DIR = os.path.join(BASE_DIR, "outputs", "tables")
FIGURES_DIR = os.path.join(BASE_DIR, "outputs", "figures")
OUTPUT_DOCX = os.path.join(REPORT_DIR, "Bao_Cao_Do_An_Hoc_May_Nhom_HaiNam_Thanh_Vu.docx")


def set_cell_shading(cell, color_hex="1F4E79"):
    """Tô màu nền cho ô tiêu đề bảng trong Word."""
    shading = OxmlElement("w:shd")
    shading.set(qn("w:fill"), color_hex)
    cell._tc.get_or_add_tcPr().append(shading)


def configure_styles(doc):
    """Thiết lập chuẩn phông chữ Times New Roman, cỡ chữ 13pt chuẩn báo cáo đại học Việt Nam."""
    normal_style = doc.styles["Normal"]
    normal_style.font.name = "Times New Roman"
    normal_style.font.size = Pt(13)
    normal_style._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    normal_style.paragraph_format.line_spacing = 1.3
    normal_style.paragraph_format.space_after = Pt(6)

    for level, size_pt in [(1, 16), (2, 14), (3, 13)]:
        h_style = doc.styles[f"Heading {level}"]
        h_style.font.name = "Times New Roman"
        h_style.font.size = Pt(size_pt)
        h_style.font.bold = True
        h_style.font.color.rgb = RGBColor(0x1F, 0x4E, 0x79)
        h_style._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
        h_style.paragraph_format.space_before = Pt(12 if level == 1 else 8)
        h_style.paragraph_format.space_after = Pt(6)


def add_para(doc, text, bold=False, italic=False, align=WD_ALIGN_PARAGRAPH.JUSTIFY):
    p = doc.add_paragraph()
    p.alignment = align
    run = p.add_run(text)
    run.bold = bold
    run.italic = italic
    return p


def add_bullet(doc, lead_bold, body_text):
    p = doc.add_paragraph(style="List Bullet")
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    r1 = p.add_run(lead_bold)
    r1.bold = True
    p.add_run(body_text)
    return p


def add_formula(doc, formula_text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(6)
    run = p.add_run(formula_text)
    run.italic = True
    run.bold = True
    run.font.size = Pt(12.5)
    return p


def add_df_table(doc, caption, df, col_headers=None):
    """Chèn bảng DataFrame vào Word với dòng tiêu đề xanh đậm chữ trắng rõ đẹp."""
    p_cap = doc.add_paragraph()
    p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_cap.paragraph_format.space_before = Pt(8)
    p_cap.paragraph_format.space_after = Pt(4)
    r_cap = p_cap.add_run(caption)
    r_cap.bold = True
    r_cap.font.size = Pt(11.5)

    headers = col_headers if col_headers is not None else list(df.columns)
    table = doc.add_table(rows=1 + len(df), cols=len(headers))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    # Dòng tiêu đề
    hdr_cells = table.rows[0].cells
    for j, h_text in enumerate(headers):
        hdr_cells[j].text = ""
        p = hdr_cells[j].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(2)
        run = p.add_run(str(h_text))
        run.bold = True
        run.font.size = Pt(10.5)
        run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        set_cell_shading(hdr_cells[j], "1F4E79")

    # Các dòng dữ liệu
    for i, (_, row) in enumerate(df.iterrows()):
        row_cells = table.rows[i + 1].cells
        bg = "F2F6FA" if i % 2 == 1 else None
        for j, val in enumerate(row):
            row_cells[j].text = ""
            p = row_cells[j].paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if j == 0 else WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_after = Pt(2)
            run = p.add_run(str(val))
            run.font.size = Pt(10)
            if bg:
                set_cell_shading(row_cells[j], bg)

    doc.add_paragraph()


def add_figure(doc, image_filename, caption, width_inches=6.1):
    """Chèn ảnh biểu đồ PNG vào báo cáo kèm chú thích chuẩn dưới hình."""
    img_path = os.path.join(FIGURES_DIR, image_filename)
    if not os.path.exists(img_path):
        add_para(doc, f"[Thiếu file ảnh: {image_filename}]", italic=True, align=WD_ALIGN_PARAGRAPH.CENTER)
        return

    p_img = doc.add_paragraph()
    p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_img.paragraph_format.space_before = Pt(6)
    p_img.paragraph_format.space_after = Pt(3)
    p_img.add_run().add_picture(img_path, width=Inches(width_inches))

    p_cap = doc.add_paragraph()
    p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_cap.paragraph_format.space_after = Pt(10)
    r = p_cap.add_run(caption)
    r.italic = True
    r.bold = True
    r.font.size = Pt(11)


def build_report():
    os.makedirs(REPORT_DIR, exist_ok=True)

    # Đọc toàn bộ các bảng số liệu thực nghiệm thật 100% từ thư mục outputs/tables/
    df_quality = pd.read_csv(os.path.join(TABLES_DIR, "01_kiem_tra_chat_luong_du_lieu.csv"))
    df_num_desc = pd.read_csv(os.path.join(TABLES_DIR, "01_thong_ke_mo_ta_so.csv")).rename(columns={"Unnamed: 0": "Bien_So"})
    df_prep_sum = pd.read_csv(os.path.join(TABLES_DIR, "02_tong_ket_tien_xu_ly.csv"))
    df_prep_params = pd.read_csv(os.path.join(TABLES_DIR, "02_tham_so_imputation_va_scaler.csv"))
    df_id3_bins = pd.read_csv(os.path.join(TABLES_DIR, "02_nguong_roi_rac_hoa_id3.csv"))
    df_cv_depth = pd.read_csv(os.path.join(TABLES_DIR, "03_khao_sat_do_sau_cv.csv"))
    df_logreg_w = pd.read_csv(os.path.join(TABLES_DIR, "03_he_so_logistic_regression.csv"))
    df_id3_ig = pd.read_csv(os.path.join(TABLES_DIR, "03_ig_nut_goc_id3.csv"))
    df_feat_imp = pd.read_csv(os.path.join(TABLES_DIR, "03_feature_importance_cart_rf.csv"))
    df_comp_4 = pd.read_csv(os.path.join(TABLES_DIR, "04_bang_so_sanh_4_mo_hinh.csv"))
    df_cm_detail = pd.read_csv(os.path.join(TABLES_DIR, "04_chi_tiet_confusion_matrix_test.csv"))
    df_demo_pred = pd.read_csv(os.path.join(TABLES_DIR, "05_ket_qua_du_doan_ho_so_mau.csv"))

    doc = Document()
    configure_styles(doc)

    # =========================================================================
    # TRANG BÌA
    # =========================================================================
    add_para(doc, "BỘ NÔNG NGHIỆP VÀ MÔI TRƯỜNG", bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)
    add_para(doc, "TRƯỜNG ĐẠI HỌC THỦY LỢI\nKHOA CÔNG NGHỆ THÔNG TIN", bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)
    add_para(doc, "----------------------------------------\n\n", align=WD_ALIGN_PARAGRAPH.CENTER)

    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_sub = p_title.add_run("BÁO CÁO ĐỒ ÁN MÔN HỌC: HỌC MÁY (MACHINE LEARNING)\n\n")
    r_sub.bold = True
    r_sub.font.size = Pt(15)

    r_main = p_title.add_run("ĐỀ TÀI:\nDỰ ĐOÁN KHẢ NĂNG TRẢ NỢ CỦA KHÁCH HÀNG VAY VỐN\n(CREDIT DEFAULT PREDICTION)\n\n")
    r_main.bold = True
    r_main.font.size = Pt(18)
    r_main.font.color.rgb = RGBColor(0x1F, 0x4E, 0x79)

    add_para(doc, "Các thuật toán nghiên cứu & thực nghiệm:\n1. Logistic Regression  |  2. Decision Tree (ID3)\n3. Decision Tree (CART)  |  4. Random Forest\n",
             italic=True, align=WD_ALIGN_PARAGRAPH.CENTER)

    add_para(doc, "Giảng viên hướng dẫn: Thầy Chiểu", bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)
    add_para(doc, "Nhóm sinh viên thực hiện (Năm 3 - Ngành Công nghệ thông tin):\n1. Hải Nam (Trưởng nhóm GitHub - Chương I & Chương II)\n2. Thành (Thuật toán & Mô hình - Chương III: 3.1 - 3.4)\n3. Vũ (Đánh giá thực nghiệm & Chương trình - Phần 3.5 & Chương IV)\n",
             align=WD_ALIGN_PARAGRAPH.CENTER)
    add_para(doc, "\nHà Nội - Năm 2026", bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)
    doc.add_page_break()

    # =========================================================================
    # LỜI CẢM ƠN & DANH MỤC
    # =========================================================================
    doc.add_heading("LỜI CẢM ƠN", level=1)
    add_para(doc, (
        "Lời đầu tiên, nhóm thực hiện gồm ba thành viên Hải Nam, Thành và Vũ xin gửi lời cảm ơn chân thành "
        "và sâu sắc nhất tới Thầy Chiểu – giảng viên giảng dạy và hướng dẫn trực tiếp môn Học máy tại Khoa "
        "Công nghệ thông tin, Trường Đại học Thủy Lợi. Thông qua hệ thống bài giảng lý thuyết rõ ràng trên lớp "
        "cùng các bài tập thực hành từ Hồi quy tuyến tính, Cây quyết định ID3/CART, Hồi quy Logistic đến Học "
        "kết hợp (Ensemble Learning), Thầy đã giúp nhóm xây dựng nền tảng tư duy toán học và lập trình thực "
        "nghiệm đúng bản chất."
    ))
    add_para(doc, (
        "Trong quá trình thực hiện đồ án 'Dự đoán khả năng trả nợ của khách hàng vay vốn', nhóm đặc biệt chú "
        "trọng tính trung thực học thuật: tự cài đặt thuật toán ID3 đa nhánh theo đúng nguyên lý Entropy & "
        "Information Gain, thiết kế quy trình tiền xử lý chống rò rỉ dữ liệu (Anti-Data Leakage) tuyệt đối và "
        "phân tích thực chất từng chỉ số đánh giá dựa trên 100% số liệu chạy thật. Do kiến thức và kinh nghiệm "
        "thực tế của sinh viên năm 3 còn những hạn chế nhất định, nhóm rất mong nhận được những ý kiến đóng góp "
        "quý báu của Thầy để hoàn thiện hơn. Nhóm xin chân thành cảm ơn Thầy!"
    ))

    doc.add_heading("MỤC LỤC TỔNG QUAN & PHÂN CÔNG NHIỆM VỤ", level=1)
    df_team = pd.DataFrame([
        ["1. Hải Nam", "Quản lý GitHub, tải Dataset, phân tích EDA, làm sạch dữ liệu, chống Data Leakage, chuẩn hóa & rời rạc hóa ID3", "Chương I & Chương II"],
        ["2. Thành", "Cơ sở toán học & huấn luyện 4 thuật toán: Logistic Regression, ID3 thuần Python, CART, Random Forest (OOB)", "Chương III (3.1 – 3.4)"],
        ["3. Vũ", "Đánh giá 8 tiêu chí Phần 3.5 (Confusion Matrix, FP/FN, F1, ROC-AUC), hàm predict_loan_status & tổng hợp báo cáo", "Phần 3.5, Chương IV & Kết luận"],
    ], columns=["Thành viên", "Nhiệm vụ kỹ thuật phụ trách", "Phần trình bày chính"])
    add_df_table(doc, "Bảng 0.1: Phân công nhiệm vụ chi tiết trong nhóm thực hiện", df_team)

    doc.add_heading("DANH MỤC BẢNG BIỂU VÀ HÌNH VẼ CHÍNH", level=2)
    add_bullet(doc, "Danh mục Bảng biểu: ",
               "Bảng 1.1 (Mô tả 12 thuộc tính & dữ liệu thiếu); Bảng 1.2 (Thống kê mô tả 7 biến số); "
               "Bảng 2.1 (Tổng kết quy trình làm sạch & chia tập Train/Test); Bảng 2.2 (Tham số Median & StandardScaler tập Train); "
               "Bảng 2.3 (Ngưỡng rời rạc hóa tứ phân vị cho ID3); Bảng 3.1 (Khảo sát 5-Fold CV chọn độ sâu cắt tỉa); "
               "Bảng 3.2 (Top trọng số Logistic Regression); Bảng 3.3 (Information Gain tại nút gốc ID3); "
               "Bảng 3.4 (So sánh Feature Importance CART vs Random Forest); Bảng 3.5 (Tổng hợp kết quả 4 mô hình); "
               "Bảng 3.6 (Phân tích Confusion Matrix & sai lầm tài chính FP vs FN); Bảng 4.1 (Kết quả dự đoán 3 kịch bản khách hàng mới).")
    add_bullet(doc, "Danh mục Hình vẽ: ",
               "Hình 1.1 -> Hình 1.4 (Phân tích khám phá dữ liệu EDA); Hình 2.1 (Minh họa lọc ngoại lai, chuẩn hóa Z-score & rời rạc hóa ID3); "
               "Hình 3.1 (Đường cong 5-Fold CV chống Overfitting); Hình 3.2 (Xếp hạng Information Gain nút gốc ID3); "
               "Hình 3.3 (Cây nhị phân CART 3 tầng đầu); Hình 3.4 (Top 10 Feature Importance của Random Forest); "
               "Hình 3.5 (Confusion Matrix 4 mô hình trên tập Test); Hình 3.6 (Đường cong ROC-AUC); Hình 3.7 (Biểu đồ so sánh 5 độ đo).")
    doc.add_page_break()

    # =========================================================================
    # CHƯƠNG I: MÔ TẢ BÀI TOÁN VÀ PHÂN TÍCH DỮ LIỆU THÔ
    # =========================================================================
    doc.add_heading("CHƯƠNG I. MÔ TẢ BÀI TOÁN VÀ PHÂN TÍCH DỮ LIỆU THÔ", level=1)

    doc.add_heading("1.1. Lý do chọn đề tài và ý nghĩa thực tế của bài toán chấm điểm tín dụng", level=2)
    add_para(doc, (
        "Trong hoạt động ngân hàng thương mại và tổ chức tài chính tiêu dùng, nghiệp vụ cấp tín dụng (cho vay) "
        "luôn là nguồn mang lại doanh thu chủ lực nhưng đồng thời tiềm ẩn rủi ro nợ xấu lớn nhất. Khi một khách "
        "hàng nộp hồ sơ vay vốn, hội đồng tín dụng phải trả lời câu hỏi cốt lõi: Liệu khách hàng này có khả năng "
        "hoàn trả đầy đủ gốc và lãi đúng hạn hay sẽ rơi vào tình trạng vỡ nợ (Default)? Nếu xét duyệt thủ công "
        "hoàn toàn bằng cảm tính, ngân hàng dễ gặp hai thái cực rủi ro: hoặc giải ngân nhầm cho khách hàng không "
        "đủ năng lực trả nợ gây mất trắng vốn gốc, hoặc quá khắt khe làm từ chối nhầm khách hàng uy tín và đánh "
        "mất thị phần."
    ))
    add_para(doc, (
        "Do đó, việc ứng dụng các thuật toán Học máy có giám sát (Supervised Machine Learning) để xây dựng mô "
        "hình dự đoán khả năng trả nợ (Credit Default Prediction) từ dữ liệu lịch sử là bài toán thiết thực, "
        "kinh điển và phù hợp trực tiếp với nội dung môn học Học máy."
    ))

    doc.add_heading("1.2. Nguồn dữ liệu và giải thích chi tiết các đặc trưng", level=2)
    add_para(doc, (
        "Nhóm sử dụng bộ dữ liệu chuẩn quốc tế 'Credit Risk Dataset' được lưu trữ công khai trên hệ thống học "
        "thuật OpenML (Data ID: 43454), tải tự động qua thư viện scikit-learn trong file "
        "`src/01_tai_va_kham_pha_du_lieu.py`. Bộ dữ liệu gốc gồm 32.581 bản ghi hồ sơ vay vốn với 12 cột: "
        "11 đặc trưng đầu vào (7 biến số liên tục và 4 biến phân loại) cùng 1 biến mục tiêu nhị phân `loan_status`. "
        "Toàn bộ 11 đặc trưng đều là thông tin thu thập tại thời điểm thẩm định hồ sơ vay nên đảm bảo không chứa "
        "biến phát sinh sau giải ngân."
    ))
    add_df_table(
        doc,
        "Bảng 1.1: Mô tả chi tiết 12 thuộc tính và tình trạng dữ liệu khuyết trên tập dữ liệu thô",
        df_quality,
        col_headers=["Tên cột", "Loại biến", "Số dòng thiếu", "Tỷ lệ thiếu (%)", "Số giá trị riêng", "Ý nghĩa nghiệp vụ thực tế"],
    )

    doc.add_heading("1.3. Quy ước biến mục tiêu và phân tích khám phá dữ liệu (EDA)", level=2)
    add_para(doc, (
        "Để thống nhất xuyên suốt từ tiền xử lý, huấn luyện đến phân tích Ma trận nhầm lẫn ở Phần 3.5, nhóm "
        "xác minh và quy ước chặt chẽ biến mục tiêu `loan_status` như sau:"
    ))
    add_bullet(doc, "Nhãn 0 (Lớp Negative – Non-default / Trả nợ tốt): ",
               "Khách hàng thực hiện nghĩa vụ trả nợ đầy đủ, đúng hạn. Có 25.473 mẫu, chiếm 78,18% tổng dữ liệu.")
    add_bullet(doc, "Nhãn 1 (Lớp Positive – Default / Vỡ nợ): ",
               "Khách hàng vi phạm nghĩa vụ thanh toán / không trả được nợ (đối tượng rủi ro cần phát hiện). "
               "Có 7.108 mẫu, chiếm 21,82% tổng dữ liệu.")
    add_para(doc, (
        "Tỷ lệ mất cân bằng giữa hai lớp là 3,58 : 1 (78,18% so với 21,82%). Đây là mức mất cân bằng rất đặc "
        "trưng của dữ liệu ngân hàng thực tế: đủ để gây ra hiện tượng ảo tưởng độ chính xác (Accuracy Paradox) "
        "nếu người làm mô hình chỉ nhìn vào chỉ số Accuracy tổng thể."
    ))
    add_figure(doc, "01_phan_bo_nhan_loan_status.png",
               "Hình 1.1: Phân bố số lượng và tỷ lệ phần trăm hai lớp của biến mục tiêu loan_status (N = 32.581)", width_inches=5.8)

    add_df_table(
        doc,
        "Bảng 1.2: Thống kê mô tả 7 biến số liên tục trên tập dữ liệu thô",
        df_num_desc,
        col_headers=["Biến số", "Count", "Mean", "Std", "Min", "Q1 (25%)", "Median (50%)", "Q3 (75%)", "Max", "Skewness"],
    )

    add_para(doc, (
        "Qua Bảng 1.1, Bảng 1.2 và các biểu đồ EDA dưới đây, nhóm rút ra 4 phát hiện quan trọng làm cơ sở trực "
        "tiếp cho thiết kế Tiền xử lý ở Chương II:"
    ))
    add_bullet(doc, "Thứ nhất – Tồn tại bản ghi trùng lặp và ngoại lai nhập lỗi vô lý: ",
               "Phát hiện 165 dòng trùng lặp hoàn toàn trên cả 12 cột; 5 bản ghi có tuổi khách hàng lên tới "
               "123 và 144 tuổi (`person_age > 100`); 2 bản ghi có thâm niên làm việc `person_emp_length = 123` năm "
               "(trong khi khách hàng mới ngoài 20 tuổi).")
    add_bullet(doc, "Thứ hai – Dữ liệu khuyết và phân phối lệch phải mạnh: ",
               "Cột `person_emp_length` thiếu 895 dòng (2,75%) và cột `loan_int_rate` thiếu 3.116 dòng (9,56%). "
               "Đồng thời biến thu nhập `person_income` có độ lệch phải rất lớn (Skewness = 32,87; Max = 6.000.000 USD "
               "kéo Mean = 66.075 USD cao hơn hẳn Median = 55.000 USD).")
    add_bullet(doc, "Thứ ba – Quy luật rủi ro đơn điệu trên các biến phân loại: ",
               "Tỷ lệ vỡ nợ tăng liên tục theo hạng khoản vay `loan_grade` từ hạng A (10,0%), B (16,3%), C (20,7%) "
               "vọt lên D (59,0%), E (64,4%), F (70,5%) và G (98,4%). Nhóm ở nhà thuê (`RENT`) vỡ nợ 31,6%, cao gấp "
               "hơn 4 lần nhóm sở hữu nhà riêng (`OWN`: 7,5%).")
    add_bullet(doc, "Thứ tư – Tương quan tuyến tính: ",
               "Ma trận tương quan Pearson cho thấy `loan_percent_income` (r = +0,38) và `loan_int_rate` (r = +0,34) "
               "tương quan dương mạnh nhất với khả năng vỡ nợ; trong khi `person_age` và `cb_person_cred_hist_length` "
               "tương quan chặt với nhau (r = +0,86), đòi hỏi mô hình tuyến tính phải có điều chuẩn L2.")

    add_figure(doc, "01_phan_bo_bien_so_va_ngoai_lai.png",
               "Hình 1.2: Phân phối các biến số tiêu biểu và phát hiện điểm ngoại lai vô lý trên dữ liệu thô", width_inches=6.2)
    add_figure(doc, "01_ty_le_vo_no_theo_nhom.png",
               "Hình 1.3: Tỷ lệ vỡ nợ (%) theo Hạng khoản vay, Hình thức sở hữu nhà và Lịch sử nợ xấu", width_inches=6.2)
    add_figure(doc, "01_ma_tran_tuong_quan.png",
               "Hình 1.4: Ma trận tương quan tuyến tính Pearson giữa các biến số và trạng thái vỡ nợ", width_inches=5.4)
    doc.add_page_break()

    # =========================================================================
    # CHƯƠNG II: TIỀN XỬ LÝ DỮ LIỆU
    # =========================================================================
    doc.add_heading("CHƯƠNG II. TIỀN XỬ LÝ DỮ LIỆU (CHỐNG RÒ RỈ DỮ LIỆU)", level=1)

    doc.add_heading("2.1. Làm sạch bản ghi trùng lặp và lọc ngoại lai sai logic", level=2)
    add_para(doc, (
        "Tại bước đầu tiên trong `src/02_tien_xu_ly_du_lieu.py`, nhóm loại bỏ 165 dòng trùng lặp hoàn toàn để "
        "ngăn chặn nguy cơ cùng một bản ghi vừa xuất hiện ở tập Train vừa xuất hiện ở tập Test. Tiếp theo, nhóm "
        "lọc bỏ 7 bản ghi mang giá trị ngoại lai bất khả thi về mặt sinh học và lao động (giữ điều kiện hợp lệ: "
        "`18 <= person_age <= 100`, `0 <= person_emp_length <= 60` và `person_emp_length <= person_age - 14`, "
        "đồng thời vẫn bảo toàn các dòng đang khuyết `NaN` để chuyển sang bước điền khuyết). Sau bước làm sạch, "
        "bộ dữ liệu còn 32.409 bản ghi hợp lệ. Quy tắc lọc này chỉ kiểm tra điều kiện logic độc lập trên từng "
        "dòng, không sử dụng bất kỳ đại lượng thống kê toàn cục nào nên hoàn toàn không gây Data Leakage."
    ))

    doc.add_heading("2.2. Chia tập Train/Test phân tầng trước tiên để chống Data Leakage tuyệt đối", level=2)
    add_para(doc, (
        "Nguyên tắc cốt lõi của Học máy thực nghiệm là tập Kiểm tra (Test set) phải đóng vai trò đại diện cho dữ "
        "liệu tương lai hoàn toàn chưa từng biết. Do đó, nhóm thực hiện chia dữ liệu theo tỷ lệ 80% Train "
        "(25.927 mẫu) – 20% Test (6.482 mẫu) bằng hàm `train_test_split(..., test_size=0.20, stratify=y, "
        "random_state=42)` NGAY TRƯỚC toàn bộ các bước tính tham số điền khuyết, mã hóa, chuẩn hóa và rời rạc "
        "hóa. Tham số `stratify=y` giữ tỷ lệ vỡ nợ đồng nhất gần như tuyệt đối giữa tập Train (21,87%) và tập "
        "Test (21,88%)."
    ))
    add_df_table(
        doc,
        "Bảng 2.1: Tổng kết số lượng mẫu qua các bước làm sạch và chia phân tầng Train/Test",
        df_prep_sum,
        col_headers=["Bước xử lý", "Số dòng dữ liệu", "Ghi chú chi tiết"],
    )

    doc.add_heading("2.3. Xử lý dữ liệu thiếu (Imputation)", level=2)
    add_para(doc, (
        "Đối với 2 biến số bị khuyết (`person_emp_length` và `loan_int_rate`), nhóm sử dụng `SimpleImputer` với "
        "chiến lược Trung vị (`strategy='median'`) thay vì Trung bình cộng (`mean`). Lý do là các biến tài chính "
        "và thâm niên làm việc có phân phối lệch phải mạnh; trung bình cộng dễ bị kéo lệch bởi nhóm giá trị lớn, "
        "trong khi trung vị (phân vị thứ 50%) bền vững trước nhiễu và đại diện đúng cho khách hàng điển hình. "
        "Đối với 4 biến phân loại, nhóm cấu hình `SimpleImputer(strategy='most_frequent')` (Mode) để đảm bảo khi "
        "triển khai hàm dự đoán thực tế ở Chương IV, nếu hồ sơ mới bị bỏ trống vẫn được xử lý an toàn. Toàn bộ "
        "Imputer chỉ gọi `.fit()` trên `X_train` rồi dùng tham số đó `.transform()` cho cả `X_train` và `X_test`."
    ))
    add_df_table(
        doc,
        "Bảng 2.2: Các tham số Trung vị (Imputation) và Trung bình / Độ lệch chuẩn (StandardScaler) học từ tập Train",
        df_prep_params,
        col_headers=["Biến số liên tục", "Median điền khuyết (Train)", "Mean chuẩn hóa (Train)", "Std chuẩn hóa (Train)"],
    )

    doc.add_heading("2.4. Mã hóa biến phân loại và Chuẩn hóa thang đo theo đặc thù thuật toán", level=2)
    add_bullet(doc, "Mã hóa biến phân loại (Encoding): ",
               "Biến `loan_grade` có thứ tự rủi ro tự nhiên rõ rệt từ A đến G (minh chứng ở Hình 1.3a) nên được "
               "mã hóa thứ tự (Ordinal Encoding: A=0, B=1, ..., G=6). Biến nhị phân `cb_person_default_on_file` "
               "mã hóa N=0, Y=1. Hai biến định danh không có thứ tự cao thấp là `person_home_ownership` (4 nhóm) "
               "và `loan_intent` (6 nhóm) được mã hóa One-Hot Encoding để tránh tạo ra thứ tự số học giả tạo cho "
               "Logistic Regression. Tổng số chiều đặc trưng sau mã hóa là 19 cột.")
    add_bullet(doc, "Chuẩn hóa Z-score (`StandardScaler`) cho Logistic Regression vs. Giữ đơn vị gốc cho CART & Random Forest: ",
               "Logistic Regression tối ưu tổ hợp tuyến tính z = w^T x + b kèm điều chuẩn L2 nên bắt buộc phải đưa "
               "các biến về cùng thang đo chuẩn (trung bình 0, độ lệch chuẩn 1) học từ tập Train. Ngược lại, cây "
               "quyết định CART và Random Forest tìm ngưỡng chia nhị phân x_j <= t chỉ phụ thuộc vào thứ tự sắp "
               "xếp của biến (bất biến với phép biến đổi tuyến tính đồng biến), nên giữ nguyên đơn vị gốc (USD, %, "
               "năm) giúp cây quyết định dễ đọc và giải thích trực tiếp bằng ngôn ngữ ngân hàng.")

    doc.add_heading("2.5. Rời rạc hóa dữ liệu liên tục (Discretization) phục vụ riêng cho ID3", level=2)
    add_para(doc, (
        "Theo đúng lý thuyết Chương 5 môn Học máy, thuật toán ID3 nguyên bản phân nhánh đa hướng trên các thuộc "
        "tính rời rạc; nếu đưa trực tiếp số thực liên tục có hàng nghìn giá trị khác nhau vào ID3, độ lợi thông "
        "tin (Information Gain) sẽ bị thiên vị giả tạo cho các cột nhiều giá trị và gây Overfitting nghiêm trọng. "
        "Do đó, nhóm rời rạc hóa 7 biến số liên tục thành 4 khoảng phân vị (`1_Thap`, `2_Trung_binh`, `3_Kha_cao`, "
        "`4_Cao`) dựa trên các mốc tứ phân vị Q1 (25%), Q2 (50%), Q3 (75%) tính duy nhất trên tập Train, với hai "
        "đầu mút mở rộng ra `(-inf, +inf)` để bao phủ mọi giá trị ngoại biên ở tập Test."
    ))
    df_bins_display = df_id3_bins[["Bien_So", "Khoang_1_Thap", "Khoang_2_Trung_binh", "Khoang_3_Kha_cao", "Khoang_4_Cao"]]
    add_df_table(
        doc,
        "Bảng 2.3: Bảng quy đổi ngưỡng rời rạc hóa 4 khoảng tứ phân vị (học từ tập Train) phục vụ thuật toán ID3",
        df_bins_display,
        col_headers=["Biến số", "1_Thap (<= Q1)", "2_Trung_binh (Q1 - Q2]", "3_Kha_cao (Q2 - Q3]", "4_Cao (> Q3)"],
    )
    add_figure(doc, "02_minh_hoa_tien_xu_ly.png",
               "Hình 2.1: Minh họa trực quan kết quả lọc ngoại lai, chuẩn hóa Z-score và rời rạc hóa biến số tại Chương II", width_inches=6.2)
    doc.add_page_break()

    # =========================================================================
    # CHƯƠNG III: CƠ SỞ LÝ THUYẾT VÀ THỰC NGHIỆM
    # =========================================================================
    doc.add_heading("CHƯƠNG III. CƠ SỞ LÝ THUYẾT VÀ THỰC NGHIỆM", level=1)

    # --- 3.1 Logistic Regression ---
    doc.add_heading("3.1. Thuật toán Hồi quy Logistic (Logistic Regression)", level=2)
    add_para(doc, (
        "Hồi quy Logistic là mô hình phân lớp tuyến tính xác suất. Với vector đặc trưng đầu vào đã chuẩn hóa x, "
        "mô hình tính tổ hợp tuyến tính và nén qua hàm kích hoạt Sigmoid để ước lượng xác suất khách hàng vỡ nợ:"
    ))
    add_formula(doc, "z = w^T x + b,     P(y = 1 | x) = sigma(z) = 1 / (1 + e^(-z))")
    add_para(doc, (
        "Với ngưỡng phân loại chuẩn Threshold = 0,5: nếu P(y = 1 | x) >= 0,5 (tương đương z >= 0), mô hình dự "
        "đoán nhãn 1 (Vỡ nợ); ngược lại dự đoán nhãn 0 (Trả nợ tốt). Để ước lượng bộ tham số (w, b), mô hình cực "
        "tiểu hóa hàm mất mát Binary Cross-Entropy (Log Loss) có thành phần điều chuẩn L2:"
    ))
    add_formula(doc, "J(w, b) = -(1/N) * sum_{i=1..N} [ y_i ln(y_hat_i) + (1 - y_i) ln(1 - y_hat_i) ] + (lambda / 2) * ||w||_2^2")
    add_para(doc, (
        "Lý do dùng hàm Binary Cross-Entropy thay vì MSE là vì khi ghép hàm phi tuyến Sigmoid vào MSE, hàm mục "
        "tiêu trở thành hàm không lồi (non-convex) dễ kẹt ở cực tiểu địa phương; trong khi Binary Cross-Entropy "
        "là hàm lồi nghiêm ngặt đảm bảo hội tụ tới nghiệm tối ưu toàn cục. Nhóm cài đặt bằng lớp "
        "`LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000, random_state=42)` của `scikit-learn`. Hệ số "
        "chặn học được là b = -1,5989; Top 8 trọng số có độ lớn |w_j| mạnh nhất được trích xuất tại Bảng 3.1."
    ))
    add_df_table(
        doc,
        "Bảng 3.1: Top 8 đặc trưng có trọng số |w_j| ảnh hưởng lớn nhất trong mô hình Logistic Regression",
        df_logreg_w.head(8),
        col_headers=["Đặc trưng", "Trọng số w_j", "|w_j|", "Odds Ratio (e^w)", "Chiều tác động tới rủi ro"],
    )

    # --- 3.2 Decision Tree ID3 ---
    doc.add_heading("3.2. Thuật toán Cây quyết định – ID3 (Tự cài đặt thuần Python)", level=2)
    add_para(doc, (
        "Thuật toán ID3 (Iterative Dichotomiser 3 – Quinlan, 1986) xây dựng cây quyết định phân lớp đa nhánh theo "
        "chiến lược tham lam từ trên xuống dựa trên Độ hỗn loạn thông tin (Entropy) và Độ lợi thông tin "
        "(Information Gain):"
    ))
    add_formula(doc, "H(S) = - sum_{c in {0, 1}} p_c * log2(p_c)   (đơn vị: bit)")
    add_formula(doc, "Gain(S, A) = H(S) - H(A, S) = H(S) - sum_{k=1..K} (|S_k| / |S|) * H(S_k)")
    add_para(doc, (
        "Lưu ý đồng bộ lý thuyết trên lớp: Trong slide bài giảng của thầy Chiểu, một số ví dụ tính Entropy theo "
        "logarit tự nhiên ln (đơn vị nat), còn sách bài tập dùng log2 (đơn vị bit). Do log2(p) = ln(p)/ln(2) chỉ "
        "khác nhau hằng số nhân dương 1/ln(2), thứ tự xếp hạng Information Gain và cấu trúc cây ID3 hoàn toàn "
        "giống hệt nhau 100%. Đặc biệt, vì lớp `DecisionTreeClassifier` trong `scikit-learn` thực chất cài đặt "
        "cây nhị phân CART, nhóm đã tự lập trình class `ID3DecisionTree` thuần Python trong file "
        "`src/id3_tu_cai_dat.py` chia đa nhánh thực sự trên 11 thuộc tính rời rạc hóa, có giới hạn `max_depth`, "
        "`min_samples_split=25` và lưu tỷ lệ mẫu lớp 1 ở nút lá để xuất xác suất `predict_proba` vẽ đường cong ROC."
    ))
    add_para(doc, (
        "Để chọn độ sâu cắt tỉa có căn cứ khoa học mà không chạm vào tập Test, nhóm chạy kiểm định chéo phân tầng "
        "5-Fold Stratified CV trên tập Train (Bảng 3.2 & Hình 3.1). Kết quả cho thấy ID3 đạt đỉnh CV ROC-AUC = "
        "0,8843 tại `max_depth = 4` (tạo ra 477 nút và 370 nút lá); nếu tăng độ sâu lên 6–11, điểm Train tăng lên "
        "0,9498 nhưng điểm CV giảm xuống 0,8643 do chia đa nhánh làm các nút lá bị xé quá nhỏ (Overfitting)."
    ))
    add_df_table(
        doc,
        "Bảng 3.2: Kết quả khảo sát 5-Fold Stratified CV trên tập Train để chọn max_depth cho ID3 và CART",
        df_cv_depth,
        col_headers=["Mô hình", "Giới hạn max_depth", "Train Acc", "5-Fold CV Acc", "Train ROC-AUC", "5-Fold CV ROC-AUC"],
    )
    add_figure(doc, "03_chon_do_sau_chong_overfitting.png",
               "Hình 3.1: Đường cong Train vs 5-Fold CV ROC-AUC theo độ sâu max_depth của cây ID3 và cây CART", width_inches=6.2)

    add_df_table(
        doc,
        "Bảng 3.3: Bảng xếp hạng Information Gain (bit) của 11 thuộc tính tại Nút gốc cây ID3 (H(S) = 0,7578 bit)",
        df_id3_ig,
        col_headers=["Hạng", "Thuộc tính", "Entropy gốc H(S)", "Information Gain (bit)", "Entropy còn lại H(A, S)"],
    )
    add_figure(doc, "03_information_gain_nut_goc_id3.png",
               "Hình 3.2: Xếp hạng Độ lợi thông tin (Information Gain) của 11 thuộc tính tại Nút gốc cây ID3", width_inches=6.0)

    # --- 3.3 Decision Tree CART ---
    doc.add_heading("3.3. Thuật toán Cây quyết định – CART (Classification and Regression Tree)", level=2)
    add_para(doc, (
        "Khác với ID3 chia đa nhánh trên thuộc tính rời rạc dựa vào Entropy, thuật toán CART (Breiman et al., "
        "1984) luôn xây dựng cây nhị phân (mỗi nút cha chỉ tách thành đúng 2 nút con Đúng/Sai) và đo độ tạp chất "
        "bằng chỉ số Gini Impurity:"
    ))
    add_formula(doc, "Gini(S) = 1 - sum_{c in {0, 1}} p_c^2,     Gini_split(S) = (|S_trai|/|S|) Gini(S_trai) + (|S_phai|/|S|) Gini(S_phai)")
    add_para(doc, (
        "Tại mỗi nút, CART duyệt các ngưỡng cắt trung điểm x_j <= t ngay trên biến số thực (không cần rời rạc hóa "
        "trước nên bảo toàn độ phân giải chi tiết của dữ liệu số) để tìm phép chia có Gini_split nhỏ nhất. Dựa "
        "trên khảo sát 5-Fold CV ở Bảng 3.2, nhóm chọn `max_depth = 10` kết hợp `min_samples_leaf = 10`, thu được "
        "cây nhị phân gồm 140 nút lá. Hình 3.3 minh họa 3 tầng đầu tiên của cây CART: nút gốc cắt ngay tại "
        "`loan_percent_income <= 0.305`; đặc biệt nhánh khách hàng vay trên 30,5% thu nhập và đang ở nhà thuê "
        "(`RENT`) chiếm 7,2% tập dữ liệu có tỷ lệ vỡ nợ lên tới 100% (Gini = 0,0)."
    ))
    add_figure(doc, "03_cay_quyet_dinh_cart_minh_hoa.png",
               "Hình 3.3: Trực quan hóa 3 tầng đầu tiên của cây quyết định nhị phân CART (Gini Impurity)", width_inches=6.3)

    # --- 3.4 Random Forest ---
    doc.add_heading("3.4. Thuật toán Rừng ngẫu nhiên (Random Forest)", level=2)
    add_para(doc, (
        "Một cây quyết định đơn lẻ thường có phương sai (Variance) cao: chỉ cần tập dữ liệu huấn luyện biến động "
        "nhỏ, các nhánh phía dưới có thể thay đổi mạnh. Random Forest (Breiman, 2001) khắc phục triệt để hạn chế "
        "này nhờ kết hợp hai tầng ngẫu nhiên hóa:"
    ))
    add_bullet(doc, "1. Kỹ thuật lấy mẫu Bootstrap & Đánh giá Out-of-Bag (OOB): ",
               "Từ N = 25.927 mẫu tập Train, mỗi cây trong 200 cây được huấn luyện trên một mẫu rút ngẫu nhiên có "
               "hoàn lại kích thước N. Theo lý thuyết Chương 8, xác suất một mẫu không xuất hiện trong mẫu Bootstrap "
               "là (1 - 1/N)^N -> 1/e ~ 36,8%. Tập ~36,8% mẫu ngoài túi này tạo thành điểm đánh giá nội tại OOB "
               "Accuracy = 0,9320 (93,20%) trên mô hình của nhóm.")
    add_bullet(doc, "2. Lấy mẫu ngẫu nhiên tập con đặc trưng (Feature Subspace): ",
               "Tại mỗi nút phân nhánh, thay vì xét cả 19 đặc trưng, thuật toán chỉ chọn ngẫu nhiên m = sqrt(19) ~ 4 "
               "đặc trưng ứng viên. Cơ chế này phá vỡ sự thống trị của một vài biến mạnh, làm giảm tương quan giữa "
               "các cây và giảm mạnh phương sai khi lấy trung bình xác suất dự đoán của cả 200 cây.")
    add_para(doc, (
        "Bằng chứng thực nghiệm rất rõ nét thể hiện ở Bảng 3.4: trên một cây CART đơn lẻ, biến lãi suất "
        "`loan_int_rate` có độ quan trọng chỉ 0,0090 do bị `loan_grade` che khuất; nhưng trong Random Forest, nhờ "
        "Feature Subspace, `loan_int_rate` được khai thác mạnh mẽ với độ quan trọng tăng gấp 11 lần lên 0,0997!"
    ))
    add_df_table(
        doc,
        "Bảng 3.4: Đối chiếu mức độ quan trọng của Top 10 đặc trưng giữa CART đơn lẻ và Random Forest",
        df_feat_imp.head(10),
        col_headers=["Đặc trưng", "Độ quan trọng trên CART", "Độ quan trọng trên Random Forest"],
    )
    add_figure(doc, "03_top10_feature_importance_rf.png",
               "Hình 3.4: Xếp hạng 10 đặc trưng quan trọng nhất trong dự báo rủi ro tín dụng (Random Forest)", width_inches=6.0)

    # --- 3.5 Kết quả thực nghiệm & đánh giá ---
    doc.add_heading("3.5. Kết quả thực nghiệm, đánh giá và thảo luận chuyên sâu", level=2)
    add_para(doc, (
        "Sau khi hoàn tất huấn luyện trên tập Train, cả 4 mô hình được mở niêm phong đánh giá độc lập một lần duy "
        "nhất trên tập Test gồm N = 6.482 khách hàng (5.064 khách trả tốt – 78,12% và 1.418 khách vỡ nợ – 21,88%)."
    ))

    doc.add_heading("3.5.1. Ý nghĩa các độ đo đánh giá trong quản trị rủi ro tín dụng", level=3)
    add_bullet(doc, "1. Ma trận nhầm lẫn (Confusion Matrix – TP, TN, FP, FN): ",
               "Với lớp Positive (1) = Vỡ nợ và Negative (0) = Trả nợ tốt: TP là số khách vỡ nợ được mô hình phát "
               "hiện đúng (từ chối cho vay kịp thời, bảo toàn vốn); TN là số khách trả tốt được duyệt vay đúng "
               "(thu lãi an toàn). Ngược lại, FP (False Positive – Báo động giả) là đánh nhầm khách tốt thành vỡ "
               "nợ khiến ngân hàng từ chối cho vay và mất chi phí cơ hội thu lãi; còn FN (False Negative – Bỏ lọt "
               "nợ xấu) là đánh nhầm khách vỡ nợ thành khách tốt, dẫn tới giải ngân tiền thật và mất trắng cả gốc "
               "lẫn lãi! Về mặt tài chính ngân hàng, 1 sai lầm FN gây thiệt hại nặng gấp khoảng 10 lần 1 sai lầm FP.")
    add_bullet(doc, "2. Accuracy và Nghịch lý độ chính xác (Accuracy Paradox): ",
               "Accuracy = (TP + TN) / N đo tỷ lệ đoán đúng tổng thể. Tuy nhiên trên tập Test của nhóm, một mô "
               "hình 'vô dụng' duyệt bừa 100% hồ sơ (luôn đoán nhãn 0) vẫn đạt Accuracy = 5.064 / 6.482 = 78,12%, "
               "nhưng bỏ lọt toàn bộ 1.418 ca vỡ nợ (Recall = 0%). Vì vậy không thể dùng Accuracy làm thước đo duy nhất.")
    add_bullet(doc, "3. Precision (Độ chuẩn xác = TP / (TP + FP)): ",
               "Trong số những khách hàng bị mô hình phán là 'sẽ vỡ nợ', bao nhiêu phần trăm thực sự vỡ nợ? "
               "Precision cao giúp ngân hàng bảo vệ tập khách hàng tốt, không từ chối oan khách uy tín.")
    add_bullet(doc, "4. Recall / Sensitivity (Độ nhạy = TP / (TP + FN)) & Đánh đổi Precision – Recall: ",
               "Trong tổng số 1.418 khách thực sự vỡ nợ ngoài đời, mô hình bắt trúng được bao nhiêu phần trăm? "
               "Ngân hàng rất coi trọng Recall để chặn nợ xấu, nhưng nếu hạ ngưỡng cực đoan để đẩy Recall lên 100% "
               "bất chấp Precision thì mọi khách hàng đều bị từ chối và ngân hàng tê liệt hoạt động cho vay.")
    add_bullet(doc, "5. F1-Score (= 2 * Precision * Recall / (Precision + Recall)): ",
               "Sử dụng trung bình điều hòa (Harmonic Mean) thay vì trung bình cộng vì trung bình điều hòa phạt rất "
               "nặng nếu một trong hai chỉ số Precision hoặc Recall quá thấp, buộc mô hình phải cân bằng cả hai vế.")
    add_bullet(doc, "6. Đường cong ROC và diện tích ROC-AUC: ",
               "Đường cong ROC biểu diễn sự đánh đổi giữa trục tung TPR (Recall) và trục hoành FPR (= FP / (FP + TN)) "
               "khi dịch chuyển ngưỡng phân loại Threshold từ 0 đến 1. Khác với Accuracy hay F1 chỉ tính tại một "
               "ngưỡng 0,5 cố định, diện tích dưới đường cong AUC đo năng lực phân hạng rủi ro tổng quát của mô "
               "hình trên mọi ngưỡng (AUC = 0,5 tương đương đoán ngẫu nhiên; càng gần 1,0 khả năng tách biệt khách "
               "tốt và khách xấu càng xuất sắc).")

    doc.add_heading("3.5.2. Bảng tổng hợp kết quả thực nghiệm và Trực quan hóa", level=3)
    add_df_table(
        doc,
        "Bảng 3.5: Bảng so sánh tổng hợp 4 thuật toán trên tập Train và tập Test độc lập (Ngưỡng = 0,5)",
        df_comp_4,
        col_headers=["Mô hình", "Train Acc", "Test Acc", "Precision", "Recall", "F1-Score", "Train AUC", "Test ROC-AUC", "Độ lệch AUC"],
    )
    add_df_table(
        doc,
        "Bảng 3.6: Chi tiết Confusion Matrix và tỷ lệ sai lầm tài chính (FP vs FN) trên tập Test (N = 6.482)",
        df_cm_detail,
        col_headers=["Mô hình", "TN (Duyệt đúng)", "FP (Từ chối nhầm)", "FN (Bỏ lọt nợ xấu)", "TP (Bắt trúng nợ xấu)", "Tỷ lệ FP (%)", "Tỷ lệ bỏ lọt FN (%)"],
    )

    add_figure(doc, "04_confusion_matrix_4_mo_hinh.png",
               "Hình 3.5: So sánh trực quan Confusion Matrix của 4 thuật toán trên tập Test độc lập (N = 6.482)", width_inches=6.2)
    add_figure(doc, "04_roc_curve_so_sanh_4_mo_hinh.png",
               "Hình 3.6: Đường cong ROC và diện tích AUC của 4 thuật toán trên tập Test", width_inches=5.6)
    add_figure(doc, "04_so_sanh_5_chi_so_test.png",
               "Hình 3.7: Biểu đồ cột so sánh 5 độ đo đánh giá trên tập Test của 4 thuật toán", width_inches=6.2)

    doc.add_heading("3.5.3. Phân tích và thảo luận chuyên sâu từ số liệu thực nghiệm", level=3)
    add_bullet(doc, "1. Kiểm soát Overfitting hiệu quả trên cả 4 mô hình: ",
               "Độ chênh lệch giữa Train Accuracy và Test Accuracy của cả 4 thuật toán đều dưới 0,7% (Logistic: "
               "85,46% vs 84,85%; ID3: 88,26% vs 88,49%; CART: 93,44% vs 93,20%; Random Forest: 93,66% vs 93,52%). "
               "Đặc biệt điểm Test Accuracy của Random Forest (93,52%) khớp sát với điểm OOB (93,20%), kiểm chứng "
               "hoàn hảo lý thuyết Out-of-Bag của Chương 8.")
    add_bullet(doc, "2. Lý giải sự cải thiện bậc thang từ Logistic -> ID3 -> CART -> Random Forest: ",
               "Logistic Regression có kết quả thấp nhất (F1 = 0,5902; AUC = 0,8638; bỏ lọt tới FN = 711 ca nợ xấu "
               "~ 50,14%) vì siêu phẳng tuyến tính không nắm bắt được các luật tương tác phi tuyến ngưỡng gắt (ví "
               "dụ: vay > 30,5% thu nhập KẾT HỢP ở nhà thuê -> vỡ nợ 100%). Cây ID3 tự cài đặt khắc phục được tính "
               "phi tuyến nên đẩy Recall vọt lên 65,30% và F1 lên 0,7129 (giảm 219 ca bỏ lọt nợ xấu), nhưng do phải "
               "gom các biến số thành 4 khoảng tứ phân vị thô nên mất thông tin chi tiết tại biên cắt. Cây nhị phân "
               "CART cắt trực tiếp trên số thực nên tạo bước nhảy vọt lớn lên F1 = 0,8231 và AUC = 0,9178.")
    add_bullet(doc, "3. So sánh tinh tế giữa CART và Random Forest ở góc độ nghiệp vụ ngân hàng: ",
               "Tại ngưỡng mặc định 0,5, CART bắt được nhiều hơn Random Forest 12 ca nợ xấu (TP = 1.026 vs 1.014; "
               "Recall = 72,36% vs 71,51%), nhưng đổi lại Random Forest giảm số khách tốt bị từ chối oan (FP) hơn "
               "3 lần – từ 49 người xuống chỉ còn đúng 16 người trên 5.064 khách tốt (Precision đạt 98,45%), giúp "
               "Random Forest dẫn đầu về cả Test Accuracy (93,52%), F1-Score (0,8284) và đặc biệt là ROC-AUC "
               "(0,9327). Nhờ đường cong ROC nằm cao hơn toàn cục, trong giai đoạn kinh tế biến động cần siết chặt "
               "nợ xấu, ngân hàng chỉ cần hạ nhẹ ngưỡng quyết định của Random Forest xuống khoảng 0,35–0,40 là có "
               "thể nâng Recall vượt trên 76% mà vẫn giữ Precision rất cao.")
    add_bullet(doc, "4. Đánh đổi giữa Khả năng giải thích (White-box) và Độ chính xác (Ensemble): ",
               "Mặc dù Random Forest đạt độ chính xác và độ ổn định cao nhất, tổ hợp 200 cây đóng vai trò gần như "
               "'hộp đen' khó giải thích từng luật đơn lẻ cho khách hàng bị từ chối. Ngược lại, Logistic Regression "
               "(qua hệ số Odds Ratio) và ID3 / CART (qua nhánh luật Nếu – Thì tường minh) có tính minh bạch rất "
               "cao, thường được các ngân hàng dùng phối hợp cùng Random Forest trong quy trình thẩm định thực tế.")
    doc.add_page_break()

    # =========================================================================
    # CHƯƠNG IV: XÂY DỰNG CHƯƠNG TRÌNH
    # =========================================================================
    doc.add_heading("CHƯƠNG IV. XÂY DỰNG CHƯƠNG TRÌNH DỰ ĐOÁN", level=1)

    doc.add_heading("4.1. Tổ chức cấu trúc mã nguồn chương trình", level=2)
    add_para(doc, (
        "Chương trình được thiết kế theo nguyên tắc đơn giản, rõ ràng, chuẩn học thuật (KISS) và tách thành 6 "
        "module Python đánh số tuần tự trong thư mục cô lập `Do_An_Tin_Dung/src/`:"
    ))
    add_bullet(doc, "`01_tai_va_kham_pha_du_lieu.py`: ", "Tải tự động dữ liệu OpenML, kiểm tra chất lượng và xuất 4 biểu đồ EDA.")
    add_bullet(doc, "`02_tien_xu_ly_du_lieu.py`: ", "Làm sạch, chia Stratified 80/20, Impute, Encode, Scale, Binning và đóng gói `preprocessor_bundle.joblib`.")
    add_bullet(doc, "`id3_tu_cai_dat.py`: ", "Cài đặt class `ID3DecisionTree` thuần Python hỗ trợ cắt tỉa `max_depth` và `predict_proba`.")
    add_bullet(doc, "`03_huan_luyen_mo_hinh.py`: ", "Khảo sát 5-Fold CV chọn siêu tham số, huấn luyện và lưu 4 mô hình `.joblib`.")
    add_bullet(doc, "`04_danh_gia_va_so_sanh.py`: ", "Tính toán trọn bộ chỉ số đánh giá Phần 3.5 và vẽ Confusion Matrix, ROC-AUC.")
    add_bullet(doc, "`05_du_doan_khach_hang_moi.py`: ", "Cung cấp hàm `predict_loan_status(customer_data)` cho ứng dụng thực tế.")

    doc.add_heading("4.2. Quy trình lưu trữ Pipeline và hàm `predict_loan_status(customer_data)`", level=2)
    add_para(doc, (
        "Để đảm bảo một hồ sơ khách hàng mới khi đưa vào dự đoán được biến đổi đồng nhất 100% với lúc huấn luyện "
        "(chống lỗi Training-Serving Skew), hàm `predict_loan_status(customer_data, model_name='random_forest', "
        "threshold=0.5)` thực hiện tuần tự 4 bước: (1) Nạp gói tham số `preprocessor_bundle.joblib`; (2) Tự động "
        "điền khuyết bằng đúng Median/Mode tập Train nếu khách hàng bỏ trống thông tin; (3) Biến đổi sang đúng "
        "định dạng của mô hình được chọn (Z-score 19 cột cho Logistic, đơn vị gốc 19 cột cho CART/RF, hoặc 11 cột "
        "khoảng tứ phân vị cho ID3); (4) Trả về nhãn dự đoán (0/1), xác suất vỡ nợ P(y=1|x), phân tầng mức độ rủi "
        "ro (Thấp / Trung bình / Cao) và khuyến nghị xét duyệt."
    ))

    doc.add_heading("4.3. Kết quả chạy thử minh họa trên các hồ sơ khách hàng mẫu", level=2)
    add_para(doc, (
        "Nhóm tiến hành kiểm thử hàm `predict_loan_status` trên 3 kịch bản thực tế: (1) Hồ sơ khách hàng tốt (30 "
        "tuổi, thu nhập 85.000 USD, sở hữu nhà riêng, hạng A, vay 8.000 USD cho giáo dục); (2) Hồ sơ khách hàng "
        "rủi ro cao (22 tuổi, thu nhập 24.000 USD, ở nhà thuê, hạng E, vay 10.000 USD chiếm 42% thu nhập để đảo "
        "nợ, từng có lịch sử nợ xấu); và (3) Hồ sơ bị bỏ trống lãi suất và thâm niên làm việc nhằm kiểm chứng cơ "
        "chế tự động điền khuyết. Kết quả thực chạy từ `05_du_doan_khach_hang_moi.py` được trình bày tại Bảng 4.1."
    ))
    add_df_table(
        doc,
        "Bảng 4.1: Kết quả chạy thử hàm predict_loan_status(customer_data) với cả 4 mô hình trên 3 kịch bản",
        df_demo_pred,
        col_headers=["Hồ sơ khách hàng kiểm thử", "Mô hình dự đoán", "Nhãn", "Xác suất vỡ nợ", "Mức rủi ro", "Khuyến nghị tín dụng"],
    )

    # =========================================================================
    # KẾT LUẬN & TÀI LIỆU THAM KHẢO
    # =========================================================================
    doc.add_heading("KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN", level=1)
    add_bullet(doc, "1. Kết quả đạt được: ",
               "Nhóm đã xây dựng hoàn chỉnh quy trình Học máy chuẩn mực cho bài toán dự đoán rủi ro tín dụng trên "
               "32.581 hồ sơ thực tế, đảm bảo không rò rỉ dữ liệu, triển khai đúng bản chất toán học cả 4 thuật "
               "toán (Logistic Regression, ID3 tự cài đặt đa nhánh, CART và Random Forest đạt Test Accuracy 93,52%, "
               "Precision 98,45%, F1-Score 0,8284 và ROC-AUC 0,9327), đồng thời xây dựng thành công hàm dự đoán "
               "khách hàng mới chạy ổn định.")
    add_bullet(doc, "2. Hạn chế thực tế của đề tài: ",
               "Tại ngưỡng phân loại mặc định 0,5, mô hình tốt nhất vẫn còn bỏ lọt khoảng 28% số ca vỡ nợ tinh vi "
               "(những khách hàng có hồ sơ ban đầu đẹp nhưng gặp biến cố tài chính sau giải ngân); bộ dữ liệu tĩnh "
               "chưa phản ánh biến động kinh tế vĩ mô theo thời gian (lạm phát, biến động lãi suất thị trường).")
    add_bullet(doc, "3. Hướng phát triển tiếp theo: ",
               "(i) Áp dụng học nhạy cảm chi phí (Cost-Sensitive Learning / điều chỉnh `class_weight` và tối ưu "
               "ngưỡng quyết định Threshold dựa trên ma trận lợi nhuận – tổn thất thực tế của ngân hàng); "
               "(ii) Tích hợp phương pháp giải thích mô hình hậu nghiệm (SHAP) để giải thích minh bạch từng phán "
               "quyết của Random Forest; (iii) Đóng gói hàm `predict_loan_status` thành giao diện tra cứu nội bộ "
               "hỗ trợ cán bộ thẩm định tín dụng.")

    doc.add_heading("TÀI LIỆU THAM KHẢO", level=1)
    refs = [
        "[1] Bài giảng và Tài liệu hướng dẫn thực hành môn Học máy (Machine Learning) – Thầy Chiểu, Khoa Công nghệ thông tin, Trường Đại học Thủy Lợi.",
        "[2] Vũ Hữu Tiệp (2020), Machine Learning cơ bản, Nhà xuất bản Khoa học và Kỹ thuật.",
        "[3] Quinlan, J. R. (1986), Induction of Decision Trees, Machine Learning, 1(1), 81–106.",
        "[4] Breiman, L., Friedman, J., Stone, C. J., & Olshen, R. A. (1984), Classification and Regression Trees (CART), CRC Press.",
        "[5] Breiman, L. (2001), Random Forests, Machine Learning, 45(1), 5–32.",
        "[6] OpenML Repository – Credit Risk Dataset (Data ID: 43454), truy cập tự động qua thư viện Scikit-learn.",
    ]
    for ref in refs:
        add_para(doc, ref, align=WD_ALIGN_PARAGRAPH.LEFT)

    doc.save(OUTPUT_DOCX)
    print(f"[OK] Đã xuất thành công Báo cáo Word đầy đủ 4 Chương tại:\n     {OUTPUT_DOCX}")
    print(f"[OK] Dung lượng file Word: {os.path.getsize(OUTPUT_DOCX) / (1024 * 1024):.2f} MB")


if __name__ == "__main__":
    build_report()

