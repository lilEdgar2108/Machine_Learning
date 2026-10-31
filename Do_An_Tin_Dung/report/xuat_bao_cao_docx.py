"""
SCRIPT SINH BÁO CÁO ĐỒ ÁN MÔN HỌC MÁY (.DOCX) ĐÚNG 100% MẪU QUY ĐỊNH ĐẠI HỌC THỦY LỢI
Dựa trên:
1. File mẫu quy định `Huong dan trinh bay DATN (26-2-2020).docx` do Thầy Tạ Quang Chiểu cung cấp.
2. Thông tin nhóm sinh viên thực hiện từ `BTL_ML.docx`:
   - Vũ Đức Hải Nam (2451261036)
   - Nguyễn Ngọc Thành (2451261059)
   - Trần Thiên Vũ (2451261069)
   - Giảng viên hướng dẫn: Thầy Tạ Quang Chiểu
3. Toàn bộ bảng dữ liệu thực nghiệm từ `outputs/tables/` và 12 biểu đồ PNG từ `outputs/figures/`.
"""

import os
import sys
import pandas as pd
from docx import Document
from docx.shared import Cm, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

REPORT_DIR = os.path.dirname(os.path.abspath(__file__))
if REPORT_DIR not in sys.path:
    sys.path.insert(0, REPORT_DIR)

from tlu_docx_helpers import (
    prepare_tlu_template_document,
    add_section_break_paragraph,
    set_table_borders_none,
    set_cell_shading,
    set_cell_margins,
    format_run_tnr,
    add_tlu_table,
    add_tlu_figure,
    omml_r,
    omml_frac,
    omml_sub,
    omml_sup,
    omml_subsup,
    omml_sum,
    omml_delim,
    omml_hat,
    add_tlu_equation,
    add_toc_field,
    inject_ieee_sources_xml,
)

BASE_DIR = os.path.dirname(REPORT_DIR)
TABLES_DIR = os.path.join(BASE_DIR, "outputs", "tables")
FIGURES_DIR = os.path.join(BASE_DIR, "outputs", "figures")
SRC_DIR = os.path.join(BASE_DIR, "src")
TEMPLATE_DOCX = os.path.join(REPORT_DIR, "template", "Huong_dan_trinh_bay_DATN_TLU.docx")
LOGO_PATH = os.path.join(REPORT_DIR, "template", "logo_dhtl.png")
OUTPUT_DOCX = os.path.join(REPORT_DIR, "Bao_Cao_Do_An_Hoc_May_Nhom_HaiNam_Thanh_Vu.docx")


# =============================================================================
# CÁC HÀM TIỆN ÍCH SOẠN THẢO ĐOẠN VĂN CHUẨN STYLE ĐHTL
# =============================================================================

def add_content(doc, text):
    """Đoạn văn nội dung chuẩn Style `Content`: Times New Roman 13pt, 1.5 lines, before 10pt, justify."""
    p = doc.add_paragraph(style="Content")
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    r = p.add_run(text)
    format_run_tnr(r, size_pt=13, bold=False, italic=False)
    return p


def add_bullet_item(doc, lead_bold, body_text):
    """Gạch đầu dòng chuẩn Style `Bullet`: Times New Roman 13pt, 1.5 lines, before 0pt, after 0pt."""
    p = doc.add_paragraph(style="Bullet")
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    if lead_bold:
        r1 = p.add_run(lead_bold)
        format_run_tnr(r1, size_pt=13, bold=True, italic=False)
    r2 = p.add_run(body_text)
    format_run_tnr(r2, size_pt=13, bold=False, italic=False)
    return p


def add_chapter_heading(doc, title_uppercase, page_break_before=True):
    """
    Tiêu đề Chương chuẩn Style `Heading 1` (tự động sinh tiền tố `CHƯƠNG 1`, `CHƯƠNG 2`...
    nhờ Multilevel List numId=28 trong mẫu ĐHTL, Times New Roman 14pt, in đậm, IN HOA).
    """
    if page_break_before:
        doc.add_page_break()
    p = doc.add_paragraph(style="Heading 1")
    r = p.add_run(title_uppercase.upper())
    format_run_tnr(r, size_pt=14, bold=True, uppercase=True)
    return p


def add_h2(doc, title_text):
    """Tiểu mục cấp 1 chuẩn Style `Heading 2` (tự động đánh số 1.1, 1.2..., 13pt đậm)."""
    p = doc.add_paragraph(style="Heading 2")
    r = p.add_run(title_text)
    format_run_tnr(r, size_pt=13, bold=True, italic=False)
    return p


def add_h3(doc, title_text):
    """Tiểu mục cấp 2 chuẩn Style `Heading 3` (tự động đánh số 3.5.1..., 13pt đậm & nghiêng)."""
    p = doc.add_paragraph(style="Heading 3")
    r = p.add_run(title_text)
    format_run_tnr(r, size_pt=13, bold=True, italic=True)
    return p


def add_unnumbered_h1(doc, title_uppercase, page_break_before=True):
    """Đề mục lớn không đánh số chương chuẩn Style `Heading 1N` (Danh mục, Kết luận, TLTK, Phụ lục)."""
    if page_break_before:
        doc.add_page_break()
    p = doc.add_paragraph(style="Heading 1N")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(title_uppercase.upper())
    format_run_tnr(r, size_pt=14, bold=True, uppercase=True)
    return p


def add_ministry_header_table(doc):
    """Bảng tiêu đề 2 cột không viền trên Trang bìa chính & Trang bìa phụ (cỡ chữ 14, in hoa)."""
    tbl = doc.add_table(rows=1, cols=2)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders_none(tbl)
    widths = [Cm(7.5), Cm(8.5)]
    texts = ["BỘ GIÁO DỤC VÀ ĐÀO TẠO", "BỘ NÔNG NGHIỆP VÀ MÔI TRƯỜNG"]
    for j in range(2):
        cell = tbl.rows[0].cells[j]
        cell.width = widths[j]
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.line_spacing = 1.15
        r = p.add_run(texts[j])
        format_run_tnr(r, size_pt=13.5, bold=False, uppercase=True)
    return tbl


def add_centered_line(doc, text, size_pt=14, bold=False, italic=False, uppercase=False,
                      space_before_pt=0, space_after_pt=6, line_spacing=1.3):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(space_before_pt)
    p.paragraph_format.space_after = Pt(space_after_pt)
    p.paragraph_format.line_spacing = line_spacing
    r = p.add_run(text)
    format_run_tnr(r, size_pt=size_pt, bold=bold, italic=italic, uppercase=uppercase)
    return p


def add_code_block(doc, code_str):
    """Khung mã nguồn gọn sạch trong Phụ lục (cỡ chữ 10pt nằm trong giới hạn 10-13pt của Mục 2.3.2)."""
    tbl = doc.add_table(rows=1, cols=1)
    tbl.style = "Table Grid"
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = tbl.rows[0].cells[0]
    set_cell_shading(cell, "F7F7F7")
    set_cell_margins(cell, top=80, bottom=80, left=120, right=120)
    p = cell.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(0)
    r = p.add_run(code_str.strip())
    format_run_tnr(r, size_pt=10.0, bold=False, italic=False)


# =============================================================================
# HÀM CHÍNH XÂY DỰNG TOÀN BỘ BÁO CÁO ĐỒ ÁN MÔN HỌC
# =============================================================================

def build_report():
    # Nạp 100% số liệu thực nghiệm từ outputs/tables/
    df_quality = pd.read_csv(os.path.join(TABLES_DIR, "01_kiem_tra_chat_luong_du_lieu.csv"))
    df_num_desc = pd.read_csv(os.path.join(TABLES_DIR, "01_thong_ke_mo_ta_so.csv")).rename(columns={"Unnamed: 0": "Bien_So"})
    df_prep_sum = pd.read_csv(os.path.join(TABLES_DIR, "02_tong_ket_tien_xu_ly.csv"))
    df_prep_params = pd.read_csv(os.path.join(TABLES_DIR, "02_tham_so_imputation_va_scaler.csv"))
    df_id3_bins = pd.read_csv(os.path.join(TABLES_DIR, "02_nguong_roi_rac_hoa_id3.csv"))
    df_cv_depth = pd.read_csv(os.path.join(TABLES_DIR, "03_khao_sat_do_sau_cv.csv"))
    df_logreg_w = pd.read_csv(os.path.join(TABLES_DIR, "03_he_so_logistic_regression.csv"))
    df_id3_ig = pd.read_csv(os.path.join(TABLES_DIR, "03_ig_nut_goc_id3.csv"))
    df_feat_imp = pd.read_csv(os.path.join(TABLES_DIR, "03_feature_importance_cart_rf.csv"))
    df_model_cfg = pd.read_csv(os.path.join(TABLES_DIR, "03_cau_hinh_4_mo_hinh.csv"))
    df_comp_4 = pd.read_csv(os.path.join(TABLES_DIR, "04_bang_so_sanh_4_mo_hinh.csv"))
    df_cm_detail = pd.read_csv(os.path.join(TABLES_DIR, "04_chi_tiet_confusion_matrix_test.csv"))
    df_demo_pred = pd.read_csv(os.path.join(TABLES_DIR, "05_ket_qua_du_doan_ho_so_mau.csv"))

    with open(os.path.join(TABLES_DIR, "03_so_do_nhanh_dau_cay_id3.txt"), "r", encoding="utf-8") as f:
        id3_tree_txt = f.read()

    # Khởi tạo trực tiếp từ file mẫu chuẩn của Trường Đại học Thủy Lợi
    doc = Document(TEMPLATE_DOCX)
    sect0_pr, sect1_pr = prepare_tlu_template_document(doc)

    # =========================================================================
    # SECTION 1: TRANG BÌA CHÍNH + TRANG BÌA PHỤ + TỜ PHÂN CÔNG (KHÔNG ĐÁNH SỐ TRANG)
    # =========================================================================

    # --- 1A. TRANG BÌA CHÍNH ---
    add_ministry_header_table(doc)
    add_centered_line(doc, "TRƯỜNG ĐẠI HỌC THỦY LỢI", size_pt=14, bold=True, uppercase=True,
                      space_before_pt=4, space_after_pt=18)

    p_logo = doc.add_paragraph()
    p_logo.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_logo.paragraph_format.space_before = Pt(6)
    p_logo.paragraph_format.space_after = Pt(18)
    p_logo.add_run().add_picture(LOGO_PATH, width=Cm(4.2))

    add_centered_line(
        doc,
        "VŨ ĐỨC HẢI NAM – 2451261036\nNGUYỄN NGỌC THÀNH – 2451261059\nTRẦN THIÊN VŨ – 2451261069",
        size_pt=14, bold=False, uppercase=True, space_before_pt=12, space_after_pt=28, line_spacing=1.45
    )

    add_centered_line(
        doc,
        "DỰ ĐOÁN KHẢ NĂNG TRẢ NỢ CỦA KHÁCH HÀNG VAY VỐN",
        size_pt=16, bold=True, uppercase=True, space_before_pt=16, space_after_pt=28, line_spacing=1.35
    )

    add_centered_line(
        doc,
        "BÁO CÁO ĐỒ ÁN MÔN HỌC: HỌC MÁY",
        size_pt=14, bold=False, uppercase=True, space_before_pt=12, space_after_pt=96
    )

    add_centered_line(doc, "HÀ NỘI, NĂM 2026", size_pt=14, bold=False, uppercase=True,
                      space_before_pt=24, space_after_pt=0)

    # --- 1B. TRANG BÌA PHỤ ---
    doc.add_page_break()
    add_ministry_header_table(doc)
    add_centered_line(doc, "TRƯỜNG ĐẠI HỌC THỦY LỢI", size_pt=14, bold=True, uppercase=True,
                      space_before_pt=4, space_after_pt=28)

    add_centered_line(
        doc,
        "VŨ ĐỨC HẢI NAM – 2451261036\nNGUYỄN NGỌC THÀNH – 2451261059\nTRẦN THIÊN VŨ – 2451261069",
        size_pt=14, bold=False, uppercase=True, space_before_pt=18, space_after_pt=32, line_spacing=1.45
    )

    add_centered_line(
        doc,
        "DỰ ĐOÁN KHẢ NĂNG TRẢ NỢ CỦA KHÁCH HÀNG VAY VỐN",
        size_pt=16, bold=True, uppercase=True, space_before_pt=16, space_after_pt=36, line_spacing=1.35
    )

    # Bảng thông tin Ngành, Mã số cấp IV, Học phần & Giảng viên hướng dẫn (cỡ chữ 13 chuẩn mẫu)
    tbl_meta = doc.add_table(rows=4, cols=2)
    tbl_meta.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders_none(tbl_meta)
    meta_rows = [
        ("Ngành:", "Công nghệ thông tin"),
        ("Mã số:", "7480201"),
        ("Học phần:", "Học máy (Machine Learning)"),
        ("NGƯỜI HƯỚNG DẪN:", "Tạ Quang Chiểu"),
    ]
    for ri, (k_txt, v_txt) in enumerate(meta_rows):
        c0, c1 = tbl_meta.rows[ri].cells[0], tbl_meta.rows[ri].cells[1]
        c0.width, c1.width = Cm(5.5), Cm(8.5)
        for cell, txt, is_b in [(c0, k_txt, ri == 3), (c1, v_txt, ri == 3)]:
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_before = Pt(4)
            p.paragraph_format.space_after = Pt(4)
            p.paragraph_format.line_spacing = 1.3
            r = p.add_run(txt)
            format_run_tnr(r, size_pt=13, bold=is_b)

    add_centered_line(doc, "HÀ NỘI, NĂM 2026", size_pt=14, bold=False, uppercase=True,
                      space_before_pt=110, space_after_pt=0)

    # --- 1C. TỜ NHIỆM VỤ & PHÂN CÔNG THỰC HIỆN ĐỒ ÁN (KHÔNG ĐÁNH SỐ THEO MỤC 2.3.3) ---
    doc.add_page_break()
    add_centered_line(doc, "TRƯỜNG ĐẠI HỌC THỦY LỢI\nKHOA CÔNG NGHỆ THÔNG TIN", size_pt=13, bold=True,
                      uppercase=True, space_before_pt=0, space_after_pt=18, line_spacing=1.3)
    add_centered_line(doc, "TỜ NHIỆM VỤ VÀ PHÂN CÔNG THỰC HIỆN ĐỒ ÁN MÔN HỌC", size_pt=14, bold=True,
                      uppercase=True, space_before_pt=6, space_after_pt=14)

    add_content(doc, "1. Tên đề tài: Dự đoán khả năng trả nợ của khách hàng vay vốn (Credit Default Prediction).")
    add_content(doc, "2. Học phần: Học máy (Machine Learning) – Giảng viên hướng dẫn: Thầy Tạ Quang Chiểu.")
    add_content(doc, "3. Các thuật toán nghiên cứu và thực nghiệm bắt buộc: (1) Hồi quy Logistic (Logistic Regression); "
                     "(2) Cây quyết định ID3 tự cài đặt thuần Python; (3) Cây quyết định CART; (4) Rừng ngẫu nhiên (Random Forest).")
    add_content(doc, "4. Bảng phân công nhiệm vụ chi tiết giữa các thành viên trong nhóm:")

    tbl_team = doc.add_table(rows=4, cols=5)
    tbl_team.style = "Table Grid"
    tbl_team.alignment = WD_TABLE_ALIGNMENT.CENTER
    team_headers = ["TT", "Họ và tên sinh viên", "Mã sinh viên", "Nội dung công việc & Module phụ trách", "Chương báo cáo"]
    team_data = [
        ["1", "Vũ Đức Hải Nam\n(Trưởng nhóm GitHub)", "2451261036",
         "Tải dữ liệu tự động, phân tích khám phá dữ liệu (EDA), làm sạch dữ liệu, chia tập phân tầng chống rò rỉ dữ liệu, điền khuyết, mã hóa, chuẩn hóa và rời rạc hóa (`src/01`, `src/02`).",
         "Chương 1 &\nChương 2"],
        ["2", "Nguyễn Ngọc Thành", "2451261059",
         "Nghiên cứu cơ sở toán học, lập trình lớp `ID3DecisionTree` đa nhánh từ đầu, khảo sát 5-Fold CV chống quá khớp và huấn luyện 4 mô hình (`src/id3_tu_cai_dat.py`, `src/03`).",
         "Chương 3\n(Mục 3.1 – 3.4)"],
        ["3", "Trần Thiên Vũ", "2451261069",
         "Đánh giá so sánh 4 mô hình trên tập kiểm tra độc lập (Confusion Matrix, FP/FN, Precision, Recall, F1, ROC-AUC), xây dựng hàm `predict_loan_status` và tổng hợp báo cáo (`src/04`, `src/05`).",
         "Mục 3.5,\nChương 4 &\nKết luận"],
    ]
    for j, h_txt in enumerate(team_headers):
        cell = tbl_team.rows[0].cells[j]
        set_cell_shading(cell, "EAEAEA")
        set_cell_margins(cell, 70, 70, 70, 70)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(h_txt)
        format_run_tnr(r, size_pt=11, bold=True)

    for i, row_vals in enumerate(team_data):
        for j, val in enumerate(row_vals):
            cell = tbl_team.rows[i + 1].cells[j]
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            set_cell_margins(cell, 60, 60, 70, 70)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if j in (1, 3) else WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.line_spacing = 1.2
            r = p.add_run(val)
            format_run_tnr(r, size_pt=11, bold=(j == 1))

    # Kết thúc Section 1 (không đánh số trang) -> chuyển sang Section 2 đánh số La Mã i, ii, iii...
    add_section_break_paragraph(doc, sect0_pr)

    # =========================================================================
    # SECTION 2: PHẦN MỞ ĐẦU (ĐÁNH SỐ TRANG i, ii, iii... TỪ LỜI CAM ĐOAN)
    # =========================================================================

    # --- 2A. LỜI CAM ĐOAN (Trang i) ---
    add_unnumbered_h1(doc, "LỜI CAM ĐOAN", page_break_before=False)
    add_content(
        doc,
        "Nhóm tác giả xin cam đoan đây là Báo cáo đồ án môn học Học máy do chính nhóm tác giả thực hiện dưới sự "
        "hướng dẫn của Thầy Tạ Quang Chiểu. Toàn bộ mã nguồn chương trình, thuật toán cây quyết định ID3 tự cài đặt "
        "và các bảng số liệu, biểu đồ thực nghiệm trình bày trong báo cáo này là hoàn toàn trung thực, khớp 100% với "
        "kết quả chạy thực tế từ dữ liệu và không sao chép từ bất kỳ nguồn nào dưới bất kỳ hình thức nào. Việc tham "
        "khảo các tài liệu học thuật đã được thực hiện trích dẫn và ghi nguồn tài liệu tham khảo theo đúng tiêu chuẩn "
        "IEEE theo quy định của Trường Đại học Thủy Lợi."
    )

    p_date = doc.add_paragraph(style="Content")
    p_date.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    p_date.paragraph_format.space_before = Pt(18)
    r_dt = p_date.add_run("Hà Nội, tháng 10 năm 2026")
    format_run_tnr(r_dt, size_pt=13, italic=True)

    tbl_sign = doc.add_table(rows=2, cols=3)
    tbl_sign.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders_none(tbl_sign)
    authors_sign = ["Vũ Đức Hải Nam", "Nguyễn Ngọc Thành", "Trần Thiên Vũ"]
    for j, name_str in enumerate(authors_sign):
        c_top = tbl_sign.rows[0].cells[j]
        p0 = c_top.paragraphs[0]
        p0.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p0.paragraph_format.space_after = Pt(36)
        r0 = p0.add_run("Sinh viên thực hiện\n(Ký và ghi rõ họ tên)")
        format_run_tnr(r0, size_pt=12, bold=True)

        c_bot = tbl_sign.rows[1].cells[j]
        p1 = c_bot.paragraphs[0]
        p1.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r1 = p1.add_run(name_str)
        format_run_tnr(r1, size_pt=13, bold=True)

    # --- 2B. LỜI CÁM ƠN (Trang ii) ---
    add_unnumbered_h1(doc, "LỜI CÁM ƠN", page_break_before=True)
    add_content(
        doc,
        "Lời đầu tiên, nhóm sinh viên thực hiện gồm Vũ Đức Hải Nam, Nguyễn Ngọc Thành và Trần Thiên Vũ xin bày tỏ "
        "lòng biết ơn chân thành và sâu sắc nhất tới Thầy Tạ Quang Chiểu – giảng viên trực tiếp giảng dạy và hướng "
        "dẫn học phần Học máy tại Khoa Công nghệ thông tin, Trường Đại học Thủy Lợi. Thông qua hệ thống bài giảng "
        "lý thuyết chặt chẽ cùng các bài tập thực hành chuyên sâu từ Hồi quy tuyến tính, Cây quyết định ID3 và CART, "
        "Hồi quy Logistic đến các phương pháp Học kết hợp (Ensemble Learning), Thầy đã trang bị cho nhóm nền tảng "
        "tư duy toán học vững chắc và phương pháp thực nghiệm khoa học đúng bản chất."
    )
    add_content(
        doc,
        "Trong suốt quá trình thực hiện đề tài \"Dự đoán khả năng trả nợ của khách hàng vay vốn\", nhóm luôn nỗ lực "
        "tuân thủ nghiêm ngặt quy trình chuẩn mực của Khoa học dữ liệu: chống rò rỉ dữ liệu tuyệt đối giữa tập huấn "
        "luyện và tập kiểm tra, tự lập trình thuật toán ID3 đa nhánh từ đầu và đối chiếu khách quan bốn mô hình phân "
        "lớp trên dữ liệu thực tế. Mặc dù đã rất cố gắng, do kiến thức và kinh nghiệm thực tiễn của sinh viên năm thứ "
        "ba còn những hạn chế nhất định, báo cáo khó tránh khỏi thiếu sót. Nhóm rất mong nhận được sự chỉ dẫn và góp "
        "ý quý báu của Thầy để hoàn thiện hơn. Nhóm xin trân trọng cảm ơn!"
    )

    # --- 2C. MỤC LỤC ---
    add_unnumbered_h1(doc, "MỤC LỤC", page_break_before=True)
    toc_preview = [
        ("DANH MỤC CÁC HÌNH ẢNH", True, 0.0),
        ("DANH MỤC BẢNG BIỂU", True, 0.0),
        ("DANH MỤC CÁC TỪ VIẾT TẮT VÀ GIẢI THÍCH CÁC THUẬT NGỮ", True, 0.0),
        ("CHƯƠNG 1 MÔ TẢ BÀI TOÁN VÀ PHÂN TÍCH DỮ LIỆU THÔ", True, 0.0),
        ("1.1 Lý do chọn đề tài và ý nghĩa thực tế của bài toán chấm điểm tín dụng", False, 0.6),
        ("1.2 Nguồn dữ liệu và giải thích chi tiết các đặc trưng", False, 0.6),
        ("1.3 Quy ước biến mục tiêu và phân tích khám phá dữ liệu", False, 0.6),
        ("CHƯƠNG 2 TIỀN XỬ LÝ DỮ LIỆU", True, 0.0),
        ("2.1 Làm sạch bản ghi trùng lặp và lọc ngoại lai sai logic", False, 0.6),
        ("2.2 Chia tập huấn luyện và tập kiểm tra phân tầng chống rò rỉ dữ liệu", False, 0.6),
        ("2.3 Xử lý dữ liệu thiếu", False, 0.6),
        ("2.4 Mã hóa biến phân loại và chuẩn hóa thang đo theo đặc thù thuật toán", False, 0.6),
        ("2.5 Rời rạc hóa dữ liệu liên tục phục vụ thuật toán ID3", False, 0.6),
        ("CHƯƠNG 3 CƠ SỞ LÝ THUYẾT VÀ THỰC NGHIỆM", True, 0.0),
        ("3.1 Thuật toán Hồi quy Logistic", False, 0.6),
        ("3.2 Thuật toán Cây quyết định ID3 tự cài đặt", False, 0.6),
        ("3.3 Thuật toán Cây quyết định CART", False, 0.6),
        ("3.4 Thuật toán Rừng ngẫu nhiên", False, 0.6),
        ("3.5 Kết quả thực nghiệm, đánh giá và thảo luận chuyên sâu", False, 0.6),
        ("3.5.1 Ý nghĩa các độ đo đánh giá trong quản trị rủi ro tín dụng", False, 1.2),
        ("3.5.2 Bảng tổng hợp kết quả thực nghiệm và trực quan hóa", False, 1.2),
        ("3.5.3 Phân tích và thảo luận chuyên sâu từ số liệu thực nghiệm", False, 1.2),
        ("CHƯƠNG 4 XÂY DỰNG CHƯƠNG TRÌNH DỰ ĐOÁN", True, 0.0),
        ("4.1 Tổ chức cấu trúc mã nguồn chương trình", False, 0.6),
        ("4.2 Quy trình lưu trữ Pipeline và hàm dự đoán khách hàng mới", False, 0.6),
        ("4.3 Kết quả chạy thử minh họa trên các hồ sơ khách hàng mẫu", False, 0.6),
        ("KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN", True, 0.0),
        ("TÀI LIỆU THAM KHẢO", True, 0.0),
        ("PHỤ LỤC", True, 0.0),
    ]
    add_toc_field(doc, 'TOC \\o "1-3" \\h \\z \\t "Heading 1N,1"', toc_preview)

    # --- 2D. DANH MỤC CÁC HÌNH ẢNH ---
    add_unnumbered_h1(doc, "DANH MỤC CÁC HÌNH ẢNH", page_break_before=True)
    lof_preview = [
        ("Hình 1.1 Phân bố số lượng và tỷ lệ phần trăm hai lớp của biến mục tiêu loan_status", False, 0.0),
        ("Hình 1.2 Phân phối các biến số tiêu biểu và phát hiện điểm ngoại lai vô lý trên dữ liệu thô", False, 0.0),
        ("Hình 1.3 Tỷ lệ vỡ nợ theo hạng khoản vay, hình thức sở hữu nhà và lịch sử nợ xấu", False, 0.0),
        ("Hình 1.4 Ma trận tương quan tuyến tính Pearson giữa các biến số và trạng thái vỡ nợ", False, 0.0),
        ("Hình 2.1 Minh họa trực quan kết quả lọc ngoại lai, chuẩn hóa Z-score và rời rạc hóa biến số", False, 0.0),
        ("Hình 3.1 Đường cong ROC-AUC trên tập huấn luyện và kiểm định chéo 5-Fold theo độ sâu cây", False, 0.0),
        ("Hình 3.2 Xếp hạng độ lợi thông tin của 11 thuộc tính tại nút gốc cây quyết định ID3", False, 0.0),
        ("Hình 3.3 Trực quan hóa ba tầng đầu tiên của cây quyết định nhị phân CART", False, 0.0),
        ("Hình 3.4 Xếp hạng 10 đặc trưng quan trọng nhất trong mô hình Rừng ngẫu nhiên", False, 0.0),
        ("Hình 3.5 So sánh trực quan ma trận nhầm lẫn của 4 thuật toán trên tập kiểm tra độc lập", False, 0.0),
        ("Hình 3.6 Đường cong ROC và diện tích dưới đường cong AUC của 4 thuật toán trên tập kiểm tra", False, 0.0),
        ("Hình 3.7 Biểu đồ cột so sánh 5 độ đo đánh giá trên tập kiểm tra của 4 thuật toán", False, 0.0),
    ]
    add_toc_field(doc, 'TOC \\h \\z \\c "Hình"', lof_preview)

    # --- 2E. DANH MỤC BẢNG BIỂU ---
    add_unnumbered_h1(doc, "DANH MỤC BẢNG BIỂU", page_break_before=True)
    lot_preview = [
        ("Bảng 1.1 Mô tả chi tiết 12 thuộc tính và tình trạng dữ liệu khuyết trên tập dữ liệu thô", False, 0.0),
        ("Bảng 1.2 Thống kê mô tả 7 biến số liên tục trên tập dữ liệu thô", False, 0.0),
        ("Bảng 2.1 Tổng kết số lượng mẫu qua các bước làm sạch và chia phân tầng dữ liệu", False, 0.0),
        ("Bảng 2.2 Các tham số trung vị điền khuyết và chuẩn hóa Z-score học từ tập huấn luyện", False, 0.0),
        ("Bảng 2.3 Bảng quy đổi ngưỡng rời rạc hóa 4 khoảng tứ phân vị phục vụ thuật toán ID3", False, 0.0),
        ("Bảng 3.1 Tám đặc trưng có độ lớn trọng số ảnh hưởng mạnh nhất trong Hồi quy Logistic", False, 0.0),
        ("Bảng 3.2 Kết quả khảo sát kiểm định chéo 5-Fold chọn độ sâu tối ưu cho ID3 và CART", False, 0.0),
        ("Bảng 3.3 Xếp hạng độ lợi thông tin của 11 thuộc tính tại nút gốc cây quyết định ID3", False, 0.0),
        ("Bảng 3.4 Đối chiếu mức độ quan trọng của 10 đặc trưng hàng đầu giữa CART và Random Forest", False, 0.0),
        ("Bảng 3.5 Bảng so sánh tổng hợp kết quả thực nghiệm 4 thuật toán trên tập huấn luyện và kiểm tra", False, 0.0),
        ("Bảng 3.6 Chi tiết ma trận nhầm lẫn và tỷ lệ sai lầm tài chính trên tập kiểm tra độc lập", False, 0.0),
        ("Bảng 4.1 Tổng hợp cấu hình lưu trữ và thời gian huấn luyện thực tế của 4 mô hình", False, 0.0),
        ("Bảng 4.2 Kết quả chạy thử hàm dự đoán khách hàng mới với cả 4 mô hình trên 3 kịch bản", False, 0.0),
    ]
    add_toc_field(doc, 'TOC \\h \\z \\c "Bảng"', lot_preview)

    # --- 2F. DANH MỤC CÁC TỪ VIẾT TẮT VÀ GIẢI THÍCH CÁC THUẬT NGỮ (Xếp chuẩn A -> Z theo Mục 2.3.5) ---
    add_unnumbered_h1(doc, "DANH MỤC CÁC TỪ VIẾT TẮT VÀ GIẢI THÍCH CÁC THUẬT NGỮ", page_break_before=True)
    abbr_list = [
        ("AUC", "Area Under the Curve – Diện tích dưới đường cong đặc trưng hoạt động"),
        ("BCE", "Binary Cross-Entropy – Hàm mất mát chéo nhị phân"),
        ("CART", "Classification and Regression Trees – Cây phân loại và hồi quy nhị phân"),
        ("CV", "Cross-Validation – Kiểm định chéo K phần"),
        ("EDA", "Exploratory Data Analysis – Phân tích khám phá dữ liệu"),
        ("FN", "False Negative – Âm tính giả (bỏ lọt khách hàng vỡ nợ thành trả nợ tốt)"),
        ("FP", "False Positive – Dương tính giả (cảnh báo nhầm khách hàng trả tốt thành vỡ nợ)"),
        ("FPR", "False Positive Rate – Tỷ lệ dương tính giả"),
        ("ID3", "Iterative Dichotomiser 3 – Thuật toán cây quyết định phân nhánh đa hướng ID3"),
        ("IEEE", "Institute of Electrical and Electronics Engineers – Viện Kỹ sư Điện và Điện tử"),
        ("IG", "Information Gain – Độ lợi thông tin"),
        ("MLE", "Maximum Likelihood Estimation – Ước lượng hợp lý cực đại"),
        ("MSE", "Mean Squared Error – Trung bình bình phương sai số"),
        ("OOB", "Out-of-Bag – Tập mẫu ngoài túi trong kỹ thuật lấy mẫu có hoàn lại Bootstrap"),
        ("RF", "Random Forest – Thuật toán Rừng ngẫu nhiên"),
        ("ROC", "Receiver Operating Characteristic – Đường cong đặc trưng hoạt động của bộ phân lớp"),
        ("TN", "True Negative – Âm tính thật (duyệt đúng khách hàng trả nợ tốt)"),
        ("TP", "True Positive – Dương tính thật (phát hiện đúng khách hàng vỡ nợ)"),
        ("TPR", "True Positive Rate – Tỷ lệ dương tính thật (tương đương độ nhạy Recall)"),
    ]
    tbl_abbr = doc.add_table(rows=len(abbr_list) + 1, cols=2)
    tbl_abbr.style = "Table Grid"
    tbl_abbr.alignment = WD_TABLE_ALIGNMENT.CENTER
    for j, h_t in enumerate(["Từ viết tắt", "Thuật ngữ tiếng Anh và giải thích nghĩa tiếng Việt"]):
        c = tbl_abbr.rows[0].cells[j]
        set_cell_shading(c, "EAEAEA")
        set_cell_margins(c, 60, 60, 90, 90)
        p = c.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(h_t)
        format_run_tnr(r, size_pt=12, bold=True)
    for idx_a, (short_w, desc_w) in enumerate(abbr_list):
        c0, c1 = tbl_abbr.rows[idx_a + 1].cells[0], tbl_abbr.rows[idx_a + 1].cells[1]
        c0.width, c1.width = Cm(3.2), Cm(12.8)
        set_cell_margins(c0, 50, 50, 90, 90)
        set_cell_margins(c1, 50, 50, 90, 90)
        p0 = c0.paragraphs[0]
        p0.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r0 = p0.add_run(short_w)
        format_run_tnr(r0, size_pt=12, bold=True)
        p1 = c1.paragraphs[0]
        p1.alignment = WD_ALIGN_PARAGRAPH.LEFT
        r1 = p1.add_run(desc_w)
        format_run_tnr(r1, size_pt=12, bold=False)

    # Kết thúc Section 2 (số La Mã i, ii, iii...) -> chuyển sang Section 3 đánh số Ả Rập 1, 2, 3...
    add_section_break_paragraph(doc, sect1_pr)

    # =========================================================================
    # SECTION 3: PHẦN NỘI DUNG CHÍNH (ĐÁNH SỐ TRANG 1, 2, 3... LIÊN TỤC ĐẾN HẾT)
    # =========================================================================

    # -------------------------------------------------------------------------
    # CHƯƠNG 1: MÔ TẢ BÀI TOÁN VÀ PHÂN TÍCH DỮ LIỆU THÔ
    # -------------------------------------------------------------------------
    add_chapter_heading(doc, "MÔ TẢ BÀI TOÁN VÀ PHÂN TÍCH DỮ LIỆU THÔ", page_break_before=False)

    add_h2(doc, "Lý do chọn đề tài và ý nghĩa thực tế của bài toán chấm điểm tín dụng")
    add_content(
        doc,
        "Trong hoạt động của các ngân hàng thương mại và tổ chức tài chính tiêu dùng, nghiệp vụ cấp tín dụng luôn "
        "là mảng mang lại nguồn thu nhập lãi thuần chủ lực nhưng đồng thời tiềm ẩn rủi ro mất vốn lớn nhất. Khi một "
        "khách hàng nộp hồ sơ đề nghị vay vốn, bộ phận thẩm định phải đưa ra quyết định cốt lõi: khách hàng có khả "
        "năng hoàn trả đầy đủ gốc và lãi đúng kỳ hạn hay sẽ rơi vào tình trạng mất khả năng thanh toán (vỡ nợ). Nếu "
        "quy trình xét duyệt chỉ dựa trên kinh nghiệm cảm tính thủ công, tổ chức tín dụng dễ đối mặt với hai thái cực "
        "rủi ro: hoặc giải ngân nhầm cho hồ sơ rủi ro cao dẫn đến mất trắng vốn gốc, hoặc thẩm định quá khắt khe làm "
        "từ chối nhầm khách hàng uy tín và đánh mất cơ hội kinh doanh."
    )
    add_content(
        doc,
        "Vì vậy, việc ứng dụng các thuật toán Học máy có giám sát (Supervised Machine Learning) để xây dựng mô hình "
        "phân lớp nhị phân dự đoán khả năng trả nợ dựa trên dữ liệu lịch sử là bài toán thiết thực, gắn liền trực tiếp "
        "với nội dung cốt lõi của học phần Học máy [1], [2]. Đề tài thực hiện nghiên cứu, cài đặt và đối chiếu toàn "
        "diện bốn thuật toán tiêu biểu đại diện cho các trường phái mô hình hóa khác nhau: Hồi quy Logistic tuyến tính "
        "xác suất, Cây quyết định đa nhánh ID3 tự cài đặt, Cây quyết định nhị phân CART và mô hình học kết hợp Rừng "
        "ngẫu nhiên (Random Forest)."
    )

    add_h2(doc, "Nguồn dữ liệu và giải thích chi tiết các đặc trưng")
    add_content(
        doc,
        "Để đảm bảo tính chuẩn mực học thuật và khả năng tái lập kết quả thực nghiệm, nhóm sử dụng bộ dữ liệu chuẩn "
        "quốc tế \"Credit Risk Dataset\" được lưu trữ công khai trên kho dữ liệu học thuật OpenML với mã định danh "
        "Data ID 43454 [3]. Dữ liệu được tải tự động bằng ngôn ngữ Python trong tệp mã nguồn "
        "01_tai_va_kham_pha_du_lieu.py. Bộ dữ liệu gốc bao gồm 32.581 bản ghi hồ sơ vay vốn và 12 cột thuộc tính, "
        "trong đó có 11 đặc trưng đầu vào (gồm 7 biến số liên tục và 4 biến phân loại) cùng 1 biến mục tiêu nhị phân "
        "loan_status. Toàn bộ 11 đặc trưng đầu vào đều là các thông tin thu thập tại thời điểm xét duyệt hồ sơ trước "
        "khi giải ngân nên đảm bảo không chứa biến rò rỉ từ tương lai. Chi tiết ý nghĩa nghiệp vụ và tỷ lệ khuyết thiếu "
        "của từng thuộc tính được tổng hợp tại Bảng 1.1."
    )
    add_tlu_table(
        doc, 1, 1,
        "Mô tả chi tiết 12 thuộc tính và tình trạng dữ liệu khuyết trên tập dữ liệu thô",
        df_quality,
        col_headers=["Tên cột", "Loại biến", "Số dòng thiếu", "Tỷ lệ thiếu (%)", "Số giá trị riêng", "Ý nghĩa nghiệp vụ thực tế"],
        font_size_pt=10.0,
    )

    add_h2(doc, "Quy ước biến mục tiêu và phân tích khám phá dữ liệu")
    add_content(
        doc,
        "Trước khi tiến hành thống kê khám phá dữ liệu (Exploratory Data Analysis – EDA), nhóm kiểm tra và quy ước "
        "thống nhất ý nghĩa hai lớp của biến mục tiêu loan_status xuyên suốt toàn bộ quy trình huấn luyện và đánh giá:"
    )
    add_bullet_item(
        doc, "Nhãn 0 (Lớp âm tính – Trả nợ tốt): ",
        "Khách hàng thực hiện đầy đủ nghĩa vụ trả gốc và lãi đúng hạn, gồm 25.473 bản ghi (chiếm 78,18% tổng dữ liệu)."
    )
    add_bullet_item(
        doc, "Nhãn 1 (Lớp dương tính – Vỡ nợ): ",
        "Khách hàng vi phạm nghĩa vụ thanh toán và phát sinh nợ xấu cần phát hiện, gồm 7.108 bản ghi (chiếm 21,82% tổng dữ liệu)."
    )
    add_content(
        doc,
        "Như thể hiện trực quan tại Hình 1.1, tỷ lệ chênh lệch giữa lớp trả nợ tốt và lớp vỡ nợ là 3,58 : 1 "
        "(78,18% so với 21,82%). Đây là mức mất cân bằng lớp rất điển hình của dữ liệu tài chính ngân hàng thực tế: "
        "đủ lớn để gây ra nghịch lý độ chính xác (Accuracy Paradox) nếu chỉ đánh giá mô hình bằng tỷ lệ đoán đúng "
        "tổng thể, đòi hỏi phải kết hợp phân tích sâu ma trận nhầm lẫn và các độ đo Precision, Recall, F1-Score, "
        "ROC-AUC ở Chương 3."
    )
    add_tlu_figure(
        doc, os.path.join(FIGURES_DIR, "01_phan_bo_nhan_loan_status.png"),
        1, 1, "Phân bố số lượng và tỷ lệ phần trăm hai lớp của biến mục tiêu loan_status", width_cm=14.5
    )

    add_content(
        doc,
        "Tiếp theo, nhóm thống kê các đại lượng mô tả trung tâm, độ phân tán, các mốc tứ phân vị và hệ số bất đối "
        "xứng (Skewness) của 7 biến số liên tục trên tập dữ liệu thô tại Bảng 1.2, đồng thời trực quan hóa phân phối, "
        "tỷ lệ vỡ nợ theo từng nhóm phân loại và ma trận tương quan tại Hình 1.2, Hình 1.3 và Hình 1.4."
    )
    add_tlu_table(
        doc, 1, 2,
        "Thống kê mô tả 7 biến số liên tục trên tập dữ liệu thô",
        df_num_desc,
        col_headers=["Biến số", "Count", "Mean", "Std", "Min", "Q1 (25%)", "Median", "Q3 (75%)", "Max", "Skewness"],
        font_size_pt=10.0,
    )

    add_content(
        doc,
        "Từ số liệu định lượng tại Bảng 1.1, Bảng 1.2 cùng các biểu đồ Hình 1.2, Hình 1.3 và Hình 1.4, nhóm rút ra "
        "bốn phát hiện cốt lõi làm căn cứ khoa học trực tiếp cho thiết kế quy trình tiền xử lý ở Chương 2:"
    )
    add_bullet_item(
        doc, "Phát hiện 1 – Tồn tại bản ghi trùng lặp và ngoại lai sai logic thực tế: ",
        "Kiểm tra toàn bộ 32.581 dòng phát hiện 165 bản ghi bị trùng lặp hoàn toàn trên cả 12 cột. Bên cạnh đó, "
        "biểu đồ hộp tại Hình 1.2 chỉ rõ 5 hồ sơ có tuổi khách hàng lên tới 123 và 144 tuổi (person_age > 100) và "
        "2 hồ sơ ghi nhận thâm niên làm việc 123 năm (person_emp_length = 123) trong khi người vay mới ngoài 20 tuổi. "
        "Đây là các lỗi nhập liệu bất khả thi về mặt sinh học và lao động cần phải loại bỏ."
    )
    add_bullet_item(
        doc, "Phát hiện 2 – Dữ liệu khuyết và phân phối lệch phải rất mạnh: ",
        "Thuộc tính person_emp_length khuyết 895 dòng (2,75%) và loan_int_rate khuyết 3.116 dòng (9,56%). Đặc biệt, "
        "thu nhập khách hàng person_income có độ lệch phải cực lớn (Skewness = 32,87; giá trị cực đại 6.000.000 USD "
        "kéo trung bình cộng 66.074,85 USD lệch xa trung vị 55.000 USD), chứng tỏ phải dùng trung vị (Median) thay vì "
        "trung bình cộng (Mean) khi điền dữ liệu thiếu."
    )
    add_bullet_item(
        doc, "Phát hiện 3 – Quy luật rủi ro đơn điệu trên biến phân loại có thứ tự: ",
        "Hình 1.3 cho thấy tỷ lệ vỡ nợ tăng đơn điệu rất rõ rệt theo hạng tín nhiệm khoản vay loan_grade từ hạng A "
        "(10,0%), B (16,3%), C (20,7%) vọt lên D (59,0%), E (64,4%), F (70,5%) và G (98,4%). Tương tự, nhóm khách hàng "
        "ở nhà thuê (RENT) có tỷ lệ vỡ nợ 31,6%, cao gấp hơn 4 lần nhóm tự sở hữu nhà (OWN: 7,5%)."
    )
    add_bullet_item(
        doc, "Phát hiện 4 – Cấu trúc tương quan tuyến tính giữa các biến số: ",
        "Ma trận tương quan Pearson tại Hình 1.4 xác nhận tỷ lệ tiền vay trên thu nhập loan_percent_income (r = +0,38) "
        "và lãi suất vay loan_int_rate (r = +0,34) có tương quan dương mạnh nhất với nguy cơ vỡ nợ. Đồng thời, tuổi "
        "khách hàng person_age và độ dài lịch sử tín dụng cb_person_cred_hist_length có tương quan đa cộng tuyến khá "
        "cao (r = +0,86), đặt ra yêu cầu sử dụng thành phần điều chuẩn L2 cho mô hình Hồi quy Logistic."
    )

    add_tlu_figure(
        doc, os.path.join(FIGURES_DIR, "01_phan_bo_bien_so_va_ngoai_lai.png"),
        1, 2, "Phân phối các biến số tiêu biểu và phát hiện điểm ngoại lai vô lý trên dữ liệu thô", width_cm=15.4
    )
    add_tlu_figure(
        doc, os.path.join(FIGURES_DIR, "01_ty_le_vo_no_theo_nhom.png"),
        1, 3, "Tỷ lệ vỡ nợ theo hạng khoản vay, hình thức sở hữu nhà và lịch sử nợ xấu", width_cm=15.4
    )
    add_tlu_figure(
        doc, os.path.join(FIGURES_DIR, "01_ma_tran_tuong_quan.png"),
        1, 4, "Ma trận tương quan tuyến tính Pearson giữa các biến số và trạng thái vỡ nợ", width_cm=14.0
    )

    # -------------------------------------------------------------------------
    # CHƯƠNG 2: TIỀN XỬ LÝ DỮ LIỆU
    # -------------------------------------------------------------------------
    add_chapter_heading(doc, "TIỀN XỬ LÝ DỮ LIỆU", page_break_before=True)

    add_h2(doc, "Làm sạch bản ghi trùng lặp và lọc ngoại lai sai logic")
    add_content(
        doc,
        "Toàn bộ quy trình tiền xử lý dữ liệu được cài đặt tường minh trong tệp 02_tien_xu_ly_du_lieu.py. Ở bước đầu "
        "tiên, nhóm thực hiện loại bỏ 165 bản ghi trùng lặp hoàn toàn nhằm ngăn chặn hiện tượng cùng một hồ sơ khách "
        "hàng xuất hiện đồng thời ở cả tập huấn luyện lẫn tập kiểm tra gây đánh giá cao giả tạo. Tiếp theo, nhóm lọc "
        "bỏ 7 bản ghi mang giá trị ngoại lai sai logic thực tế đã phát hiện ở Chương 1 bằng các ràng buộc miền giá trị "
        "độc lập: giữ lại các hồ sơ có tuổi từ 18 đến 100 năm, thâm niên làm việc từ 0 đến 60 năm và thỏa mãn điều kiện "
        "thâm niên không vượt quá số tuổi trừ 14 (các bản ghi đang khuyết giá trị NaN vẫn được giữ nguyên để đưa sang "
        "bước điền khuyết). Sau bước làm sạch, tập dữ liệu còn lại 32.409 bản ghi hợp lệ. Do quy tắc lọc chỉ kiểm tra "
        "logic nội tại trên từng bản ghi riêng lẻ mà không tính toán bất kỳ tham số thống kê toàn cục nào nên hoàn toàn "
        "không gây rò rỉ dữ liệu."
    )

    add_h2(doc, "Chia tập huấn luyện và tập kiểm tra phân tầng chống rò rỉ dữ liệu")
    add_content(
        doc,
        "Một trong những sai lầm học thuật phổ biến nhất trong các đồ án Học máy là thực hiện điền dữ liệu thiếu, "
        "chuẩn hóa thang đo hoặc chia khoảng rời rạc trên toàn bộ bộ dữ liệu rồi mới tách tập huấn luyện và tập kiểm "
        "tra, dẫn đến thông tin thống kê của tập kiểm tra bị rò rỉ sang quá trình học (Data Leakage) [1], [4]. Để "
        "triệt tiêu hoàn toàn nguy cơ này, ngay sau bước lọc lỗi logic, nhóm lập tức phân chia 32.409 bản ghi sạch "
        "thành hai tập độc lập theo tỷ lệ 80% tập huấn luyện (Train: 25.927 mẫu) và 20% tập kiểm tra (Test: 6.482 mẫu) "
        "sử dụng kỹ thuật lấy mẫu phân tầng (Stratified Sampling) với hạt giống ngẫu nhiên cố định random_state = 42. "
        "Như tổng hợp tại Bảng 2.1, kỹ thuật phân tầng giúp tỷ lệ lớp vỡ nợ được bảo toàn đồng nhất giữa tập huấn luyện "
        "(21,87% – 5.670 mẫu) và tập kiểm tra (21,88% – 1.418 mẫu)."
    )
    add_tlu_table(
        doc, 2, 1,
        "Tổng kết số lượng mẫu qua các bước làm sạch và chia phân tầng dữ liệu",
        df_prep_sum,
        col_headers=["Bước xử lý", "Số dòng dữ liệu", "Ghi chú chi tiết"],
        font_size_pt=10.5,
    )

    add_h2(doc, "Xử lý dữ liệu thiếu")
    add_content(
        doc,
        "Đối với hai biến số liên tục bị khuyết giá trị là person_emp_length và loan_int_rate, nhóm không xóa bỏ dòng "
        "(vì sẽ làm mất gần 10% lượng thông tin quý giá của bộ dữ liệu) mà áp dụng phương pháp điền khuyết bằng giá trị "
        "trung vị (Median Imputation). Do phân phối của các biến tài chính bị lệch phải mạnh (như đã chứng minh qua hệ "
        "số Skewness tại Bảng 1.2), giá trị trung vị nằm ở vị trí phân vị 50% không bị ảnh hưởng bởi các giá trị cực "
        "đoan nên phản ánh khách hàng điển hình chính xác hơn trung bình cộng. Đối với 4 biến phân loại, nhóm cũng cấu "
        "hình bộ điền khuyết theo giá trị xuất hiện nhiều nhất (Mode) học từ tập huấn luyện để dự phòng trường hợp hồ "
        "sơ khách hàng mới ở Chương 4 bị bỏ trống thông tin. Toàn bộ các tham số điền khuyết và chuẩn hóa chỉ được ước "
        "lượng duy nhất trên tập huấn luyện (25.927 mẫu) và trình bày chi tiết tại Bảng 2.2."
    )
    add_tlu_table(
        doc, 2, 2,
        "Các tham số trung vị điền khuyết và chuẩn hóa Z-score học từ tập huấn luyện",
        df_prep_params,
        col_headers=["Biến số liên tục", "Trung vị điền khuyết (Train)", "Trung bình chuẩn hóa (Train)", "Độ lệch chuẩn (Train)"],
        font_size_pt=10.5,
    )

    add_h2(doc, "Mã hóa biến phân loại và chuẩn hóa thang đo theo đặc thù thuật toán")
    add_content(
        doc,
        "Để tôn trọng đúng bản chất toán học của từng thuật toán, nhóm thiết kế bước mã hóa và biến đổi thang đo "
        "chuyên biệt như sau:"
    )
    add_bullet_item(
        doc, "Mã hóa biến phân loại có thứ tự và không thứ tự: ",
        "Thuộc tính hạng khoản vay loan_grade mang thứ tự rủi ro tự nhiên tăng dần từ A đến G nên được mã hóa thứ tự "
        "(Ordinal Encoding) thành các số nguyên từ 0 đến 6; thuộc tính nhị phân cb_person_default_on_file được mã hóa "
        "N = 0, Y = 1. Ngược lại, hai thuộc tính định danh không mang quan hệ lớn nhỏ là hình thức sở hữu nhà "
        "person_home_ownership (4 nhóm) và mục đích vay loan_intent (6 nhóm) được mã hóa nhị phân One-Hot Encoding "
        "thành các cột 0/1 độc lập nhằm tránh áp đặt khoảng cách số học giả tạo cho mô hình tuyến tính. Sau mã hóa, "
        "không gian đặc trưng gồm tổng cộng 19 biến."
    )
    add_bullet_item(
        doc, "Chuẩn hóa thang đo cho Hồi quy Logistic và giữ đơn vị gốc cho CART, Random Forest: ",
        "Do thu nhập hàng năm có độ lớn hàng chục nghìn USD trong khi tỷ lệ vay trên thu nhập chỉ nằm trong đoạn "
        "[0; 0,83], nếu đưa trực tiếp vào Hồi quy Logistic, thuật toán tối ưu hội tụ chậm và thành phần điều chuẩn L2 "
        "sẽ phạt không công bằng giữa các trọng số. Vì vậy, nhánh dữ liệu dành cho Hồi quy Logistic được chuẩn hóa "
        "Z-score theo công thức (2-1). Ngược lại, cây quyết định nhị phân CART và Rừng ngẫu nhiên chỉ xét thứ tự sắp "
        "xếp để tìm điểm cắt nên hoàn toàn bất biến với phép biến đổi tuyến tính đồng biến; việc giữ nguyên đơn vị gốc "
        "(USD, năm, %) giúp các nút chia trên cây trực quan, dễ đọc bằng ngôn ngữ nghiệp vụ ngân hàng."
    )

    # Phương trình (2-1): Chuẩn hóa Z-score bằng Word Equation <m:oMath>
    eq_2_1 = (
        omml_subsup(omml_r("x"), omml_r("ij"), omml_r("'", italic=False))
        + omml_r(" = ", italic=False)
        + omml_frac(
            omml_sub(omml_r("x"), omml_r("ij")) + omml_r(" − ", italic=False) + omml_sub(omml_r("μ"), omml_r("j")),
            omml_sub(omml_r("σ"), omml_r("j"))
        )
    )
    add_tlu_equation(doc, eq_2_1, "2-1")
    add_content(
        doc,
        "Trong đó: x_ij là giá trị gốc của đặc trưng thứ j ở mẫu thứ i; x'_ij là giá trị không thứ nguyên sau chuẩn "
        "hóa; μ_j và σ_j lần lượt là giá trị trung bình cộng và độ lệch chuẩn của đặc trưng thứ j được tính duy nhất "
        "trên tập dữ liệu huấn luyện (Bảng 2.2)."
    )

    add_h2(doc, "Rời rạc hóa dữ liệu liên tục phục vụ thuật toán ID3")
    add_content(
        doc,
        "Theo đúng lý thuyết Chương 5 môn Học máy [1], thuật toán ID3 nguyên bản phân nhánh đa hướng dựa trên các "
        "giá trị rời rạc của thuộc tính. Nếu đưa trực tiếp các cột số thực liên tục (như thu nhập có hơn 4.000 giá trị "
        "khác nhau) vào ID3, cây sẽ tách thành hàng nghìn nhánh con chỉ chứa 1–2 mẫu làm độ lợi thông tin đạt cực đại "
        "giả tạo nhưng mất hoàn toàn khả năng tổng quát hóa. Do đó, nhóm thực hiện rời rạc hóa (Discretization) 7 biến "
        "số liên tục thành 4 mức phân loại rõ nghĩa (1_Thap, 2_Trung_binh, 3_Kha_cao, 4_Cao) dựa trên các mốc tứ phân "
        "vị Q1 (25%), Q2 (50%), Q3 (75%) học từ tập huấn luyện, với hai biên ngoài cùng mở rộng ra (-∞; +∞) để xử lý "
        "an toàn mọi hồ sơ mới. Các ngưỡng cắt thực tế được trình bày tại Bảng 2.3 và minh họa trực quan tại Hình 2.1."
    )
    df_bins_display = df_id3_bins[["Bien_So", "Khoang_1_Thap", "Khoang_2_Trung_binh", "Khoang_3_Kha_cao", "Khoang_4_Cao"]]
    add_tlu_table(
        doc, 2, 3,
        "Bảng quy đổi ngưỡng rời rạc hóa 4 khoảng tứ phân vị phục vụ thuật toán ID3",
        df_bins_display,
        col_headers=["Biến số", "1_Thap (<= Q1)", "2_Trung_binh (Q1 - Q2]", "3_Kha_cao (Q2 - Q3]", "4_Cao (> Q3)"],
        font_size_pt=10.0,
    )
    add_tlu_figure(
        doc, os.path.join(FIGURES_DIR, "02_minh_hoa_tien_xu_ly.png"),
        2, 1, "Minh họa trực quan kết quả lọc ngoại lai, chuẩn hóa Z-score và rời rạc hóa biến số", width_cm=15.4
    )

    # -------------------------------------------------------------------------
    # CHƯƠNG 3: CƠ SỞ LÝ THUYẾT VÀ THỰC NGHIỆM
    # -------------------------------------------------------------------------
    add_chapter_heading(doc, "CƠ SỞ LÝ THUYẾT VÀ THỰC NGHIỆM", page_break_before=True)

    # --- 3.1 Logistic Regression ---
    add_h2(doc, "Thuật toán Hồi quy Logistic")
    add_content(
        doc,
        "Hồi quy Logistic (Logistic Regression) là mô hình phân lớp tuyến tính xác suất kinh điển do Cox đề xuất [5] "
        "và được giảng dạy tại Chương 6 học phần Học máy [1]. Với vector đặc trưng đầu vào đã chuẩn hóa x ∈ R^d, mô "
        "hình tính tổng trọng số tuyến tính z = w^T x + b rồi ánh xạ qua hàm kích hoạt Sigmoid để ước lượng xác suất "
        "hậu nghiệm khách hàng thuộc lớp vỡ nợ (y = 1) theo phương trình (3-1):"
    )
    eq_3_1 = (
        omml_r("P", italic=True)
        + omml_delim(omml_r("y") + omml_r(" = 1 | ", italic=False) + omml_r("x", bold=True))
        + omml_r(" = ", italic=False)
        + omml_r("σ") + omml_delim(omml_r("z"))
        + omml_r(" = ", italic=False)
        + omml_frac(
            omml_r("1", italic=False),
            omml_r("1 + ", italic=False) + omml_sup(
                omml_r("e"),
                omml_r("−", italic=False) + omml_delim(omml_sup(omml_r("w", bold=True), omml_r("T", italic=False)) + omml_r("x", bold=True) + omml_r(" + ", italic=False) + omml_r("b"))
            )
        )
    )
    add_tlu_equation(doc, eq_3_1, "3-1")
    add_content(
        doc,
        "Trong đó: w ∈ R^19 là vector trọng số đặc trưng; b là hệ số chặn (bias); z là điểm số tuyến tính (logit); "
        "σ(z) ∈ (0; 1) là xác suất dự đoán khách hàng vỡ nợ. Tại ngưỡng quyết định tiêu chuẩn 0,5, nếu P(y = 1 | x) ≥ "
        "0,5 (tương đương z ≥ 0) mô hình gán nhãn 1 (vỡ nợ), ngược lại gán nhãn 0 (trả nợ tốt)."
    )
    add_content(
        doc,
        "Để tìm bộ tham số tối ưu (w, b) trên tập huấn luyện gồm N mẫu, thuật toán xuất phát từ nguyên lý ước lượng "
        "hợp lý cực đại (MLE) và cực tiểu hóa hàm mất mát chéo nhị phân (Binary Cross-Entropy) kết hợp thành phần điều "
        "chuẩn L2 theo phương trình (3-2):"
    )
    eq_3_2 = (
        omml_r("J") + omml_delim(omml_r("w", bold=True) + omml_r(", ", italic=False) + omml_r("b"))
        + omml_r(" = −", italic=False)
        + omml_frac(omml_r("1", italic=False), omml_r("N"))
        + omml_sum(
            omml_r("i") + omml_r("=1", italic=False),
            omml_r("N"),
            omml_delim(
                omml_sub(omml_r("y"), omml_r("i")) + omml_r(" ln", italic=False) + omml_delim(omml_sub(omml_hat(omml_r("y")), omml_r("i")))
                + omml_r(" + ", italic=False)
                + omml_delim(omml_r("1 − ", italic=False) + omml_sub(omml_r("y"), omml_r("i")))
                + omml_r(" ln", italic=False) + omml_delim(omml_r("1 − ", italic=False) + omml_sub(omml_hat(omml_r("y")), omml_r("i"))),
                beg="[", end="]"
            )
        )
        + omml_r(" + ", italic=False)
        + omml_frac(omml_r("λ"), omml_r("2", italic=False))
        + omml_subsup(omml_delim(omml_r("w", bold=True), beg="‖", end="‖"), omml_r("2", italic=False), omml_r("2", italic=False))
    )
    add_tlu_equation(doc, eq_3_2, "3-2")
    add_content(
        doc,
        "Trong đó: N = 25.927 là số lượng mẫu tập huấn luyện; y_i ∈ {0; 1} là nhãn thực tế của khách hàng thứ i; "
        "ŷ_i = P(y_i = 1 | x_i) là xác suất dự đoán; λ > 0 là hệ số điều chuẩn phạt độ lớn chuẩn Euclid của trọng số "
        "(trong thư viện scikit-learn [4], tham số nghịch đảo C = 1/λ được đặt bằng 1,0). Việc sử dụng hàm mất mát "
        "Binary Cross-Entropy thay vì hàm trung bình bình phương sai số (MSE) đảm bảo hàm mục tiêu J(w, b) là hàm lồi "
        "nghiêm ngặt, giúp bộ giải tối ưu L-BFGS hội tụ chắc chắn về nghiệm cực tiểu toàn cục. Kết quả huấn luyện thu "
        "được hệ số chặn b = -1,5989; tám đặc trưng có độ lớn trọng số |w_j| tác động mạnh nhất tới khả năng vỡ nợ "
        "được trích xuất tại Bảng 3.1."
    )
    add_tlu_table(
        doc, 3, 1,
        "Tám đặc trưng có độ lớn trọng số ảnh hưởng mạnh nhất trong Hồi quy Logistic",
        df_logreg_w.head(8),
        col_headers=["Đặc trưng", "Trọng số w_j", "|w_j|", "Tỷ số chênh (e^w)", "Chiều tác động tới rủi ro"],
        font_size_pt=10.0,
    )

    # --- 3.2 Decision Tree ID3 ---
    add_h2(doc, "Thuật toán Cây quyết định ID3 tự cài đặt")
    add_content(
        doc,
        "Thuật toán ID3 (Iterative Dichotomiser 3) do Quinlan giới thiệu [6] xây dựng cây quyết định phân nhánh đa "
        "hướng theo chiến lược tham lam từ gốc xuống lá dựa trên lý thuyết thông tin Shannon [1]. Độ hỗn loạn thông tin "
        "(Entropy) của một tập mẫu S đối với bài toán phân lớp nhị phân được định nghĩa theo phương trình (3-3):"
    )
    eq_3_3 = (
        omml_r("H") + omml_delim(omml_r("S"))
        + omml_r(" = −", italic=False)
        + omml_sum(
            omml_r("c") + omml_r("∈{0,1}", italic=False),
            "",
            omml_sub(omml_r("p"), omml_r("c")) + omml_r(" ", italic=False)
            + omml_sub(omml_r("log", italic=False), omml_r("2", italic=False))
            + omml_delim(omml_sub(omml_r("p"), omml_r("c")))
        )
    )
    add_tlu_equation(doc, eq_3_3, "3-3")
    add_content(
        doc,
        "Trong đó: p_c là tỷ lệ số mẫu thuộc lớp c trong tập S (quy ước 0 · log_2(0) = 0; đơn vị đo là bit). Khi dùng "
        "một thuộc tính rời rạc A có K giá trị riêng biệt để chia tập S thành K tập con S_1, S_2, ..., S_K, độ lợi "
        "thông tin (Information Gain) đạt được xác định theo phương trình (3-4):"
    )
    eq_3_4 = (
        omml_r("Gain", italic=False) + omml_delim(omml_r("S") + omml_r(", ", italic=False) + omml_r("A"))
        + omml_r(" = ", italic=False)
        + omml_r("H") + omml_delim(omml_r("S"))
        + omml_r(" − ", italic=False)
        + omml_sum(
            omml_r("k") + omml_r("=1", italic=False),
            omml_r("K"),
            omml_frac(
                omml_delim(omml_sub(omml_r("S"), omml_r("k")), beg="|", end="|"),
                omml_delim(omml_r("S"), beg="|", end="|")
            )
            + omml_r("H") + omml_delim(omml_sub(omml_r("S"), omml_r("k")))
        )
    )
    add_tlu_equation(doc, eq_3_4, "3-4")
    add_content(
        doc,
        "Trong đó: |S| và |S_k| lần lượt là lực lượng mẫu tại nút cha S và nhánh con thứ k. Tại mỗi bước đệ quy, ID3 "
        "chọn thuộc tính A* có Gain(S, A) lớn nhất làm nút phân nhánh và loại A* khỏi danh sách thuộc tính ứng viên "
        "trên nhánh con đó. Lưu ý rằng trong bài giảng trên lớp [1], Entropy đôi khi được tính theo logarit tự nhiên "
        "ln (đơn vị nat), trong khi tài liệu thực hành sử dụng log_2 (đơn vị bit); theo đẳng thức đổi cơ số logarit "
        "(3-5), hai đại lượng này chỉ sai khác nhau hằng số nhân dương 1/ln(2) ≈ 1,4427 nên thứ tự xếp hạng thuộc tính "
        "và cấu trúc cây quyết định ID3 sinh ra hoàn toàn bất biến 100%:"
    )
    eq_3_5 = (
        omml_sub(omml_r("log", italic=False), omml_r("2", italic=False)) + omml_delim(omml_sub(omml_r("p"), omml_r("c")))
        + omml_r(" = ", italic=False)
        + omml_frac(
            omml_r("ln", italic=False) + omml_delim(omml_sub(omml_r("p"), omml_r("c"))),
            omml_r("ln", italic=False) + omml_delim(omml_r("2", italic=False))
        )
    )
    add_tlu_equation(doc, eq_3_5, "3-5")
    add_content(
        doc,
        "Trong đó: ln(·) là logarit tự nhiên cơ số e ≈ 2,71828. Đặc biệt, do lớp DecisionTreeClassifier của thư viện "
        "scikit-learn chỉ cài đặt cây nhị phân CART (kể cả khi truyền tham số criterion='entropy'), nhóm đã tự xây dựng "
        "hoàn chỉnh lớp ID3DecisionTree thuần Python trong tệp id3_tu_cai_dat.py để thực thi đúng phép phân nhánh đa "
        "hướng của Quinlan [6], đồng thời bổ sung cơ chế lưu phân phối xác suất lớp tại nút lá phục vụ tính toán ROC-AUC."
    )
    add_content(
        doc,
        "Để lựa chọn độ sâu cắt tỉa trước (Pre-pruning) một cách khách quan mà tuyệt đối không chạm vào tập kiểm tra, "
        "nhóm thực hiện kiểm định chéo phân tầng 5-Fold trên tập huấn luyện cho cả ID3 và CART (Bảng 3.2 và Hình 3.1). "
        "Số liệu chỉ rõ cây đa nhánh ID3 đạt đỉnh CV ROC-AUC = 0,8843 tại max_depth = 4 (gồm 477 nút và 370 nút lá); "
        "khi tăng độ sâu lên 6–11, điểm trên tập huấn luyện vọt lên 0,9498 nhưng điểm kiểm định chéo suy giảm còn "
        "0,8643 do phân nhánh đa hướng làm số nhánh con tăng bùng nổ khiến mỗi lá chứa quá ít mẫu (quá khớp). Tại nút "
        "gốc của cây ID3 (H(S) = 0,7578 bit), bảng xếp hạng độ lợi thông tin của cả 11 thuộc tính được xuất trực tiếp "
        "tại Bảng 3.3 và minh họa ở Hình 3.2, trong đó loan_grade dẫn đầu với IG = 0,1078 bit và chia thành 7 nhánh A–G."
    )
    add_tlu_table(
        doc, 3, 2,
        "Kết quả khảo sát kiểm định chéo 5-Fold chọn độ sâu tối ưu cho ID3 và CART",
        df_cv_depth,
        col_headers=["Mô hình", "Giới hạn độ sâu", "Train Acc", "5-Fold CV Acc", "Train ROC-AUC", "5-Fold CV ROC-AUC"],
        font_size_pt=10.0,
    )
    add_tlu_figure(
        doc, os.path.join(FIGURES_DIR, "03_chon_do_sau_chong_overfitting.png"),
        3, 1, "Đường cong ROC-AUC trên tập huấn luyện và kiểm định chéo 5-Fold theo độ sâu cây", width_cm=15.4
    )
    add_tlu_table(
        doc, 3, 3,
        "Xếp hạng độ lợi thông tin của 11 thuộc tính tại nút gốc cây quyết định ID3",
        df_id3_ig,
        col_headers=["Hạng", "Thuộc tính", "Entropy gốc H(S)", "Information Gain (bit)", "Entropy còn lại H(A, S)"],
        font_size_pt=10.5,
    )
    add_tlu_figure(
        doc, os.path.join(FIGURES_DIR, "03_information_gain_nut_goc_id3.png"),
        3, 2, "Xếp hạng độ lợi thông tin của 11 thuộc tính tại nút gốc cây quyết định ID3", width_cm=15.0
    )

    # --- 3.3 Decision Tree CART ---
    add_h2(doc, "Thuật toán Cây quyết định CART")
    add_content(
        doc,
        "Khác với cây đa nhánh ID3, thuật toán CART (Classification and Regression Trees) của Breiman et al. [7] luôn "
        "phân tách mỗi nút cha thành đúng hai nút con nhị phân (Đúng / Sai) theo điều kiện ngưỡng x_j ≤ t trực tiếp "
        "trên biến số thực mà không cần rời rạc hóa trước. Độ tạp chất tại một nút S được đo bằng chỉ số Gini theo "
        "phương trình (3-6), và chất lượng của phép chia nhị phân tại đặc trưng j với ngưỡng t được đánh giá qua chỉ "
        "số Gini có trọng số theo phương trình (3-7):"
    )
    eq_3_6 = (
        omml_r("Gini", italic=False) + omml_delim(omml_r("S"))
        + omml_r(" = 1 − ", italic=False)
        + omml_sum(
            omml_r("c") + omml_r("∈{0,1}", italic=False),
            "",
            omml_sup(omml_sub(omml_r("p"), omml_r("c")), omml_r("2", italic=False))
        )
        + omml_r(" = 2", italic=False)
        + omml_sub(omml_r("p"), omml_r("1", italic=False))
        + omml_delim(omml_r("1 − ", italic=False) + omml_sub(omml_r("p"), omml_r("1", italic=False)))
    )
    add_tlu_equation(doc, eq_3_6, "3-6")
    eq_3_7 = (
        omml_sub(omml_r("Gini", italic=False), omml_r("split", italic=False))
        + omml_delim(omml_r("S") + omml_r(", ", italic=False) + omml_r("j") + omml_r(", ", italic=False) + omml_r("t"))
        + omml_r(" = ", italic=False)
        + omml_frac(omml_delim(omml_sub(omml_r("S"), omml_r("trái", italic=False)), beg="|", end="|"), omml_delim(omml_r("S"), beg="|", end="|"))
        + omml_r("Gini", italic=False) + omml_delim(omml_sub(omml_r("S"), omml_r("trái", italic=False)))
        + omml_r(" + ", italic=False)
        + omml_frac(omml_delim(omml_sub(omml_r("S"), omml_r("phải", italic=False)), beg="|", end="|"), omml_delim(omml_r("S"), beg="|", end="|"))
        + omml_r("Gini", italic=False) + omml_delim(omml_sub(omml_r("S"), omml_r("phải", italic=False)))
    )
    add_tlu_equation(doc, eq_3_7, "3-7")
    add_content(
        doc,
        "Trong đó: S_trái = {x ∈ S | x_j ≤ t} và S_phải = {x ∈ S | x_j > t} là hai tập con sau phép cắt nhị phân; "
        "cặp (j*, t*) tối ưu được chọn sao cho Gini_split đạt cực tiểu. Khảo sát 5-Fold CV ở Bảng 3.2 cho thấy cây nhị "
        "phân CART cần độ sâu lớn hơn ID3 (max_depth = 10 kết hợp min_samples_leaf = 10, tạo ra 140 nút lá với CV "
        "ROC-AUC đạt 0,9140) vì mỗi tầng cây nhị phân chỉ tách đôi dữ liệu. Hình 3.3 minh họa ba tầng đầu tiên của cây "
        "CART: nút gốc tách tại ngưỡng loan_percent_income ≤ 0,305; đáng chú ý, nhánh khách hàng vay vượt quá 30,5% "
        "thu nhập đồng thời đang ở nhà thuê đạt độ tinh khiết tuyệt đối (Gini = 0,0) với tỷ lệ vỡ nợ rất cao."
    )
    add_tlu_figure(
        doc, os.path.join(FIGURES_DIR, "03_cay_quyet_dinh_cart_minh_hoa.png"),
        3, 3, "Trực quan hóa ba tầng đầu tiên của cây quyết định nhị phân CART", width_cm=15.6
    )

    # --- 3.4 Random Forest ---
    add_h2(doc, "Thuật toán Rừng ngẫu nhiên")
    add_content(
        doc,
        "Một cây quyết định đơn lẻ thường nhạy cảm với biến động nhỏ của tập mẫu huấn luyện (phương sai cao). Thuật "
        "toán Rừng ngẫu nhiên (Random Forest) do Breiman đề xuất [8] khắc phục nhược điểm này bằng cách huấn luyện song "
        "song B = 200 cây quyết định nhị phân thông qua hai cơ chế ngẫu nhiên hóa độc lập [1]:"
    )
    add_bullet_item(
        doc, "Lấy mẫu có hoàn lại Bootstrap và kiểm định ngoài túi Out-of-Bag (OOB): ",
        "Mỗi cây thứ b được huấn luyện trên tập mẫu Bootstrap D_b rút ngẫu nhiên có hoàn lại N lần từ N = 25.927 mẫu "
        "tập huấn luyện. Xác suất để một bản ghi cụ thể không được chọn vào D_b tuân theo giới hạn giải tích ở phương "
        "trình (3-8), tức trung bình có khoảng 36,8% số mẫu nằm ngoài túi được dùng làm tập kiểm định nội tại OOB."
    )
    add_bullet_item(
        doc, "Lấy mẫu ngẫu nhiên không gian con đặc trưng (Feature Subspace): ",
        "Tại mỗi nút chia, thay vì duyệt toàn bộ d = 19 đặc trưng, thuật toán chỉ rút ngẫu nhiên m = ⌊√19⌋ = 4 đặc "
        "trưng ứng viên. Việc này phá vỡ sự thống trị của vài đặc trưng quá mạnh ở tầng trên, làm giảm hệ số tương "
        "quan giữa các cây thành viên và giảm mạnh phương sai tổng thể khi lấy trung bình xác suất dự đoán theo "
        "phương trình (3-9)."
    )
    eq_3_8 = (
        omml_r("P") + omml_delim(omml_r("x") + omml_r(" ∉ ", italic=False) + omml_sub(omml_r("D"), omml_r("b")))
        + omml_r(" = ", italic=False)
        + omml_sup(
            omml_delim(omml_r("1 − ", italic=False) + omml_frac(omml_r("1", italic=False), omml_r("N"))),
            omml_r("N")
        )
        + omml_r(" ≈ ", italic=False)
        + omml_frac(omml_r("1", italic=False), omml_r("e"))
        + omml_r(" ≈ 0,368", italic=False)
    )
    add_tlu_equation(doc, eq_3_8, "3-8")
    eq_3_9 = (
        omml_sub(omml_hat(omml_r("P")), omml_r("RF", italic=False))
        + omml_delim(omml_r("y") + omml_r(" = 1 | ", italic=False) + omml_r("x", bold=True))
        + omml_r(" = ", italic=False)
        + omml_frac(omml_r("1", italic=False), omml_r("B"))
        + omml_sum(
            omml_r("b") + omml_r("=1", italic=False),
            omml_r("B"),
            omml_sub(omml_hat(omml_r("P")), omml_r("b"))
            + omml_delim(omml_r("y") + omml_r(" = 1 | ", italic=False) + omml_r("x", bold=True))
        )
    )
    add_tlu_equation(doc, eq_3_9, "3-9")
    add_content(
        doc,
        "Trong đó: D_b là tập mẫu Bootstrap thứ b; B = 200 là tổng số cây trong rừng; P̂_b(y = 1 | x) là xác suất dự "
        "đoán của cây thứ b. Trên thực nghiệm, điểm đánh giá nội tại OOB Score của mô hình đạt 0,9320 (93,20%). Đặc "
        "biệt, Bảng 3.4 và Hình 3.4 chứng minh rõ hiệu quả của cơ chế không gian con đặc trưng: trên cây CART đơn lẻ, "
        "biến lãi suất loan_int_rate chỉ có độ quan trọng 0,0090 do bị loan_grade che khuất; sang Random Forest, "
        "loan_int_rate được khai thác hiệu quả với độ quan trọng tăng gấp 11 lần lên 0,0997."
    )
    add_tlu_table(
        doc, 3, 4,
        "Đối chiếu mức độ quan trọng của 10 đặc trưng hàng đầu giữa CART và Random Forest",
        df_feat_imp.head(10),
        col_headers=["Đặc trưng", "Độ quan trọng trên CART", "Độ quan trọng trên Random Forest"],
        font_size_pt=10.5,
    )
    add_tlu_figure(
        doc, os.path.join(FIGURES_DIR, "03_top10_feature_importance_rf.png"),
        3, 4, "Xếp hạng 10 đặc trưng quan trọng nhất trong mô hình Rừng ngẫu nhiên", width_cm=15.0
    )

    # --- 3.5 Kết quả thực nghiệm & đánh giá ---
    add_h2(doc, "Kết quả thực nghiệm, đánh giá và thảo luận chuyên sâu")
    add_content(
        doc,
        "Sau khi hoàn tất giai đoạn huấn luyện trên 25.927 mẫu tập Train, cả bốn mô hình được mở niêm phong để đánh "
        "giá độc lập một lần duy nhất trên tập kiểm tra gồm 6.482 hồ sơ khách hàng (5.064 khách hàng trả tốt – 78,12% "
        "và 1.418 khách hàng vỡ nợ – 21,88%) bằng tệp mã nguồn 04_danh_gia_va_so_sanh.py."
    )

    add_h3(doc, "Ý nghĩa các độ đo đánh giá trong quản trị rủi ro tín dụng")
    add_content(
        doc,
        "Từ bốn thành phần của ma trận nhầm lẫn (Confusion Matrix) gồm Dương tính thật (TP – phát hiện đúng người sẽ "
        "vỡ nợ), Âm tính thật (TN – duyệt đúng người trả nợ tốt), Dương tính giả (FP – từ chối nhầm khách hàng tốt) và "
        "Âm tính giả (FN – giải ngân nhầm cho người sẽ vỡ nợ), các chỉ số đánh giá cốt lõi được xác định theo các "
        "phương trình (3-10) và (3-11) [1], [9]:"
    )
    eq_3_10 = (
        omml_r("Accuracy", italic=False) + omml_r(" = ", italic=False)
        + omml_frac(omml_r("TP + TN", italic=False), omml_r("TP + TN + FP + FN", italic=False))
        + omml_r(",   Precision", italic=False) + omml_r(" = ", italic=False)
        + omml_frac(omml_r("TP", italic=False), omml_r("TP + FP", italic=False))
        + omml_r(",   Recall", italic=False) + omml_r(" = ", italic=False)
        + omml_frac(omml_r("TP", italic=False), omml_r("TP + FN", italic=False))
    )
    add_tlu_equation(doc, eq_3_10, "3-10")
    eq_3_11 = (
        omml_sub(omml_r("F"), omml_r("1", italic=False))
        + omml_r(" = ", italic=False)
        + omml_frac(
            omml_r("2 · Precision · Recall", italic=False),
            omml_r("Precision + Recall", italic=False)
        )
    )
    add_tlu_equation(doc, eq_3_11, "3-11")
    add_content(
        doc,
        "Trong đó: Accuracy là độ chính xác tổng thể; Precision đo tỷ lệ dự báo nợ xấu chuẩn xác; Recall (độ nhạy hay "
        "TPR) đo tỷ lệ nợ xấu thực tế bị bắt giữ; F_1 là trung bình điều hòa cân bằng giữa Precision và Recall. Dưới "
        "góc độ quản trị ngân hàng, sai lầm loại II (FN – bỏ lọt nợ xấu) nguy hiểm gấp nhiều lần sai lầm loại I (FP – "
        "báo động giả): một ca FP chỉ làm ngân hàng mất chi phí cơ hội thu tiền lãi, trong khi một ca FN khiến ngân "
        "hàng mất trắng toàn bộ vốn gốc giải ngân. Tuy nhiên, nếu hạ ngưỡng phân loại cực đoan để đẩy Recall lên 100% "
        "thì FP sẽ tăng vọt làm tê liệt hoạt động tín dụng; do đó nhóm kết hợp thêm diện tích dưới đường cong đặc trưng "
        "hoạt động ROC-AUC [9] để đo lường khả năng phân hạng rủi ro tổng quát trên mọi ngưỡng quyết định từ 0 đến 1."
    )

    add_h3(doc, "Bảng tổng hợp kết quả thực nghiệm và trực quan hóa")
    add_content(
        doc,
        "Kết quả thực nghiệm định lượng của cả bốn thuật toán trên tập huấn luyện và tập kiểm tra độc lập tại ngưỡng "
        "mặc định 0,5 được tổng hợp đầy đủ tại Bảng 3.5; chi tiết bốn ô của ma trận nhầm lẫn cùng tỷ lệ sai lầm tài "
        "chính được đối chiếu tại Bảng 3.6 và trực quan hóa qua các biểu đồ Hình 3.5, Hình 3.6 và Hình 3.7."
    )
    add_tlu_table(
        doc, 3, 5,
        "Bảng so sánh tổng hợp kết quả thực nghiệm 4 thuật toán trên tập huấn luyện và kiểm tra",
        df_comp_4,
        col_headers=["Mô hình", "Train Acc", "Test Acc", "Precision", "Recall", "F1-Score", "Train AUC", "Test AUC", "Độ lệch AUC"],
        font_size_pt=10.0,
    )
    add_tlu_table(
        doc, 3, 6,
        "Chi tiết ma trận nhầm lẫn và tỷ lệ sai lầm tài chính trên tập kiểm tra độc lập",
        df_cm_detail,
        col_headers=["Mô hình", "TN (Duyệt đúng)", "FP (Từ chối nhầm)", "FN (Bỏ lọt nợ xấu)", "TP (Bắt trúng nợ xấu)", "Tỷ lệ FP (%)", "Tỷ lệ FN (%)"],
        font_size_pt=10.0,
    )
    add_tlu_figure(
        doc, os.path.join(FIGURES_DIR, "04_confusion_matrix_4_mo_hinh.png"),
        3, 5, "So sánh trực quan ma trận nhầm lẫn của 4 thuật toán trên tập kiểm tra độc lập", width_cm=15.4
    )
    add_tlu_figure(
        doc, os.path.join(FIGURES_DIR, "04_roc_curve_so_sanh_4_mo_hinh.png"),
        3, 6, "Đường cong ROC và diện tích dưới đường cong AUC của 4 thuật toán trên tập kiểm tra", width_cm=14.2
    )
    add_tlu_figure(
        doc, os.path.join(FIGURES_DIR, "04_so_sanh_5_chi_so_test.png"),
        3, 7, "Biểu đồ cột so sánh 5 độ đo đánh giá trên tập kiểm tra của 4 thuật toán", width_cm=15.4
    )

    add_h3(doc, "Phân tích và thảo luận chuyên sâu từ số liệu thực nghiệm")
    add_content(
        doc,
        "Từ các con số thực nghiệm tại Bảng 3.5, Bảng 3.6 và ba biểu đồ Hình 3.5, Hình 3.6, Hình 3.7, nhóm rút ra bốn "
        "nhận định chuyên môn quan trọng:"
    )
    add_bullet_item(
        doc, "Thứ nhất – Kiểm soát hiện tượng quá khớp (Overfitting) chặt chẽ trên cả 4 mô hình: ",
        "Độ chênh lệch giữa độ chính xác huấn luyện (Train Acc) và độ chính xác kiểm tra (Test Acc) của cả bốn thuật "
        "toán đều nhỏ hơn 0,7 điểm phần trăm (Logistic: 85,46% so với 84,85%; ID3: 88,26% so với 88,49%; CART: 93,44% "
        "so với 93,20%; Random Forest: 93,66% so với 93,52%). Đặc biệt, độ chính xác kiểm tra độc lập của Random Forest "
        "(93,52%) khớp rất sát với ước lượng nội tại OOB Score (93,20%), kiểm chứng thực nghiệm thuyết phục cho lý "
        "thuyết Bootstrap ở phương trình (3-8)."
    )
    add_bullet_item(
        doc, "Thứ hai – Minh chứng nghịch lý độ chính xác và sự tiến bộ bậc thang giữa các họ mô hình: ",
        "Một bộ phân lớp ngây thơ duyệt toàn bộ hồ sơ vẫn đạt Accuracy = 78,12% nhưng Recall = 0%. Hồi quy Logistic "
        "nâng Accuracy lên 84,85% nhưng vẫn bỏ lọt tới FN = 711 ca nợ xấu (Recall chỉ đạt 49,86%; F1 = 0,5902) do biên "
        "quyết định siêu phẳng tuyến tính không bắt được các tổ hợp điều kiện ngưỡng phi tuyến. Cây đa nhánh ID3 tự cài "
        "đặt khắc phục được tính phi tuyến, giảm 219 ca bỏ lọt nợ xấu (FN giảm còn 492; Recall tăng lên 65,30%; F1 = "
        "0,7129), song việc gom biến liên tục thành 4 khoảng tứ phân vị làm mất độ mịn tại biên cắt. Cây nhị phân CART "
        "chia ngưỡng trực tiếp trên số thực nên tạo bước nhảy vọt lớn lên F1 = 0,8231 và ROC-AUC = 0,9178."
    )
    add_bullet_item(
        doc, "Thứ ba – Đối chiếu nghiệp vụ giữa CART và Random Forest: ",
        "Tại ngưỡng mặc định 0,5, CART bắt nhiều hơn Random Forest 12 ca nợ xấu (TP = 1.026 so với 1.014; Recall = "
        "72,36% so với 71,51%), nhưng đổi lại Random Forest giảm số khách hàng tốt bị từ chối oan (FP) hơn 3 lần – từ "
        "49 người xuống chỉ còn đúng 16 người trên tổng số 5.064 khách hàng tốt (tỷ lệ FP chỉ 0,32%; Precision đạt mức "
        "xuất sắc 98,45%). Nhờ đó Random Forest dẫn đầu toàn diện về Test Accuracy (93,52%), F1-Score (0,8284) và đặc "
        "biệt là ROC-AUC (0,9327 – Hình 3.6). Đường cong ROC bao trùm phía trên cho phép ngân hàng linh hoạt hạ nhẹ "
        "ngưỡng quyết định của Random Forest xuống vùng 0,35–0,40 trong giai đoạn kinh tế biến động để đẩy Recall vượt "
        "trên 76% mà vẫn giữ tỷ lệ từ chối nhầm thấp hơn các mô hình đơn lẻ."
    )
    add_bullet_item(
        doc, "Thứ tư – Sự đánh đổi giữa độ chính xác kết hợp và tính giải thích minh bạch: ",
        "Mặc dù Random Forest đạt hiệu năng dự báo cao nhất, tổ hợp 200 cây hoạt động tương tự một hộp đen khó diễn "
        "giải lý do từ chối cụ thể cho từng khách hàng đơn lẻ. Trong khi đó, Hồi quy Logistic (qua tỷ số chênh Odds "
        "Ratio ở Bảng 3.1) và cây ID3/CART (qua tập luật Nếu – Thì tường minh) mang lại tính giải thích rất cao, phù "
        "hợp để sử dụng song hành cùng Random Forest trong hệ thống hỗ trợ thẩm định tín dụng thực tế."
    )

    # -------------------------------------------------------------------------
    # CHƯƠNG 4: XÂY DỰNG CHƯƠNG TRÌNH DỰ ĐOÁN
    # -------------------------------------------------------------------------
    add_chapter_heading(doc, "XÂY DỰNG CHƯƠNG TRÌNH DỰ ĐOÁN", page_break_before=True)

    add_h2(doc, "Tổ chức cấu trúc mã nguồn chương trình")
    add_content(
        doc,
        "Chương trình thực nghiệm của đề tài được tổ chức cô lập, rõ ràng trong thư mục Do_An_Tin_Dung/ theo đúng "
        "trình tự các chương của báo cáo nhằm đảm bảo tính dễ đọc, dễ bảo trì và tái lập kết quả 100%. Sáu tệp mã "
        "nguồn trong thư mục src/ bao gồm: (1) 01_tai_va_kham_pha_du_lieu.py tải dữ liệu OpenML và xuất bảng biểu, "
        "hình ảnh EDA; (2) 02_tien_xu_ly_du_lieu.py thực thi làm sạch, chia phân tầng, biến đổi dữ liệu và đóng gói "
        "tham số preprocessor_bundle.joblib; (3) id3_tu_cai_dat.py cài đặt lớp ID3DecisionTree đa nhánh từ đầu; "
        "(4) 03_huan_luyen_mo_hinh.py chạy kiểm định chéo 5-Fold và huấn luyện, lưu trữ bốn mô hình; "
        "(5) 04_danh_gia_va_so_sanh.py tính toán toàn bộ độ đo đánh giá và vẽ biểu đồ đối chiếu; và "
        "(6) 05_du_doan_khach_hang_moi.py triển khai hàm dự đoán hồ sơ mới. Cấu hình chi tiết và thời gian huấn luyện "
        "của bốn mô hình đã đóng gói được trình bày tại Bảng 4.1."
    )
    add_tlu_table(
        doc, 4, 1,
        "Tổng hợp cấu hình lưu trữ và thời gian huấn luyện thực tế của 4 mô hình",
        df_model_cfg,
        col_headers=["Mô hình", "Nhánh dữ liệu huấn luyện", "Siêu tham số chính", "Thời gian (s)", "Thông tin cấu trúc"],
        font_size_pt=10.0,
    )

    add_h2(doc, "Quy trình lưu trữ Pipeline và hàm dự đoán khách hàng mới")
    add_content(
        doc,
        "Để ngăn chặn triệt để lỗi lệch phân phối giữa giai đoạn huấn luyện và giai đoạn triển khai thực tế "
        "(Training-Serving Skew), nhóm đóng gói toàn bộ các đối tượng biến đổi đã học từ tập huấn luyện vào tệp "
        "preprocessor_bundle.joblib (bao gồm bộ điền khuyết trung vị/yếu vị, danh sách 19 cột mã hóa One-Hot/Ordinal, "
        "bộ chuẩn hóa StandardScaler và từ điển các mốc tứ phân vị của ID3). Khi có một hồ sơ khách hàng mới dưới dạng "
        "từ điển Python (dict), hàm predict_loan_status(customer_data, model_name='random_forest', threshold=0.5) chỉ "
        "gọi phương thức biến đổi (.transform()) dựa trên hệ quy chiếu của tập huấn luyện mà tuyệt đối không gọi lại "
        ".fit_transform(), sau đó nạp mô hình tương ứng để trả về nhãn dự đoán, xác suất vỡ nợ, mức phân tầng rủi ro "
        "và khuyến nghị xét duyệt tín dụng."
    )

    add_h2(doc, "Kết quả chạy thử minh họa trên các hồ sơ khách hàng mẫu")
    add_content(
        doc,
        "Nhóm tiến hành kiểm thử hàm predict_loan_status với cả bốn mô hình trên ba kịch bản hồ sơ thực tế đại diện "
        "cho các tình huống nghiệp vụ điển hình tại ngân hàng: (1) Hồ sơ khách hàng tốt (30 tuổi, thu nhập 85.000 USD, "
        "sở hữu nhà riêng, hạng tín nhiệm A, vay 8.000 USD phục vụ giáo dục); (2) Hồ sơ khách hàng rủi ro cao (22 tuổi, "
        "thu nhập 24.000 USD, ở nhà thuê, hạng E, vay 10.000 USD chiếm 42% thu nhập để đảo nợ và từng có lịch sử nợ "
        "xấu); và (3) Hồ sơ bị bỏ trống thông tin lãi suất và thâm niên làm việc nhằm kiểm chứng cơ chế tự động điền "
        "khuyết. Toàn bộ kết quả chạy thực tế từ chương trình được tổng hợp tại Bảng 4.2, cho thấy sự đồng thuận rất "
        "cao và hợp lý về mặt nghiệp vụ của cả bốn mô hình."
    )
    add_tlu_table(
        doc, 4, 2,
        "Kết quả chạy thử hàm dự đoán khách hàng mới với cả 4 mô hình trên 3 kịch bản",
        df_demo_pred,
        col_headers=["Hồ sơ khách hàng kiểm thử", "Mô hình dự đoán", "Nhãn", "Xác suất vỡ nợ", "Mức rủi ro", "Khuyến nghị tín dụng"],
        font_size_pt=9.5,
    )

    # -------------------------------------------------------------------------
    # KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN
    # -------------------------------------------------------------------------
    add_unnumbered_h1(doc, "KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN", page_break_before=True)
    add_content(
        doc,
        "Qua quá trình nghiên cứu lý thuyết và triển khai thực nghiệm hoàn chỉnh đề tài \"Dự đoán khả năng trả nợ của "
        "khách hàng vay vốn\", nhóm tổng kết các kết quả đạt được, hạn chế tồn tại và hướng phát triển tiếp theo như sau:"
    )
    add_bullet_item(
        doc, "1. Kết quả đạt được: ",
        "Nhóm đã xây dựng thành công quy trình Học máy khép kín chuẩn học thuật trên 32.581 hồ sơ tín dụng thực tế, "
        "đảm bảo chống rò rỉ dữ liệu tuyệt đối và triển khai đúng bản chất toán học cả bốn thuật toán bắt buộc (Hồi quy "
        "Logistic, Cây quyết định đa nhánh ID3 tự cài đặt thuần Python, Cây nhị phân CART và Rừng ngẫu nhiên). Mô hình "
        "Rừng ngẫu nhiên đạt hiệu năng cao nhất trên tập kiểm tra độc lập với Test Accuracy = 93,52%, Precision = "
        "98,45%, F1-Score = 0,8284 và ROC-AUC = 0,9327, đồng thời được tích hợp mượt mà vào hàm dự đoán khách hàng mới."
    )
    add_bullet_item(
        doc, "2. Hạn chế thực tế của đề tài: ",
        "Tại ngưỡng quyết định mặc định 0,5, mô hình tốt nhất vẫn còn bỏ lọt khoảng 28% số trường hợp vỡ nợ tinh vi "
        "(những hồ sơ có chỉ số thu nhập ban đầu ổn định nhưng gặp biến cố tài chính sau giải ngân). Ngoài ra, bộ dữ "
        "liệu mang tính chất lát cắt tĩnh nên chưa phản ánh các yếu tố chu kỳ kinh tế vĩ mô theo thời gian."
    )
    add_bullet_item(
        doc, "3. Hướng phát triển tiếp theo: ",
        "(i) Nghiên cứu học máy nhạy cảm chi phí (Cost-Sensitive Learning) và tối ưu hóa ngưỡng quyết định dựa trên ma "
        "trận lợi nhuận lãi vay so với tổn thất mất vốn gốc thực tế của ngân hàng; (ii) Kết hợp kỹ thuật giải thích hậu "
        "nghiệm SHAP để minh bạch hóa phán quyết của Rừng ngẫu nhiên cho từng hồ sơ vay; (iii) Xây dựng giao diện phần "
        "mềm tra cứu nội bộ phục vụ trực tiếp cán bộ thẩm định tín dụng."
    )

    # -------------------------------------------------------------------------
    # TÀI LIỆU THAM KHẢO (CHUẨN IEEE 100% THEO MỤC 2.4 & 2.5 CỦA MẪU ĐHTL)
    # -------------------------------------------------------------------------
    add_unnumbered_h1(doc, "TÀI LIỆU THAM KHẢO", page_break_before=True)

    # Danh sách trích dẫn xếp đúng thứ tự xuất hiện [1] -> [9] (IEEE Reference Order)
    # Mỗi phần tử là danh sách các đoạn (text, is_italic) để in nghiêng chuẩn tên sách/tạp chí
    ieee_refs = [
        ("[1]", [
            ("T. Q. Chiểu, ", False),
            ("Bài giảng và Tài liệu hướng dẫn thực hành môn Học máy (Machine Learning)", True),
            (", Khoa Công nghệ thông tin, Trường Đại học Thủy Lợi, Hà Nội, 2025.", False),
        ]),
        ("[2]", [
            ("V. H. Tiệp, ", False),
            ("Machine Learning cơ bản", True),
            (". Hà Nội: NXB Khoa học và Kỹ thuật, 2020.", False),
        ]),
        ("[3]", [
            ("J. Vanschoren ", False), ("et al.", True),
            (', "OpenML: networked science in machine learning," ', False),
            ("ACM SIGKDD Explorations Newsletter", True),
            (", vol. 15, no. 2, pp. 49–60, Jun. 2014. [Online]. Available: https://www.openml.org/d/43454. [Accessed: Oct. 31, 2026].", False),
        ]),
        ("[4]", [
            ("F. Pedregosa ", False), ("et al.", True),
            (', "Scikit-learn: Machine learning in Python," ', False),
            ("Journal of Machine Learning Research", True),
            (", vol. 12, pp. 2825–2830, Oct. 2011.", False),
        ]),
        ("[5]", [
            ('D. R. Cox, "The regression analysis of binary sequences," ', False),
            ("Journal of the Royal Statistical Society: Series B (Methodological)", True),
            (", vol. 20, no. 2, pp. 215–232, Jul. 1958.", False),
        ]),
        ("[6]", [
            ('J. R. Quinlan, "Induction of decision trees," ', False),
            ("Machine Learning", True),
            (", vol. 1, no. 1, pp. 81–106, Mar. 1986.", False),
        ]),
        ("[7]", [
            ("L. Breiman ", False), ("et al.", True), (", ", False),
            ("Classification and Regression Trees", True),
            (". Belmont, CA, USA: Wadsworth International Group, 1984.", False),
        ]),
        ("[8]", [
            ('L. Breiman, "Random forests," ', False),
            ("Machine Learning", True),
            (", vol. 45, no. 1, pp. 5–32, Oct. 2001.", False),
        ]),
        ("[9]", [
            ('T. Fawcett, "An introduction to ROC analysis," ', False),
            ("Pattern Recognition Letters", True),
            (", vol. 27, no. 8, pp. 861–874, Jun. 2006.", False),
        ]),
    ]

    tbl_bib = doc.add_table(rows=len(ieee_refs), cols=2)
    tbl_bib.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders_none(tbl_bib)
    for idx_r, (num_tag, segments) in enumerate(ieee_refs):
        tr = tbl_bib.rows[idx_r]
        tr._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))
        c0, c1 = tr.cells[0], tr.cells[1]
        c0.width, c1.width = Cm(1.3), Cm(14.7)
        set_cell_margins(c0, 60, 60, 20, 40)
        set_cell_margins(c1, 60, 60, 40, 20)

        p0 = c0.paragraphs[0]
        p0.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p0.paragraph_format.line_spacing = 1.3
        r_tag = p0.add_run(num_tag)
        format_run_tnr(r_tag, size_pt=13, bold=False)

        p1 = c1.paragraphs[0]
        p1.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p1.paragraph_format.line_spacing = 1.3
        for seg_txt, seg_ita in segments:
            r_seg = p1.add_run(seg_txt)
            format_run_tnr(r_seg, size_pt=13, bold=False, italic=seg_ita)

    # -------------------------------------------------------------------------
    # PHỤ LỤC (CHUẨN MỤC 1.3 MẪU ĐHTL: SƠ ĐỒ CÂY ID3 & MÃ NGUỒN MINH CHỨNG)
    # -------------------------------------------------------------------------
    add_unnumbered_h1(doc, "PHỤ LỤC", page_break_before=True)

    p_pa = doc.add_paragraph(style="Content")
    r_pa = p_pa.add_run("Phụ lục A. Sơ đồ cấu trúc hai tầng nhánh đầu tiên của cây quyết định đa nhánh ID3 tự cài đặt")
    format_run_tnr(r_pa, size_pt=13, bold=True)
    add_content(
        doc,
        "Sơ đồ dưới đây được trích xuất tự động từ mô hình ID3 đã huấn luyện trên tập Train (N = 25.927 mẫu) tại tệp "
        "outputs/tables/03_so_do_nhanh_dau_cay_id3.txt, minh chứng trực tiếp cơ chế phân nhánh đa hướng theo 7 hạng "
        "tín nhiệm A–G tại nút gốc loan_grade:"
    )
    id3_compact = "\n".join([ln for ln in id3_tree_txt.splitlines() if ln.strip()])
    add_code_block(doc, id3_compact)

    p_pb = doc.add_paragraph(style="Content")
    r_pb = p_pb.add_run("Phụ lục B. Hướng dẫn thực thi tái lập 100% kết quả thực nghiệm và báo cáo")
    format_run_tnr(r_pb, size_pt=13, bold=True)
    add_content(
        doc,
        "Để chạy lại toàn bộ quy trình từ tải dữ liệu thô, tiền xử lý chống rò rỉ dữ liệu, huấn luyện bốn mô hình, "
        "xuất biểu đồ đánh giá đến sinh tệp báo cáo Word chuẩn định dạng, mở cửa sổ dòng lệnh tại thư mục gốc của kho "
        "lưu trữ và thực thi tuần tự các lệnh sau:"
    )
    repro_cmds = (
        "pip install -r Do_An_Tin_Dung/requirements.txt\n"
        "python Do_An_Tin_Dung/src/01_tai_va_kham_pha_du_lieu.py\n"
        "python Do_An_Tin_Dung/src/02_tien_xu_ly_du_lieu.py\n"
        "python Do_An_Tin_Dung/src/03_huan_luyen_mo_hinh.py\n"
        "python Do_An_Tin_Dung/src/04_danh_gia_va_so_sanh.py\n"
        "python Do_An_Tin_Dung/src/05_du_doan_khach_hang_moi.py\n"
        "python Do_An_Tin_Dung/report/xuat_bao_cao_docx.py"
    )
    add_code_block(doc, repro_cmds)

    # Lưu tài liệu và đồng bộ cơ sở dữ liệu nguồn trích dẫn IEEE vào customXml/item1.xml
    doc.save(OUTPUT_DOCX)
    inject_ieee_sources_xml(OUTPUT_DOCX)

    print(f"[OK] Đã tạo thành công Báo cáo chuẩn mẫu ĐH Thủy Lợi của Thầy Tạ Quang Chiểu tại:\n     {OUTPUT_DOCX}")
    print(f"[OK] Dung lượng file Word: {os.path.getsize(OUTPUT_DOCX) / (1024 * 1024):.2f} MB")


if __name__ == "__main__":
    build_report()
