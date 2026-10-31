"""
MODULE HỖ TRỢ ĐỊNH DẠNG BÁO CÁO WORD CHUẨN QUY ĐỊNH TRƯỜNG ĐẠI HỌC THỦY LỢI
Bám sát 100% file mẫu `Huong dan trinh bay DATN (26-2-2020).docx` do Thầy Tạ Quang Chiểu cung cấp:
- Khổ giấy A4 (21,0 x 29,7 cm); Lề Trái 3,0 cm, Phải 2,0 cm, Trên 2,5 cm, Dưới 2,5 cm; Footer 0,5 cm.
- Chia 3 Section: Bìa & Tờ nhiệm vụ (không đánh số) -> Mở đầu (i, ii, iii...) -> Nội dung (1, 2, 3...).
- Đúng các Style chuẩn: Heading 1 (CHƯƠNG 1..), Heading 2 (1.1..), Heading 3 (3.5.1..), Heading 1N,
  Content (Times New Roman 13pt, 1.5 lines, before 10pt, justify), Bullet, Caption (12pt italic center).
- Công thức toán học tạo bằng đối tượng Word Equation gốc (<m:oMath>) kèm số hiệu (x-y) sát lề phải.
- Trích dẫn và Danh mục Tài liệu tham khảo chuẩn kiểu IEEE [1], [2]...
"""

import copy
import zipfile
import xml.etree.ElementTree as ET
from docx.shared import Cm, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import qn, nsdecls

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
M_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"


def enforce_times_new_roman_on_style(style, size_pt, bold=None, italic=None, all_caps=False,
                                     space_before_pt=None, space_after_pt=None, line_spacing=None,
                                     align=None, first_line_Zero=True):
    """Ép cứng phông Times New Roman đen đậm trên cả 4 bảng mã (ascii, hAnsi, eastAsia, cs) của Style."""
    font = style.font
    font.name = "Times New Roman"
    font.size = Pt(size_pt)
    font.color.rgb = RGBColor(0x00, 0x00, 0x00)
    if bold is not None:
        font.bold = bold
    if italic is not None:
        font.italic = italic
    font.all_caps = all_caps

    rPr = style._element.get_or_add_rPr()
    rFonts = rPr.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = OxmlElement("w:rFonts")
        rPr.insert(0, rFonts)
    # Xóa các thuộc tính theme font để không bao giờ bị nhảy sang Cambria/Calibri
    for attr in list(rFonts.attrib.keys()):
        if "Theme" in attr or "theme" in attr:
            del rFonts.attrib[attr]
    rFonts.set(qn("w:ascii"), "Times New Roman")
    rFonts.set(qn("w:hAnsi"), "Times New Roman")
    rFonts.set(qn("w:eastAsia"), "Times New Roman")
    rFonts.set(qn("w:cs"), "Times New Roman")

    sz_val = str(int(round(size_pt * 2)))
    for tag in ("w:sz", "w:szCs"):
        el = rPr.find(qn(tag))
        if el is None:
            el = OxmlElement(tag)
            rPr.append(el)
        el.set(qn("w:val"), sz_val)

    if bold is False:
        for tag in ("w:b", "w:bCs"):
            el = rPr.find(qn(tag))
            if el is None:
                el = OxmlElement(tag)
                rPr.append(el)
            el.set(qn("w:val"), "0")

    pf = style.paragraph_format
    if space_before_pt is not None:
        pf.space_before = Pt(space_before_pt)
    if space_after_pt is not None:
        pf.space_after = Pt(space_after_pt)
    if line_spacing is not None:
        pf.line_spacing = line_spacing
    if align is not None:
        pf.alignment = align
    if first_line_Zero:
        pf.first_line_indent = Pt(0)


def prepare_tlu_template_document(doc):
    """
    Trích xuất 3 cấu hình Section gốc của ĐH Thủy Lợi (Bìa không số -> Mở đầu La Mã -> Nội dung Ả Rập),
    dọn sạch toàn bộ văn bản hướng dẫn mẫu, xóa các liên kết ngoài thừa và chuẩn hóa toàn bộ Styles.
    """
    body = doc._body._body
    all_sect_prs = [copy.deepcopy(sp) for sp in body.findall(f".//{{{W_NS}}}sectPr")]
    if len(all_sect_prs) >= 3:
        sect0_pr, sect1_pr, sect2_pr = all_sect_prs[0], all_sect_prs[1], all_sect_prs[-1]
    else:
        raise RuntimeError("File mẫu không đủ 3 Section chuẩn ĐH Thủy Lợi.")

    # Đảm bảo cả 3 Section đều đúng chuẩn A4 (21x29.7cm), Lề T=2.5, B=2.5, L=3.0, R=2.0, Footer=0.5cm
    for sp in (sect0_pr, sect1_pr, sect2_pr):
        pgMar = sp.find(qn("w:pgMar"))
        if pgMar is not None:
            pgMar.set(qn("w:top"), "1418")     # 2.5 cm
            pgMar.set(qn("w:bottom"), "1418")  # 2.5 cm
            pgMar.set(qn("w:left"), "1701")    # 3.0 cm
            pgMar.set(qn("w:right"), "1134")   # 2.0 cm
            pgMar.set(qn("w:header"), "283")   # 0.5 cm
            pgMar.set(qn("w:footer"), "283")   # 0.5 cm

    # Chuẩn hóa Footer của cả 3 Section trước khi dọn body (xóa dòng trống thừa dưới số trang)
    for sec in doc.sections:
        ftr = sec.footer
        if len(ftr.paragraphs) > 1:
            for extra_p in ftr.paragraphs[1:]:
                extra_p._element.getparent().remove(extra_p._element)
        if ftr.paragraphs:
            ftr.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
            for r in ftr.paragraphs[0].runs:
                format_run_tnr(r, size_pt=13, bold=False, italic=False)

    # Xóa toàn bộ phần tử con trong <w:body>, giữ lại sect2_pr làm sectPr cuối tài liệu
    for child in list(body):
        body.remove(child)
    body.append(sect2_pr)

    # Dọn các hyperlink ngoài ổ D:\, F:\ cũ trong rels của file mẫu
    stale_rids = [
        rid for rid, rel in list(doc.part.rels.items())
        if rel.is_external and ("Box" in str(rel.target_ref) or "file:///" in str(rel.target_ref))
    ]
    for rid in stale_rids:
        del doc.part.rels[rid]

    # Thiết lập thông số chính xác 100% theo Bảng 2.2 của Hướng dẫn trình bày ĐATN ĐH Thủy Lợi
    styles = doc.styles
    enforce_times_new_roman_on_style(
        styles["Normal"], 13, bold=False, italic=False,
        space_before_pt=0, space_after_pt=0, line_spacing=1.15, align=WD_ALIGN_PARAGRAPH.LEFT
    )
    enforce_times_new_roman_on_style(
        styles["Footer"], 13, bold=False, italic=False,
        space_before_pt=0, space_after_pt=0, line_spacing=1.0, align=WD_ALIGN_PARAGRAPH.CENTER
    )
    enforce_times_new_roman_on_style(
        styles["Heading 1"], 14, bold=True, italic=False, all_caps=True,
        space_before_pt=24, space_after_pt=24, line_spacing=1.0, align=WD_ALIGN_PARAGRAPH.LEFT
    )
    enforce_times_new_roman_on_style(
        styles["Heading 1N"], 14, bold=True, italic=False, all_caps=True,
        space_before_pt=24, space_after_pt=24, line_spacing=1.0, align=WD_ALIGN_PARAGRAPH.CENTER
    )
    enforce_times_new_roman_on_style(
        styles["Heading 2"], 13, bold=True, italic=False,
        space_before_pt=6, space_after_pt=12, line_spacing=1.0, align=WD_ALIGN_PARAGRAPH.LEFT,
        first_line_Zero=False
    )
    enforce_times_new_roman_on_style(
        styles["Heading 3"], 13, bold=True, italic=True,
        space_before_pt=6, space_after_pt=12, line_spacing=1.0, align=WD_ALIGN_PARAGRAPH.LEFT,
        first_line_Zero=False
    )
    enforce_times_new_roman_on_style(
        styles["Heading 4"], 13, bold=False, italic=True,
        space_before_pt=6, space_after_pt=12, line_spacing=1.0, align=WD_ALIGN_PARAGRAPH.LEFT,
        first_line_Zero=False
    )
    enforce_times_new_roman_on_style(
        styles["Content"], 13, bold=False, italic=False,
        space_before_pt=10, space_after_pt=0, line_spacing=1.5, align=WD_ALIGN_PARAGRAPH.JUSTIFY
    )
    enforce_times_new_roman_on_style(
        styles["Bullet"], 13, bold=False, italic=False,
        space_before_pt=0, space_after_pt=0, line_spacing=1.5, align=WD_ALIGN_PARAGRAPH.JUSTIFY,
        first_line_Zero=False
    )
    enforce_times_new_roman_on_style(
        styles["Caption"], 12, bold=False, italic=True,
        space_before_pt=6, space_after_pt=6, line_spacing=1.0, align=WD_ALIGN_PARAGRAPH.CENTER
    )

    # Bật tự động cập nhật trường (TOC, số trang, SEQ) khi mở bằng Microsoft Word
    settings = doc.settings.element
    if settings.find(qn("w:updateFields")) is None:
        upd = OxmlElement("w:updateFields")
        upd.set(qn("w:val"), "true")
        settings.append(upd)

    return sect0_pr, sect1_pr


def add_section_break_paragraph(doc, sect_pr_element):
    """Chèn ngắt Section (Next Page) bằng cách gắn bản sao <w:sectPr> vào <w:pPr> của đoạn cuối section."""
    p = doc.add_paragraph()
    pPr = p._p.get_or_add_pPr()
    pPr.append(copy.deepcopy(sect_pr_element))
    return p


def set_table_borders_none(table):
    """Ẩn toàn bộ đường viền của bảng (dùng cho bảng tiêu đề Trang bìa, Chữ ký, Danh mục IEEE)."""
    tblPr = table._tbl.tblPr
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        '<w:top w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
        '<w:left w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
        '<w:bottom w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
        '<w:right w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
        '<w:insideH w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
        '<w:insideV w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
        '</w:tblBorders>'
    )
    old = tblPr.find(qn("w:tblBorders"))
    if old is not None:
        tblPr.remove(old)
    tblPr.append(borders)


def set_cell_shading(cell, color_hex="EAEAEA"):
    shading = OxmlElement("w:shd")
    shading.set(qn("w:val"), "clear")
    shading.set(qn("w:color"), "auto")
    shading.set(qn("w:fill"), color_hex)
    cell._tc.get_or_add_tcPr().append(shading)


def set_cell_margins(cell, top=60, bottom=60, left=100, right=100):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(
        f'<w:tcMar {nsdecls("w")}>'
        f'<w:top w:w="{top}" w:type="dxa"/>'
        f'<w:bottom w:w="{bottom}" w:type="dxa"/>'
        f'<w:left w:w="{left}" w:type="dxa"/>'
        f'<w:right w:w="{right}" w:type="dxa"/>'
        f'</w:tcMar>'
    )
    tcPr.append(tcMar)


def format_run_tnr(run, size_pt=13, bold=False, italic=False, uppercase=False):
    run.font.name = "Times New Roman"
    run.font.size = Pt(size_pt)
    run.font.color.rgb = RGBColor(0, 0, 0)
    run.bold = bold
    run.italic = italic
    run.font.all_caps = uppercase
    rPr = run._r.get_or_add_rPr()
    rFonts = rPr.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = OxmlElement("w:rFonts")
        rPr.insert(0, rFonts)
    rFonts.set(qn("w:ascii"), "Times New Roman")
    rFonts.set(qn("w:hAnsi"), "Times New Roman")
    rFonts.set(qn("w:eastAsia"), "Times New Roman")
    rFonts.set(qn("w:cs"), "Times New Roman")


def add_seq_caption(doc, label_type, chapter_num, seq_index, title_text, keep_with_next=False):
    """
    Tạo chú thích chuẩn Mục 2.3.4: Style `Caption` (Times New Roman, 12pt, in nghiêng, căn giữa,
    Before 6pt, After 6pt, Single line spacing) kèm trường `SEQ` chuẩn của Microsoft Word.
    - Bảng biểu: `Bảng x.y Tên bảng` (đặt phía TRÊN bảng, keep_with_next=True).
    - Hình ảnh: `Hình x.y Tên hình` (đặt phía DƯỚI hình).
    """
    p = doc.add_paragraph(style="Caption")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keep_with_next = keep_with_next
    p.paragraph_format.keep_together = True

    r_lbl = p.add_run(f"{label_type} ")
    format_run_tnr(r_lbl, size_pt=12, bold=False, italic=True)

    # Trường STYLEREF 1 \s lấy số Chương hiện tại + trường SEQ <label_type> \* ARABIC \s 1 chuẩn Word
    for idx_f, (instr_str, disp_val) in enumerate([
        ("STYLEREF 1 \\s", str(chapter_num)),
        (f"SEQ {label_type} \\* ARABIC \\s 1", str(seq_index)),
    ]):
        if idx_f == 1:
            r_dot = p.add_run(".")
            format_run_tnr(r_dot, size_pt=12, bold=False, italic=True)
        p._p.append(parse_xml(f'<w:r {nsdecls("w")}><w:rPr><w:i/><w:sz w:val="24"/></w:rPr><w:fldChar w:fldCharType="begin"/></w:r>'))
        p._p.append(parse_xml(
            f'<w:r {nsdecls("w")}><w:rPr><w:i/><w:sz w:val="24"/></w:rPr>'
            f'<w:instrText xml:space="preserve"> {instr_str} </w:instrText></w:r>'
        ))
        p._p.append(parse_xml(f'<w:r {nsdecls("w")}><w:rPr><w:i/><w:sz w:val="24"/></w:rPr><w:fldChar w:fldCharType="separate"/></w:r>'))
        r_v = p.add_run(disp_val)
        format_run_tnr(r_v, size_pt=12, bold=False, italic=True)
        p._p.append(parse_xml(f'<w:r {nsdecls("w")}><w:rPr><w:i/><w:sz w:val="24"/></w:rPr><w:fldChar w:fldCharType="end"/></w:r>'))

    r2 = p.add_run(f" {title_text}")
    format_run_tnr(r2, size_pt=12, bold=False, italic=True)
    return p


def add_tlu_table(doc, chapter_num, table_index, caption_title, df, col_headers=None, font_size_pt=10.5):
    """
    Chèn bảng biểu chuẩn ĐH Thủy Lợi:
    - Chú thích `Bảng x.y ...` nằm PHÍA TRÊN bảng, dính liền với bảng (`keep_with_next=True`).
    - Dòng tiêu đề lặp lại nếu bảng dài qua trang (`w:tblHeader`), cấm cắt đôi dòng (`w:cantSplit`).
    - Độ tương phản in đen trắng sắc nét (chữ đen đậm Times New Roman 10-10.5pt trên nền tiêu đề xám sáng).
    """
    add_seq_caption(doc, "Bảng", chapter_num, table_index, caption_title, keep_with_next=True)

    headers = col_headers if col_headers is not None else list(df.columns)
    table = doc.add_table(rows=1 + len(df), cols=len(headers))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    # Dòng tiêu đề bảng
    hdr_row = table.rows[0]
    trPr_hdr = hdr_row._tr.get_or_add_trPr()
    trPr_hdr.append(OxmlElement("w:tblHeader"))
    trPr_hdr.append(OxmlElement("w:cantSplit"))

    for j, h_text in enumerate(headers):
        cell = hdr_row.cells[j]
        cell.text = ""
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        set_cell_shading(cell, "EAEAEA")
        set_cell_margins(cell, top=70, bottom=70, left=80, right=80)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.line_spacing = 1.1
        r = p.add_run(str(h_text))
        format_run_tnr(r, size_pt=font_size_pt, bold=True)

    # Các dòng dữ liệu
    for i, (_, row) in enumerate(df.iterrows()):
        tr = table.rows[i + 1]
        trPr = tr._tr.get_or_add_trPr()
        trPr.append(OxmlElement("w:cantSplit"))
        for j, val in enumerate(row):
            cell = tr.cells[j]
            cell.text = ""
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            set_cell_margins(cell, top=55, bottom=55, left=80, right=80)
            p = cell.paragraphs[0]
            # Cột đầu hoặc cột mô tả dài căn trái, các cột số căn giữa
            val_str = str(val)
            is_left = (j == 0) or (len(val_str) > 28)
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if is_left else WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(1.5)
            p.paragraph_format.space_after = Pt(1.5)
            p.paragraph_format.line_spacing = 1.15
            r = p.add_run(val_str)
            format_run_tnr(r, size_pt=font_size_pt, bold=False)

    return table


def add_tlu_figure(doc, image_path, chapter_num, fig_index, caption_title, width_cm=15.2):
    """
    Chèn hình ảnh chuẩn ĐH Thủy Lợi:
    - Đoạn chứa hình căn giữa, đặt `keep_with_next=True` để KHÔNG BAO GIỜ bị ngắt trang giữa hình và chú thích.
    - Chú thích `Hình x.y ...` đặt PHÍA DƯỚI hình, Style `Caption` (12pt, in nghiêng, căn giữa).
    """
    p_img = doc.add_paragraph()
    p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_img.paragraph_format.space_before = Pt(8)
    p_img.paragraph_format.space_after = Pt(2)
    p_img.paragraph_format.line_spacing = 1.0
    p_img.paragraph_format.keep_with_next = True
    p_img.paragraph_format.keep_together = True
    p_img.add_run().add_picture(image_path, width=Cm(width_cm))

    add_seq_caption(doc, "Hình", chapter_num, fig_index, caption_title, keep_with_next=False)


# =============================================================================
# BỘ XÂY DỰNG PHƯƠNG TRÌNH WORD EQUATION GỐC (<m:oMath>) CHUẨN MỤC 2.3.4
# =============================================================================

def omml_r(text, italic=True, bold=False):
    """Tạo phần tử <m:r> chuẩn Cambria Math trong Word Equation."""
    sty = "bi" if (bold and italic) else ("b" if bold else ("i" if italic else "p"))
    esc = (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    return (
        f'<m:r>'
        f'<m:rPr><m:sty m:val="{sty}"/></m:rPr>'
        f'<w:rPr><w:rFonts w:ascii="Cambria Math" w:hAnsi="Cambria Math" w:cs="Cambria Math"/><w:sz w:val="26"/></w:rPr>'
        f'<m:t xml:space="preserve">{esc}</m:t>'
        f'</m:r>'
    )


def omml_frac(num_xml, den_xml):
    return f"<m:f><m:num>{num_xml}</m:num><m:den>{den_xml}</m:den></m:f>"


def omml_sub(base_xml, sub_xml):
    return f"<m:sSub><m:e>{base_xml}</m:e><m:sub>{sub_xml}</m:sub></m:sSub>"


def omml_sup(base_xml, sup_xml):
    return f"<m:sSup><m:e>{base_xml}</m:e><m:sup>{sup_xml}</m:sup></m:sSup>"


def omml_subsup(base_xml, sub_xml, sup_xml):
    return f"<m:sSubSup><m:e>{base_xml}</m:e><m:sub>{sub_xml}</m:sub><m:sup>{sup_xml}</m:sup></m:sSubSup>"


def omml_sum(sub_xml, sup_xml, body_xml):
    sup_part = f"<m:sup>{sup_xml}</m:sup>" if sup_xml else "<m:sup/>"
    lim_loc = "undOvr"
    hide_sup = "" if sup_xml else '<m:supHide m:val="1"/>'
    return (
        f'<m:nary>'
        f'<m:naryPr><m:chr m:val="∑"/><m:limLoc m:val="{lim_loc}"/>{hide_sup}</m:naryPr>'
        f'<m:sub>{sub_xml}</m:sub>{sup_part}<m:e>{body_xml}</m:e>'
        f'</m:nary>'
    )


def omml_delim(inner_xml, beg="(", end=")"):
    return (
        f'<m:d>'
        f'<m:dPr><m:begChr m:val="{beg}"/><m:endChr m:val="{end}"/></m:dPr>'
        f'<m:e>{inner_xml}</m:e>'
        f'</m:d>'
    )


def omml_hat(base_xml):
    return f'<m:acc><m:accPr><m:chr m:val="̂"/></m:accPr><m:e>{base_xml}</m:e></m:acc>'


def add_tlu_equation(doc, omath_inner_xml, eq_number_str):
    """
    Chèn phương trình bằng đối tượng Word Equation (<m:oMath>) chính gốc:
    - Căn giữa phương trình tại vị trí 8,0 cm (4531 twips).
    - Đánh số phương trình dạng `(x-y)` nằm sát lề phải tại vị trí 16,0 cm (9062 twips) đúng Mục 2.3.4.
    - Giữ liền với đoạn giải thích `Trong đó:` ngay phía dưới (`keep_with_next=True`).
    """
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.keep_with_next = True
    p.paragraph_format.keep_together = True

    pPr = p._p.get_or_add_pPr()
    tabs = parse_xml(
        f'<w:tabs {nsdecls("w")}>'
        f'<w:tab w:val="center" w:pos="4531"/>'
        f'<w:tab w:val="right" w:pos="9062"/>'
        f'</w:tabs>'
    )
    pPr.append(tabs)

    # Tab thứ nhất đưa phương trình ra chính giữa trang
    p._p.append(parse_xml(f'<w:r {nsdecls("w")}><w:tab/></w:r>'))

    # Đối tượng <m:oMath> chuẩn Microsoft Word Equation
    omath_el = parse_xml(f'<m:oMath {nsdecls("w", "m")}>{omath_inner_xml}</m:oMath>')
    p._p.append(omath_el)

    # Tab thứ hai đưa số hiệu (x-y) sát lề phải
    p._p.append(parse_xml(f'<w:r {nsdecls("w")}><w:tab/></w:r>'))
    r_num = p.add_run(f"({eq_number_str})")
    format_run_tnr(r_num, size_pt=13, bold=False, italic=False)
    return p


def add_toc_field(doc, field_instr, fallback_lines):
    """
    Chèn trường mục lục động của Word (`TOC`) kèm danh sách hiển thị sẵn để xem đẹp ngay lập tức
    trên cả Microsoft Word, Google Docs và trình xem trước, đồng thời bấm Update Field (F9) tự động.
    """
    p = doc.add_paragraph(style="Normal")
    p.paragraph_format.line_spacing = 1.25
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(4)

    p._p.append(parse_xml(f'<w:r {nsdecls("w")}><w:fldChar w:fldCharType="begin"/></w:r>'))
    p._p.append(parse_xml(f'<w:r {nsdecls("w")}><w:instrText xml:space="preserve"> {field_instr} </w:instrText></w:r>'))
    p._p.append(parse_xml(f'<w:r {nsdecls("w")}><w:fldChar w:fldCharType="separate"/></w:r>'))

    for idx, (text_line, is_bold, indent_cm) in enumerate(fallback_lines):
        target_p = p if idx == 0 else doc.add_paragraph(style="Normal")
        target_p.paragraph_format.line_spacing = 1.25
        target_p.paragraph_format.space_before = Pt(2)
        target_p.paragraph_format.space_after = Pt(2)
        if indent_cm > 0:
            target_p.paragraph_format.left_indent = Cm(indent_cm)
        r = target_p.add_run(text_line)
        format_run_tnr(r, size_pt=13, bold=is_bold, italic=False)

    last_p = doc.paragraphs[-1]
    last_p._p.append(parse_xml(f'<w:r {nsdecls("w")}><w:fldChar w:fldCharType="end"/></w:r>'))


def inject_ieee_sources_xml(docx_path):
    """
    Ghi đè `customXml/item1.xml` bên trong gói `.docx` bằng đúng 9 nguồn tài liệu tham khảo IEEE
    của đề tài (thay thế hoàn toàn 10 nguồn mẫu tiếng Anh cũ trong file hướng dẫn của trường).
    """
    ieee_sources_xml = """<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<b:Sources SelectedStyle="\\IEEE2006OfficeOnline.xsl" StyleName="IEEE" Version="2006"
 xmlns:b="http://schemas.openxmlformats.org/officeDocument/2006/bibliography"
 xmlns="http://schemas.openxmlformats.org/officeDocument/2006/bibliography">
  <b:Source>
    <b:Tag>TQChieu25</b:Tag>
    <b:SourceType>Book</b:SourceType>
    <b:Guid>{11111111-2222-3333-4444-555555555501}</b:Guid>
    <b:Author><b:Author><b:NameList><b:Person><b:Last>Chiểu</b:Last><b:First>Tạ Quang</b:First></b:Person></b:NameList></b:Author></b:Author>
    <b:Title>Bài giảng và Tài liệu hướng dẫn thực hành môn Học máy (Machine Learning)</b:Title>
    <b:Year>2025</b:Year>
    <b:City>Hà Nội</b:City>
    <b:Publisher>Khoa Công nghệ thông tin, Trường Đại học Thủy Lợi</b:Publisher>
    <b:RefOrder>1</b:RefOrder>
  </b:Source>
  <b:Source>
    <b:Tag>VHTiep20</b:Tag>
    <b:SourceType>Book</b:SourceType>
    <b:Guid>{11111111-2222-3333-4444-555555555502}</b:Guid>
    <b:Author><b:Author><b:NameList><b:Person><b:Last>Tiệp</b:Last><b:First>Vũ Hữu</b:First></b:Person></b:NameList></b:Author></b:Author>
    <b:Title>Machine Learning cơ bản</b:Title>
    <b:Year>2020</b:Year>
    <b:City>Hà Nội</b:City>
    <b:Publisher>NXB Khoa học và Kỹ thuật</b:Publisher>
    <b:RefOrder>2</b:RefOrder>
  </b:Source>
  <b:Source>
    <b:Tag>OpenML14</b:Tag>
    <b:SourceType>JournalArticle</b:SourceType>
    <b:Guid>{11111111-2222-3333-4444-555555555503}</b:Guid>
    <b:Author><b:Author><b:NameList>
      <b:Person><b:Last>Vanschoren</b:Last><b:First>Joaquin</b:First></b:Person>
      <b:Person><b:Last>van Rijn</b:Last><b:First>Jan N.</b:First></b:Person>
      <b:Person><b:Last>Bischl</b:Last><b:First>Bernd</b:First></b:Person>
      <b:Person><b:Last>Torgo</b:Last><b:First>Luis</b:First></b:Person>
    </b:NameList></b:Author></b:Author>
    <b:Title>OpenML: networked science in machine learning (Credit Risk Dataset ID 43454)</b:Title>
    <b:JournalName>ACM SIGKDD Explorations Newsletter</b:JournalName>
    <b:Year>2014</b:Year>
    <b:Volume>15</b:Volume>
    <b:Issue>2</b:Issue>
    <b:Pages>49-60</b:Pages>
    <b:RefOrder>3</b:RefOrder>
  </b:Source>
  <b:Source>
    <b:Tag>Sklearn11</b:Tag>
    <b:SourceType>JournalArticle</b:SourceType>
    <b:Guid>{11111111-2222-3333-4444-555555555504}</b:Guid>
    <b:Author><b:Author><b:NameList>
      <b:Person><b:Last>Pedregosa</b:Last><b:First>Fabian</b:First></b:Person>
      <b:Person><b:Last>Varoquaux</b:Last><b:First>Gael</b:First></b:Person>
      <b:Person><b:Last>Gramfort</b:Last><b:First>Alexandre</b:First></b:Person>
    </b:NameList></b:Author></b:Author>
    <b:Title>Scikit-learn: Machine learning in Python</b:Title>
    <b:JournalName>Journal of Machine Learning Research</b:JournalName>
    <b:Year>2011</b:Year>
    <b:Volume>12</b:Volume>
    <b:Pages>2825-2830</b:Pages>
    <b:RefOrder>4</b:RefOrder>
  </b:Source>
  <b:Source>
    <b:Tag>DCox58</b:Tag>
    <b:SourceType>JournalArticle</b:SourceType>
    <b:Guid>{11111111-2222-3333-4444-555555555505}</b:Guid>
    <b:Author><b:Author><b:NameList><b:Person><b:Last>Cox</b:Last><b:First>D. R.</b:First></b:Person></b:NameList></b:Author></b:Author>
    <b:Title>The regression analysis of binary sequences</b:Title>
    <b:JournalName>Journal of the Royal Statistical Society: Series B (Methodological)</b:JournalName>
    <b:Year>1958</b:Year>
    <b:Volume>20</b:Volume>
    <b:Issue>2</b:Issue>
    <b:Pages>215-232</b:Pages>
    <b:RefOrder>5</b:RefOrder>
  </b:Source>
  <b:Source>
    <b:Tag>JRQuinlan86</b:Tag>
    <b:SourceType>JournalArticle</b:SourceType>
    <b:Guid>{11111111-2222-3333-4444-555555555506}</b:Guid>
    <b:Author><b:Author><b:NameList><b:Person><b:Last>Quinlan</b:Last><b:First>J. R.</b:First></b:Person></b:NameList></b:Author></b:Author>
    <b:Title>Induction of decision trees</b:Title>
    <b:JournalName>Machine Learning</b:JournalName>
    <b:Year>1986</b:Year>
    <b:Volume>1</b:Volume>
    <b:Issue>1</b:Issue>
    <b:Pages>81-106</b:Pages>
    <b:RefOrder>6</b:RefOrder>
  </b:Source>
  <b:Source>
    <b:Tag>LBreiman84</b:Tag>
    <b:SourceType>Book</b:SourceType>
    <b:Guid>{11111111-2222-3333-4444-555555555507}</b:Guid>
    <b:Author><b:Author><b:NameList>
      <b:Person><b:Last>Breiman</b:Last><b:First>Leo</b:First></b:Person>
      <b:Person><b:Last>Friedman</b:Last><b:First>Jerome</b:First></b:Person>
      <b:Person><b:Last>Stone</b:Last><b:First>Charles J.</b:First></b:Person>
      <b:Person><b:Last>Olshen</b:Last><b:First>R. A.</b:First></b:Person>
    </b:NameList></b:Author></b:Author>
    <b:Title>Classification and Regression Trees</b:Title>
    <b:Year>1984</b:Year>
    <b:City>Belmont, CA, USA</b:City>
    <b:Publisher>Wadsworth International Group</b:Publisher>
    <b:RefOrder>7</b:RefOrder>
  </b:Source>
  <b:Source>
    <b:Tag>LBreiman01</b:Tag>
    <b:SourceType>JournalArticle</b:SourceType>
    <b:Guid>{11111111-2222-3333-4444-555555555508}</b:Guid>
    <b:Author><b:Author><b:NameList><b:Person><b:Last>Breiman</b:Last><b:First>Leo</b:First></b:Person></b:NameList></b:Author></b:Author>
    <b:Title>Random forests</b:Title>
    <b:JournalName>Machine Learning</b:JournalName>
    <b:Year>2001</b:Year>
    <b:Volume>45</b:Volume>
    <b:Issue>1</b:Issue>
    <b:Pages>5-32</b:Pages>
    <b:RefOrder>8</b:RefOrder>
  </b:Source>
  <b:Source>
    <b:Tag>TFawcett06</b:Tag>
    <b:SourceType>JournalArticle</b:SourceType>
    <b:Guid>{11111111-2222-3333-4444-555555555509}</b:Guid>
    <b:Author><b:Author><b:NameList><b:Person><b:Last>Fawcett</b:Last><b:First>Tom</b:First></b:Person></b:NameList></b:Author></b:Author>
    <b:Title>An introduction to ROC analysis</b:Title>
    <b:JournalName>Pattern Recognition Letters</b:JournalName>
    <b:Year>2006</b:Year>
    <b:Volume>27</b:Volume>
    <b:Issue>8</b:Issue>
    <b:Pages>861-874</b:Pages>
    <b:RefOrder>9</b:RefOrder>
  </b:Source>
</b:Sources>"""

    temp_path = docx_path + ".tmp"
    with zipfile.ZipFile(docx_path, "r") as zin, zipfile.ZipFile(temp_path, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "customXml/item1.xml":
                data = ieee_sources_xml.encode("utf-8")
            zout.writestr(item, data)
    import os
    os.replace(temp_path, docx_path)
