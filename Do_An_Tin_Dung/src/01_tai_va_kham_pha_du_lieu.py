"""
CHƯƠNG I: MÔ TẢ BÀI TOÁN VÀ PHÂN TÍCH KHÁM PHÁ DỮ LIỆU THÔ (EDA)
Đồ án Học máy - Đề tài: Dự đoán khả năng trả nợ của khách hàng vay vốn
Thành viên phụ trách chính: Hải Nam
Giảng viên hướng dẫn: Thầy Chiểu - Khoa CNTT, Trường Đại học Thủy Lợi
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

# Đảm bảo hiển thị tiếng Việt có dấu không lỗi trên Windows Terminal / PowerShell
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# Thiết lập đường dẫn tương đối tính từ thư mục gốc của đồ án (Do_An_Tin_Dung/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "credit_risk_raw.csv")
FIGURES_DIR = os.path.join(BASE_DIR, "outputs", "figures")
TABLES_DIR = os.path.join(BASE_DIR, "outputs", "tables")

# Danh sách phân nhóm các cột đặc trưng theo đúng bản chất nghiệp vụ tín dụng
NUM_COLS = [
    "person_age",                  # Tuổi khách hàng (năm)
    "person_income",               # Thu nhập hàng năm (USD)
    "person_emp_length",           # Thâm niên làm việc (năm)
    "loan_amnt",                   # Số tiền xin vay (USD)
    "loan_int_rate",               # Lãi suất khoản vay (%/năm)
    "loan_percent_income",         # Tỷ lệ khoản vay / thu nhập hàng năm
    "cb_person_cred_hist_length",  # Độ dài lịch sử tín dụng (năm)
]

CAT_COLS = [
    "person_home_ownership",       # Tình trạng sở hữu nhà ở (RENT, OWN, MORTGAGE, OTHER)
    "loan_intent",                 # Mục đích vay vốn (EDUCATION, MEDICAL, VENTURE...)
    "loan_grade",                  # Hạng tín dụng khoản vay (A -> G, biến thứ tự)
    "cb_person_default_on_file",   # Lịch sử từng vỡ nợ trước đây (Y / N)
]

TARGET_COL = "loan_status"         # Biến mục tiêu: 0 = Trả nợ tốt, 1 = Vỡ nợ

FEATURE_DESCRIPTIONS_VI = {
    "person_age": ("Biến số liên tục", "Tuổi của khách hàng tại thời điểm nộp hồ sơ vay (năm)"),
    "person_income": ("Biến số liên tục", "Tổng thu nhập hàng năm của khách hàng (USD)"),
    "person_home_ownership": ("Biến phân loại (Định danh)", "Tình trạng nhà ở: RENT (Thuê), MORTGAGE (Thế chấp), OWN (Sở hữu), OTHER (Khác)"),
    "person_emp_length": ("Biến số liên tục", "Số năm kinh nghiệm/thâm niên làm việc của khách hàng (năm)"),
    "loan_intent": ("Biến phân loại (Định danh)", "Mục đích vay: EDUCATION, MEDICAL, VENTURE, PERSONAL, DEBTCONSOLIDATION, HOMEIMPROVEMENT"),
    "loan_grade": ("Biến phân loại (Thứ tự)", "Hạng tín dụng do tổ chức tài chính xếp loại từ A (tốt nhất) đến G (rủi ro cao nhất)"),
    "loan_amnt": ("Biến số liên tục", "Số tiền gốc khách hàng đăng ký vay (USD)"),
    "loan_int_rate": ("Biến số liên tục", "Lãi suất áp dụng cho khoản vay (%/năm)"),
    "loan_percent_income": ("Biến số liên tục", "Tỷ lệ giữa số tiền vay trên thu nhập hàng năm của khách hàng"),
    "cb_person_default_on_file": ("Biến phân loại (Nhị phân)", "Khách hàng đã từng bị ghi nhận vỡ nợ trong lịch sử tín dụng chưa (Y = Có, N = Chưa)"),
    "cb_person_cred_hist_length": ("Biến số liên tục", "Số năm khách hàng có lịch sử giao dịch tín dụng trên hệ thống (năm)"),
    "loan_status": ("Biến mục tiêu (Nhị phân)", "Trạng thái khoản vay: 0 = Trả nợ tốt (Non-default), 1 = Vỡ nợ (Default)"),
}


def tai_du_lieu_tho(raw_csv_path=RAW_DATA_PATH):
    """
    GIẢI THÍCH 5 ĐIỂM QUAN TRỌNG:
    1. Đoạn code thực hiện việc gì?
       Tải tự động bộ dữ liệu Credit Risk Dataset (OpenML ID = 43454) về máy và lưu thành
       file CSV cục bộ. Nếu file CSV đã tồn tại thì đọc trực tiếp để tiết kiệm thời gian.
    2. Tại sao cần thực hiện bước này?
       Đảm bảo tính tái lập (reproducibility) 100%: cả 3 thành viên Hải Nam, Thành, Vũ
       khi pull code từ GitHub đều tự động có đúng cùng một bộ dữ liệu gốc chuẩn xác.
    3. Dữ liệu đầu vào (Input) là gì?
       Đường dẫn lưu file dữ liệu thô `raw_csv_path`.
    4. Kết quả đầu ra (Output) là gì?
       DataFrame `df` chứa toàn bộ dữ liệu thô (32.581 dòng x 12 cột).
    5. Hải Nam, Thành, Vũ cần hiểu cốt lõi điều gì để trả lời thầy Chiểu?
       Biến mục tiêu `loan_status` được quy ước chuẩn bài toán quản trị rủi ro:
       Nhãn 0 = Trả nợ tốt (lớp Negative), Nhãn 1 = Vỡ nợ (lớp Positive cần phát hiện).
    """
    os.makedirs(os.path.dirname(raw_csv_path), exist_ok=True)

    if os.path.exists(raw_csv_path):
        print(f"[INFO] Đọc dữ liệu thô đã lưu tại: {raw_csv_path}")
        df = pd.read_csv(raw_csv_path)
    else:
        print("[INFO] Đang tải tự động Credit Risk Dataset (OpenML ID = 43454)...")
        dataset = fetch_openml(data_id=43454, as_frame=True, parser="auto")
        df = dataset.frame.copy()

        # Đồng bộ kiểu dữ liệu chuẩn trước khi lưu CSV lần đầu
        for col in NUM_COLS:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        for col in CAT_COLS:
            df[col] = df[col].astype("object")
        df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="raise").astype(int)

        # Sắp xếp thứ tự cột gọn gàng: 7 biến số + 4 biến phân loại + 1 biến mục tiêu
        ordered_cols = NUM_COLS + CAT_COLS + [TARGET_COL]
        df = df[ordered_cols]
        df.to_csv(raw_csv_path, index=False)
        print(f"[INFO] Đã lưu dữ liệu thô tại: {raw_csv_path}")

    return df


def phan_tich_chat_luong_du_lieu(df, tables_dir=TABLES_DIR):
    """
    GIẢI THÍCH 5 ĐIỂM QUAN TRỌNG:
    1. Đoạn code thực hiện việc gì?
       Thống kê toàn diện quy mô, phân bố nhãn 0/1, số dòng trùng lặp, tỷ lệ dữ liệu khuyết
       (missing values), thống kê mô tả và phát hiện các điểm ngoại lai vô lý về mặt nghiệp vụ.
    2. Tại sao cần thực hiện bước này?
       Không hiểu dữ liệu thô thì không thể chọn đúng kỹ thuật tiền xử lý ở Chương II.
       Phát hiện sớm dữ liệu khuyết và ngoại lai giúp tránh đưa rác vào mô hình (Garbage In, Garbage Out).
    3. Dữ liệu đầu vào (Input) là gì?
       DataFrame `df` dữ liệu thô và thư mục xuất bảng `tables_dir`.
    4. Kết quả đầu ra (Output) là gì?
       3 file bảng `.csv` phục vụ trực tiếp cho Chương I báo cáo và thông tin in ra màn hình.
    5. Hải Nam, Thành, Vũ cần hiểu cốt lõi điều gì để trả lời thầy Chiểu?
       - Dữ liệu bị mất cân bằng lớp vừa phải (~78.18% trả tốt vs ~21.82% vỡ nợ -> tỷ lệ ~3.58:1),
         do đó ở Chương III không được chỉ nhìn vào Accuracy mà phải xét Precision, Recall, F1, ROC-AUC.
       - Phát hiện ngoại lai vô lý (VD: tuổi = 144, số năm đi làm = 123) và khuyết ở 2 cột số
         (`person_emp_length`, `loan_int_rate`) là căn cứ thực tế cho Chương II.
    """
    os.makedirs(tables_dir, exist_ok=True)
    n_rows, n_cols = df.shape

    print("\n" + "=" * 78)
    print("CHƯƠNG I: KẾT QUẢ KHÁM PHÁ DỮ LIỆU THÔ (EXPLORATORY DATA ANALYSIS - EDA)")
    print("=" * 78)
    print(f"1. Quy mô bộ dữ liệu gốc : {n_rows:,} dòng x {n_cols} cột (11 đặc trưng + 1 mục tiêu)")

    # Phân bố biến mục tiêu
    target_counts = df[TARGET_COL].value_counts().sort_index()
    target_pcts = df[TARGET_COL].value_counts(normalize=True).sort_index() * 100
    imbalance_ratio = target_counts[0] / target_counts[1]

    print("2. Phân bố biến mục tiêu `loan_status`:")
    print(f"   - Nhãn 0 (Trả nợ tốt - Non-default): {target_counts[0]:,} mẫu ({target_pcts[0]:.2f}%)")
    print(f"   - Nhãn 1 (Vỡ nợ      - Default)    : {target_counts[1]:,} mẫu ({target_pcts[1]:.2f}%)")
    print(f"   - Tỷ lệ mất cân bằng lớp (0 : 1)    : {imbalance_ratio:.2f} : 1")

    # Kiểm tra trùng lặp & ngoại lai vô lý
    n_dup = int(df.duplicated().sum())
    mask_age_outlier = df["person_age"] > 100
    mask_emp_outlier = (df["person_emp_length"] > 60) | (df["person_emp_length"] > (df["person_age"] - 14))

    print("\n3. Kiểm tra trùng lặp & ngoại lai vô lý theo nghiệp vụ tín dụng:")
    print(f"   - Số bản ghi trùng lặp hoàn toàn                : {n_dup} dòng")
    print(f"   - Số bản ghi có tuổi > 100 (vô lý)              : {int(mask_age_outlier.sum())} dòng "
          f"(Giá trị: {sorted(df.loc[mask_age_outlier, 'person_age'].tolist())})")
    print(f"   - Số bản ghi có thâm niên làm việc bất hợp lý   : {int(mask_emp_outlier.sum())} dòng "
          f"(Max = {df['person_emp_length'].max()} năm)")

    # Xuất Bảng 1.1: Tổng quan & chất lượng 12 cột
    quality_rows = []
    for col in df.columns:
        loai_bien, mo_ta = FEATURE_DESCRIPTIONS_VI[col]
        n_miss = int(df[col].isna().sum())
        pct_miss = (n_miss / n_rows) * 100
        quality_rows.append({
            "Ten_Cot": col,
            "Phan_Loai_Bien": loai_bien,
            "So_Gia_Tri_Thieu": n_miss,
            "Ty_Le_Thieu_Pct": round(pct_miss, 2),
            "So_Gia_Tri_Duy_Nhat": int(df[col].nunique(dropna=True)),
            "Y_Nghia_Thuc_Te": mo_ta,
        })
    df_quality = pd.DataFrame(quality_rows)
    quality_csv = os.path.join(tables_dir, "01_kiem_tra_chat_luong_du_lieu.csv")
    df_quality.to_csv(quality_csv, index=False, encoding="utf-8-sig")

    print("\n4. Tình trạng dữ liệu thiếu (Missing Values):")
    for _, row in df_quality[df_quality["So_Gia_Tri_Thieu"] > 0].iterrows():
        print(f"   - Cột `{row['Ten_Cot']:20s}`: thiếu {row['So_Gia_Tri_Thieu']:,} dòng ({row['Ty_Le_Thieu_Pct']:.2f}%)")

    # Xuất Bảng 1.2: Thống kê mô tả 7 biến số liên tục
    desc_num = df[NUM_COLS].describe().T
    desc_num["median"] = df[NUM_COLS].median()
    desc_num["skewness"] = df[NUM_COLS].skew()
    desc_num = desc_num[["count", "mean", "std", "min", "25%", "median", "75%", "max", "skewness"]].round(2)
    num_csv = os.path.join(tables_dir, "01_thong_ke_mo_ta_so.csv")
    desc_num.to_csv(num_csv, encoding="utf-8-sig")

    print("\n5. Thống kê mô tả các biến số liên tục:")
    print(desc_num.to_string())

    # Xuất Bảng 1.3: Thống kê các biến phân loại và tỷ lệ vỡ nợ theo từng nhóm
    cat_summary_rows = []
    for col in CAT_COLS:
        grp = df.groupby(col, dropna=False)[TARGET_COL].agg(
            So_Luong="count",
            So_Vo_No="sum",
            Ty_Le_Vo_No_Pct=lambda s: round(s.mean() * 100, 2),
        ).reset_index()
        grp["Ten_Bien"] = col
        grp = grp.rename(columns={col: "Gia_Tri_Nhom"})
        grp["Ty_Trong_Mau_Pct"] = (grp["So_Luong"] / n_rows * 100).round(2)
        cat_summary_rows.append(grp[["Ten_Bien", "Gia_Tri_Nhom", "So_Luong", "Ty_Trong_Mau_Pct", "So_Vo_No", "Ty_Le_Vo_No_Pct"]])

    df_cat_summary = pd.concat(cat_summary_rows, ignore_index=True)
    cat_csv = os.path.join(tables_dir, "01_thong_ke_phan_loai.csv")
    df_cat_summary.to_csv(cat_csv, index=False, encoding="utf-8-sig")

    print(f"\n[OK] Đã xuất 3 bảng thống kê Chương I tại: {tables_dir}")
    return df_quality, desc_num, df_cat_summary


def ve_bieu_do_eda(df, figures_dir=FIGURES_DIR):
    """
    GIẢI THÍCH 5 ĐIỂM QUAN TRỌNG:
    1. Đoạn code thực hiện việc gì?
       Trực quan hóa dữ liệu thô thành 4 biểu đồ học thuật (.png) lưu trong `outputs/figures/`.
    2. Tại sao cần thực hiện bước này?
       Biểu đồ giúp nhìn thấy ngay: (a) mức độ mất cân bằng nhãn, (b) phân phối lệch & ngoại lai
       của biến số, (c) quy luật rủi ro tăng dần theo hạng tín dụng A->G, (d) tương quan tuyến tính.
    3. Dữ liệu đầu vào (Input) là gì?
       DataFrame `df` dữ liệu thô và thư mục lưu ảnh `figures_dir`.
    4. Kết quả đầu ra (Output) là gì?
       4 file ảnh PNG độ phân giải cao (dpi=180) phục vụ chèn trực tiếp vào Chương I báo cáo.
    5. Hải Nam, Thành, Vũ cần hiểu cốt lõi điều gì để trả lời thầy Chiểu?
       Nhìn Hình 1.2 (Boxplot/Histogram): biến thu nhập lệch phải mạnh và có điểm ngoại lai cực lớn,
       đó là bằng chứng trực quan vì sao ở Chương II ta chọn điền khuyết bằng Trung vị (Median)
       chứ KHÔNG dùng Trung bình cộng (Mean)!
    """
    os.makedirs(figures_dir, exist_ok=True)
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Segoe UI"]
    plt.rcParams["axes.unicode_minus"] = False

    # -----------------------------------------------------------------
    # Hình 1.1: Phân bố biến mục tiêu loan_status
    # -----------------------------------------------------------------
    counts = df[TARGET_COL].value_counts().sort_index()
    labels = ["0 - Trả nợ tốt\n(Non-default)", "1 - Vỡ nợ\n(Default)"]
    colors = ["#2b6cb0", "#e53e3e"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    bars = axes[0].bar(labels, counts.values, color=colors, width=0.52, edgecolor="black", alpha=0.9)
    for bar, cnt in zip(bars, counts.values):
        pct = cnt / len(df) * 100
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            cnt + 450,
            f"{cnt:,} mẫu\n({pct:.2f}%)",
            ha="center", va="bottom", fontsize=10, fontweight="bold"
        )
    axes[0].set_ylim(0, max(counts.values) * 1.18)
    axes[0].set_ylabel("Số lượng khách hàng")
    axes[0].set_title("Số lượng mẫu theo trạng thái khoản vay")
    axes[0].grid(axis="y", linestyle="--", alpha=0.35)

    axes[1].pie(
        counts.values,
        labels=["Trả nợ tốt (0)", "Vỡ nợ (1)"],
        autopct="%1.2f%%",
        startangle=90,
        colors=colors,
        explode=(0, 0.06),
        wedgeprops={"edgecolor": "black", "linewidth": 1},
        textprops={"fontsize": 10.5, "fontweight": "bold"},
    )
    axes[1].set_title("Tỷ trọng phân bố hai lớp (78.18% vs 21.82%)")

    fig.suptitle("Hình 1.1: Phân bố biến mục tiêu loan_status trên tập dữ liệu thô (N = 32,581)", fontsize=12.5, fontweight="bold")
    plt.tight_layout()
    fig1_path = os.path.join(figures_dir, "01_phan_bo_nhan_loan_status.png")
    plt.savefig(fig1_path, dpi=180)
    plt.close()

    # -----------------------------------------------------------------
    # Hình 1.2: Phân phối & Ngoại lai của 4 biến số tiêu biểu
    # -----------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.2))
    # (a) Boxplot tuổi làm nổi bật ngoại lai 123, 144 tuổi
    axes[0, 0].boxplot(df["person_age"].dropna(), orientation="horizontal", patch_artist=True,
                       boxprops=dict(facecolor="#90cdf4", edgecolor="black"))
    axes[0, 0].set_title("(a) Boxplot Tuổi khách hàng (ngoại lai > 100 tuổi)", fontsize=10.5)
    axes[0, 0].set_xlabel("Tuổi (năm)")
    axes[0, 0].grid(True, linestyle="--", alpha=0.35)

    # (b) Boxplot thâm niên làm việc làm nổi bật ngoại lai 123 năm
    axes[0, 1].boxplot(df["person_emp_length"].dropna(), orientation="horizontal", patch_artist=True,
                       boxprops=dict(facecolor="#fbd38d", edgecolor="black"))
    axes[0, 1].set_title("(b) Boxplot Thâm niên làm việc (ngoại lai = 123 năm)", fontsize=10.5)
    axes[0, 1].set_xlabel("Số năm làm việc")
    axes[0, 1].grid(True, linestyle="--", alpha=0.35)

    # (c) Histogram thu nhập (cắt hiển thị <= 250,000 USD để thấy rõ độ lệch phải + đường Mean vs Median)
    inc_clip = df.loc[df["person_income"] <= 250000, "person_income"]
    axes[1, 0].hist(inc_clip, bins=40, color="#68d391", edgecolor="black", alpha=0.85)
    axes[1, 0].axvline(df["person_income"].mean(), color="#e53e3e", linestyle="--", lw=2,
                       label=f"Mean = {df['person_income'].mean():,.0f} USD")
    axes[1, 0].axvline(df["person_income"].median(), color="#1a365d", linestyle="-", lw=2,
                       label=f"Median = {df['person_income'].median():,.0f} USD")
    axes[1, 0].set_title("(c) Phân phối Thu nhập (<= 250k USD) bị lệch phải mạnh", fontsize=10.5)
    axes[1, 0].set_xlabel("Thu nhập hàng năm (USD)")
    axes[1, 0].set_ylabel("Số khách hàng")
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(True, linestyle="--", alpha=0.35)

    # (d) Histogram lãi suất vay (cột có 9.56% dữ liệu thiếu)
    axes[1, 1].hist(df["loan_int_rate"].dropna(), bins=30, color="#b794f4", edgecolor="black", alpha=0.85)
    axes[1, 1].axvline(df["loan_int_rate"].mean(), color="#e53e3e", linestyle="--", lw=2,
                       label=f"Mean = {df['loan_int_rate'].mean():.2f}%")
    axes[1, 1].axvline(df["loan_int_rate"].median(), color="#1a365d", linestyle="-", lw=2,
                       label=f"Median = {df['loan_int_rate'].median():.2f}%")
    axes[1, 1].set_title("(d) Phân phối Lãi suất vay (thiếu 3,116 mẫu ~ 9.56%)", fontsize=10.5)
    axes[1, 1].set_xlabel("Lãi suất (%/năm)")
    axes[1, 1].set_ylabel("Số khách hàng")
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(True, linestyle="--", alpha=0.35)

    fig.suptitle("Hình 1.2: Khám phá phân phối biến số & phát hiện ngoại lai trên dữ liệu thô", fontsize=12.5, fontweight="bold")
    plt.tight_layout()
    fig2_path = os.path.join(figures_dir, "01_phan_bo_bien_so_va_ngoai_lai.png")
    plt.savefig(fig2_path, dpi=180)
    plt.close()

    # -----------------------------------------------------------------
    # Hình 1.3: Tỷ lệ vỡ nợ (%) theo các biến phân loại quan trọng
    # -----------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))

    # (a) Theo hạng tín dụng loan_grade (A -> G)
    grade_rates = df.groupby("loan_grade")[TARGET_COL].mean().reindex(list("ABCDEFG")) * 100
    b1 = axes[0].bar(grade_rates.index, grade_rates.values, color="#dd6b20", edgecolor="black", alpha=0.9)
    for bar, val in zip(b1, grade_rates.values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, val + 1.5, f"{val:.1f}%", ha="center", fontsize=8.5, fontweight="bold")
    axes[0].set_ylim(0, 110)
    axes[0].set_title("Tỷ lệ vỡ nợ theo Hạng khoản vay (loan_grade)", fontsize=10.5)
    axes[0].set_xlabel("Hạng tín dụng (A tốt nhất -> G rủi ro nhất)")
    axes[0].set_ylabel("Tỷ lệ vỡ nợ (%)")
    axes[0].grid(axis="y", linestyle="--", alpha=0.35)

    # (b) Theo tình trạng sở hữu nhà ở
    home_rates = df.groupby("person_home_ownership")[TARGET_COL].mean().sort_values(ascending=False) * 100
    b2 = axes[1].bar(home_rates.index, home_rates.values, color="#3182ce", edgecolor="black", alpha=0.9)
    for bar, val in zip(b2, home_rates.values):
        axes[1].text(bar.get_x() + bar.get_width() / 2, val + 1.0, f"{val:.1f}%", ha="center", fontsize=9, fontweight="bold")
    axes[1].set_ylim(0, max(home_rates.values) * 1.25)
    axes[1].set_title("Tỷ lệ vỡ nợ theo Nhà ở (person_home_ownership)", fontsize=10.5)
    axes[1].set_xlabel("Hình thức sở hữu nhà")
    axes[1].set_ylabel("Tỷ lệ vỡ nợ (%)")
    axes[1].grid(axis="y", linestyle="--", alpha=0.35)

    # (c) Theo lịch sử từng vỡ nợ
    def_rates = df.groupby("cb_person_default_on_file")[TARGET_COL].mean().reindex(["N", "Y"]) * 100
    b3 = axes[2].bar(["Chưa từng (N)", "Đã từng vỡ nợ (Y)"], def_rates.values,
                     color=["#38a169", "#e53e3e"], width=0.5, edgecolor="black", alpha=0.9)
    for bar, val in zip(b3, def_rates.values):
        axes[2].text(bar.get_x() + bar.get_width() / 2, val + 1.0, f"{val:.1f}%", ha="center", fontsize=9.5, fontweight="bold")
    axes[2].set_ylim(0, max(def_rates.values) * 1.25)
    axes[2].set_title("Tỷ lệ vỡ nợ theo Lịch sử nợ xấu (cb_default)", fontsize=10.5)
    axes[2].set_xlabel("Lịch sử tín dụng quá khứ")
    axes[2].set_ylabel("Tỷ lệ vỡ nợ (%)")
    axes[2].grid(axis="y", linestyle="--", alpha=0.35)

    fig.suptitle("Hình 1.3: Mối liên hệ giữa các đặc trưng phân loại và tỷ lệ vỡ nợ thực tế", fontsize=12.5, fontweight="bold")
    plt.tight_layout()
    fig3_path = os.path.join(figures_dir, "01_ty_le_vo_no_theo_nhom.png")
    plt.savefig(fig3_path, dpi=180)
    plt.close()

    # -----------------------------------------------------------------
    # Hình 1.4: Ma trận tương quan Pearson giữa các biến số & loan_status
    # -----------------------------------------------------------------
    corr_cols = NUM_COLS + [TARGET_COL]
    short_names = [
        "person_age",
        "person_income",
        "emp_length",
        "loan_amnt",
        "loan_int_rate",
        "loan_pct_inc",
        "cred_hist_len",
        "loan_status (y)",
    ]
    corr_mat = df[corr_cols].corr()

    fig, ax = plt.subplots(figsize=(9, 7.2))
    im = ax.imshow(corr_mat.values, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Hệ số tương quan Pearson (r)")

    ax.set_xticks(range(len(short_names)))
    ax.set_yticks(range(len(short_names)))
    ax.set_xticklabels(short_names, rotation=35, ha="right", fontsize=9.5)
    ax.set_yticklabels(short_names, fontsize=9.5)

    for i in range(len(short_names)):
        for j in range(len(short_names)):
            val = corr_mat.values[i, j]
            text_color = "white" if abs(val) >= 0.55 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=9, fontweight="bold")

    ax.set_title("Hình 1.4: Ma trận tương quan Pearson giữa các biến số và trạng thái vỡ nợ", fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    fig4_path = os.path.join(figures_dir, "01_ma_tran_tuong_quan.png")
    plt.savefig(fig4_path, dpi=180)
    plt.close()

    print(f"[OK] Đã xuất 4 biểu đồ EDA Chương I tại: {figures_dir}")


if __name__ == "__main__":
    df_raw = tai_du_lieu_tho()
    phan_tich_chat_luong_du_lieu(df_raw)
    ve_bieu_do_eda(df_raw)
