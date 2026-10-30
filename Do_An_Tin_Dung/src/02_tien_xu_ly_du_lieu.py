"""
CHƯƠNG II: TIỀN XỬ LÝ DỮ LIỆU CHUẨN HỌC THUẬT (CHỐNG TUYỆT ĐỐI DATA LEAKAGE)
Đồ án Học máy - Đề tài: Dự đoán khả năng trả nợ của khách hàng vay vốn
Thành viên phụ trách chính: Hải Nam
Giảng viên hướng dẫn: Thầy Chiểu - Khoa CNTT, Trường Đại học Thủy Lợi
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Đảm bảo hiển thị tiếng Việt có dấu trên Windows Terminal / PowerShell
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "credit_risk_raw.csv")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
FIGURES_DIR = os.path.join(BASE_DIR, "outputs", "figures")
TABLES_DIR = os.path.join(BASE_DIR, "outputs", "tables")
MODELS_DIR = os.path.join(BASE_DIR, "outputs", "models")

NUM_COLS = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
]
ORDINAL_COL = "loan_grade"
BINARY_COL = "cb_person_default_on_file"
NOMINAL_COLS = ["person_home_ownership", "loan_intent"]
CAT_COLS = [NOMINAL_COLS[0], NOMINAL_COLS[1], ORDINAL_COL, BINARY_COL]
TARGET_COL = "loan_status"

GRADE_MAPPING = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6}
BINARY_MAPPING = {"N": 0, "Y": 1}
BIN_LABELS = ["1_Thap", "2_Trung_binh", "3_Kha_cao", "4_Cao"]


def lam_sach_du_lieu_tho(df):
    """
    GIẢI THÍCH 5 ĐIỂM QUAN TRỌNG:
    1. Đoạn code thực hiện việc gì?
       Xóa các dòng trùng lặp hoàn toàn và lọc bỏ các bản ghi có giá trị ngoại lai vô lý
       về mặt logic sinh học / nghiệp vụ ngân hàng (tuổi > 100, thâm niên làm việc > 60 năm
       hoặc đi làm trước 14 tuổi).
    2. Tại sao cần thực hiện bước này?
       - Bản ghi trùng lặp nếu vừa rơi vào Train vừa rơi vào Test sẽ gây gian lận đánh giá.
       - Ngoại lai nhập sai (như 144 tuổi, 123 năm đi làm) làm méo đường biên phân lớp của
         Logistic Regression và kéo lệch trung bình/độ lệch chuẩn.
    3. Dữ liệu đầu vào (Input) là gì?
       DataFrame `df` dữ liệu thô (32.581 dòng).
    4. Kết quả đầu ra (Output) là gì?
       DataFrame `df_clean` đã loại bỏ bản ghi trùng & ngoại lai vô lý, kèm thống kê số dòng bị lọc.
    5. Hải Nam, Thành, Vũ cần hiểu cốt lõi điều gì để trả lời thầy Chiểu?
       Quy tắc lọc ở đây chỉ dựa trên logic thực tế của TỪNG DÒNG ĐỘC LẬP (tuổi <= 100, năm đi làm <= 60),
       KHÔNG dùng trung bình hay tứ phân vị của toàn tập nên tuyệt đối KHÔNG gây Data Leakage!
       Lưu ý: Các dòng đang thiếu (`NaN`) ở `person_emp_length` vẫn được GIỮ LẠI để điền khuyết ở bước sau.
    """
    n_initial = len(df)

    # Bước 1: Xóa dòng trùng lặp hoàn toàn
    df_no_dup = df.drop_duplicates().reset_index(drop=True)
    n_dup_removed = n_initial - len(df_no_dup)

    # Bước 2: Lọc ngoại lai vô lý do lỗi nhập liệu (giữ lại NaN để Impute sau)
    valid_age = (df_no_dup["person_age"] >= 18) & (df_no_dup["person_age"] <= 100)
    valid_emp = df_no_dup["person_emp_length"].isna() | (
        (df_no_dup["person_emp_length"] >= 0)
        & (df_no_dup["person_emp_length"] <= 60)
        & (df_no_dup["person_emp_length"] <= (df_no_dup["person_age"] - 14))
    )
    valid_positive = (df_no_dup["person_income"] > 0) & (df_no_dup["loan_amnt"] > 0)

    mask_valid = valid_age & valid_emp & valid_positive
    df_clean = df_no_dup.loc[mask_valid].reset_index(drop=True)
    n_outlier_removed = len(df_no_dup) - len(df_clean)

    cleaning_stats = {
        "n_initial": n_initial,
        "n_dup_removed": n_dup_removed,
        "n_outlier_removed": n_outlier_removed,
        "n_clean": len(df_clean),
    }
    return df_clean, cleaning_stats


def chia_tap_train_test_phan_tang(df_clean):
    """
    GIẢI THÍCH 5 ĐIỂM QUAN TRỌNG:
    1. Đoạn code thực hiện việc gì?
       Tách dữ liệu sạch thành tập Huấn luyện (Train - 80%) và tập Kiểm tra (Test - 20%)
       theo phương pháp lấy mẫu phân tầng (`stratify=y`) với `random_state=42`.
    2. Tại sao cần thực hiện bước này NGAY TRƯỚC KHI tính tham số tiền xử lý?
       Đây là chốt chặn quan trọng nhất để CHỐNG RÒ RỈ DỮ LIỆU (Anti-Data Leakage).
       Tập Test đóng vai trò là các khách hàng tương lai chưa từng xuất hiện; mọi tham số
       điền khuyết (Median/Mode), chuẩn hóa (Mean/Std) và ngưỡng rời rạc hóa (Quantiles)
       chỉ được phép học từ tập Train.
    3. Dữ liệu đầu vào (Input) là gì?
       DataFrame `df_clean` đã làm sạch dòng trùng và ngoại lai nhập lỗi.
    4. Kết quả đầu ra (Output) là gì?
       `X_train_raw`, `X_test_raw`, `y_train`, `y_test`.
    5. Hải Nam, Thành, Vũ cần hiểu cốt lõi điều gì để trả lời thầy Chiểu?
       Tham số `stratify=y` bắt buộc phải có vì dữ liệu mất cân bằng (~78.2% : 21.8%), giúp tỷ lệ
       khách vỡ nợ trong cả tập Train và tập Test đều giữ nguyên xấp xỉ 21.8%.
    """
    feature_cols = NUM_COLS + CAT_COLS
    X = df_clean[feature_cols].copy()
    y = df_clean[TARGET_COL].astype(int).copy()

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    return (
        X_train_raw.reset_index(drop=True),
        X_test_raw.reset_index(drop=True),
        y_train.reset_index(drop=True),
        y_test.reset_index(drop=True),
    )


def dien_khuyet_du_lieu(X_train_raw, X_test_raw):
    """
    GIẢI THÍCH 5 ĐIỂM QUAN TRỌNG:
    1. Đoạn code thực hiện việc gì?
       Điền giá trị thiếu (Missing Values) bằng Trung vị (Median) cho 7 biến số liên tục
       và bằng Giá trị phổ biến nhất (Mode / most_frequent) cho 4 biến phân loại.
    2. Tại sao dùng Median cho biến số mà KHÔNG dùng Mean?
       Trong bài toán tín dụng, các biến như thu nhập, thâm niên làm việc và lãi suất bị lệch phải
       (skewed). Mean dễ bị kéo lệch bởi nhóm khách hàng thu nhập cực cao, trong khi Median
       đại diện chính xác hơn cho khách hàng điển hình ở vị trí trung tâm.
    3. Dữ liệu đầu vào (Input) là gì?
       `X_train_raw`, `X_test_raw` (đang có giá trị NaN ở `person_emp_length` và `loan_int_rate`).
    4. Kết quả đầu ra (Output) là gì?
       `X_train_imp`, `X_test_imp` (không còn NaN) cùng 2 đối tượng `num_imputer`, `cat_imputer`.
    5. Hải Nam, Thành, Vũ cần hiểu cốt lõi điều gì để trả lời thầy Chiểu?
       Chỉ gọi `.fit()` trên `X_train_raw`, sau đó dùng đúng Median của tập Train để `.transform()`
       cho cả `X_train_raw` và `X_test_raw`!
    """
    num_imputer = SimpleImputer(strategy="median")
    cat_imputer = SimpleImputer(strategy="most_frequent")

    X_train_num = pd.DataFrame(
        num_imputer.fit_transform(X_train_raw[NUM_COLS]),
        columns=NUM_COLS,
    )
    X_test_num = pd.DataFrame(
        num_imputer.transform(X_test_raw[NUM_COLS]),
        columns=NUM_COLS,
    )

    X_train_cat = pd.DataFrame(
        cat_imputer.fit_transform(X_train_raw[CAT_COLS]),
        columns=CAT_COLS,
    )
    X_test_cat = pd.DataFrame(
        cat_imputer.transform(X_test_raw[CAT_COLS]),
        columns=CAT_COLS,
    )

    X_train_imp = pd.concat([X_train_num, X_train_cat], axis=1)
    X_test_imp = pd.concat([X_test_num, X_test_cat], axis=1)
    return X_train_imp, X_test_imp, num_imputer, cat_imputer


def ma_hoa_va_chuan_hoa_cho_logreg_cart_rf(X_train_imp, X_test_imp):
    """
    GIẢI THÍCH 5 ĐIỂM QUAN TRỌNG:
    1. Đoạn code thực hiện việc gì?
       - Mã hóa biến phân loại theo đúng bản chất:
         + Ordinal Encoding cho `loan_grade` (A=0 -> G=6 vì có thứ tự rủi ro tăng dần).
         + Binary Encoding cho `cb_person_default_on_file` (N=0, Y=1).
         + One-Hot Encoding cho `person_home_ownership` & `loan_intent` (biến định danh không thứ tự).
       - Tạo 2 bộ dữ liệu:
         + Bộ chưa chuẩn hóa thang đo (`X_train_tree`, `X_test_tree`) cho CART & Random Forest.
         + Bộ đã chuẩn hóa `StandardScaler` (`X_train_logreg`, `X_test_logreg`) cho Logistic Regression.
    2. Tại sao Logistic Regression cần `StandardScaler` còn Cây quyết định (CART, RF) thì không bắt buộc?
       - Logistic Regression tính tổ hợp tuyến tính z = w^T x + b kèm điều chuẩn L2 (phạt ||w||^2).
         Nếu không chuẩn hóa, biến `person_income` (hàng chục nghìn USD) có thang đo gấp hàng vạn lần
         `loan_percent_income` (0.0 - 0.8), làm thuật toán tối ưu hội tụ chậm và phạt L2 mất công bằng.
       - Ngược lại, CART và Random Forest chỉ tìm ngưỡng cắt bất biến theo phép biến đổi đơn điệu tăng
         (x <= t tương đương (x - mu)/sigma <= (t - mu)/sigma), nên giữ nguyên đơn vị gốc giúp đọc cây dễ hiểu hơn!
    3. Dữ liệu đầu vào (Input) là gì?
       `X_train_imp`, `X_test_imp` đã điền khuyết.
    4. Kết quả đầu ra (Output) là gì?
       (`X_train_tree`, `X_test_tree`), (`X_train_logreg`, `X_test_logreg`), `onehot_encoder`, `scaler`, `scale_cols`.
    5. Hải Nam, Thành, Vũ cần hiểu cốt lõi điều gì để trả lời thầy Chiểu?
       Tại sao `loan_grade` dùng Ordinal Encoding còn `loan_intent` dùng One-Hot Encoding?
       Vì hạng A < B < C < ... < G có thứ tự mức độ rủi ro thực sự; còn mục đích vay (Y tế, Giáo dục, Mua nhà)
       không có thứ tự lớn nhỏ, nếu gán 0, 1, 2, 3 sẽ khiến Logistic Regression hiểu nhầm Y tế < Mua nhà!
    """
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    ohe_train_arr = ohe.fit_transform(X_train_imp[NOMINAL_COLS])
    ohe_test_arr = ohe.transform(X_test_imp[NOMINAL_COLS])
    ohe_cols = ohe.get_feature_names_out(NOMINAL_COLS).tolist()

    def _build_encoded_df(df_imp, ohe_arr):
        df_base = df_imp[NUM_COLS].copy()
        df_base[ORDINAL_COL] = df_imp[ORDINAL_COL].map(GRADE_MAPPING).astype(float)
        df_base[BINARY_COL] = df_imp[BINARY_COL].map(BINARY_MAPPING).astype(float)
        df_ohe = pd.DataFrame(ohe_arr, columns=ohe_cols, index=df_imp.index)
        return pd.concat([df_base, df_ohe], axis=1)

    # Bộ dữ liệu giữ nguyên đơn vị thực tế phục vụ CART và Random Forest
    X_train_tree = _build_encoded_df(X_train_imp, ohe_train_arr)
    X_test_tree = _build_encoded_df(X_test_imp, ohe_test_arr)

    # Bộ dữ liệu chuẩn hóa Z-score (StandardScaler) phục vụ Logistic Regression
    scale_cols = NUM_COLS + [ORDINAL_COL]
    scaler = StandardScaler()

    X_train_logreg = X_train_tree.copy()
    X_test_logreg = X_test_tree.copy()
    X_train_logreg[scale_cols] = scaler.fit_transform(X_train_tree[scale_cols])
    X_test_logreg[scale_cols] = scaler.transform(X_test_tree[scale_cols])

    return X_train_tree, X_test_tree, X_train_logreg, X_test_logreg, ohe, scaler, scale_cols


def roi_rac_hoa_cho_id3(X_train_imp, X_test_imp):
    """
    GIẢI THÍCH 5 ĐIỂM QUAN TRỌNG:
    1. Đoạn code thực hiện việc gì?
       Chuyển đổi 7 biến số liên tục thành các khoảng phân loại rời rạc (`1_Thap`, `2_Trung_binh`,
       `3_Kha_cao`, `4_Cao`) dựa trên tứ phân vị (Quantile 25%, 50%, 75%) của tập Train,
       kết hợp với 4 biến phân loại gốc để tạo dữ liệu thuần rời rạc cho thuật toán ID3.
    2. Tại sao bắt buộc phải rời rạc hóa (Discretization / Binning) cho ID3?
       Thuật toán ID3 nguyên bản (Quinlan, 1986 - Chương 5 bài giảng thầy Chiểu) tính Information Gain
       bằng cách chia đa nhánh theo từng giá trị rời rạc của thuộc tính. Nếu để số thực liên tục
       (như thu nhập có hàng nghìn con số khác nhau), ID3 sẽ chia ra hàng nghìn nhánh con 1 mẫu ->
       Information Gain bị thổi phồng giả tạo và cây bị Overfitting hoàn toàn!
    3. Dữ liệu đầu vào (Input) là gì?
       `X_train_imp`, `X_test_imp` đã điền khuyết.
    4. Kết quả đầu ra (Output) là gì?
       `X_train_id3`, `X_test_id3` (11 cột kiểu chuỗi phân loại) và từ điển ngưỡng cắt `bin_edges_dict`.
    5. Hải Nam, Thành, Vũ cần hiểu cốt lõi điều gì để trả lời thầy Chiểu?
       Các mốc cắt Q1 (25%), Q2 (50%), Q3 (75%) CHỈ được tính trên `X_train_imp`, còn đầu mút hai bên
       mở rộng ra `(-inf, +inf)` để mọi khách hàng mới ở tập Test hay ngoài thực tế (dù thu nhập rất thấp
       hay rất cao) đều rơi gọn vào khoảng hợp lệ mà không bao giờ sinh lỗi NaN!
    """
    bin_edges_dict = {}
    bin_table_rows = []

    X_train_id3 = pd.DataFrame(index=X_train_imp.index)
    X_test_id3 = pd.DataFrame(index=X_test_imp.index)

    for col in NUM_COLS:
        q25, q50, q75 = np.quantile(X_train_imp[col].values, [0.25, 0.50, 0.75])
        internal_cuts = np.unique([float(q25), float(q50), float(q75)])
        edges = [-np.inf] + internal_cuts.tolist() + [np.inf]
        labels = BIN_LABELS[: len(edges) - 1]
        bin_edges_dict[col] = (edges, labels)

        X_train_id3[col] = pd.cut(X_train_imp[col], bins=edges, labels=labels).astype(str)
        X_test_id3[col] = pd.cut(X_test_imp[col], bins=edges, labels=labels).astype(str)

        bin_table_rows.append({
            "Bien_So": col,
            "Moc_Q1_25pct": round(float(q25), 4),
            "Moc_Q2_50pct_Median": round(float(q50), 4),
            "Moc_Q3_75pct": round(float(q75), 4),
            "Khoang_1_Thap": f"<= {q25:.2f}",
            "Khoang_2_Trung_binh": f"({q25:.2f}, {q50:.2f}]",
            "Khoang_3_Kha_cao": f"({q50:.2f}, {q75:.2f}]",
            "Khoang_4_Cao": f"> {q75:.2f}",
        })

    # Giữ nguyên 4 biến phân loại vốn đã rời rạc để ID3 chia nhánh đa hướng tự nhiên
    for col in CAT_COLS:
        X_train_id3[col] = X_train_imp[col].astype(str)
        X_test_id3[col] = X_test_imp[col].astype(str)

    df_bin_table = pd.DataFrame(bin_table_rows)
    return X_train_id3, X_test_id3, bin_edges_dict, df_bin_table


def ve_bieu_do_tien_xu_ly(df_raw, df_clean, X_train_tree, X_train_logreg, X_train_id3, figures_dir=FIGURES_DIR):
    """Xuất biểu đồ minh họa hiệu quả của Chương II (lọc ngoại lai, chuẩn hóa Z-score & rời rạc hóa ID3)."""
    os.makedirs(figures_dir, exist_ok=True)
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Segoe UI"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    # (a) So sánh tuổi trước và sau lọc ngoại lai
    axes[0].boxplot(
        [df_raw["person_age"].dropna(), df_clean["person_age"]],
        tick_labels=["Trước làm sạch\n(Max = 144)", "Sau làm sạch\n(Max <= 100)"],
        patch_artist=True,
        boxprops=dict(facecolor="#90cdf4", edgecolor="black"),
    )
    axes[0].set_title("(a) Lọc ngoại lai biến Tuổi (person_age)", fontsize=10.5)
    axes[0].set_ylabel("Tuổi (năm)")
    axes[0].grid(axis="y", linestyle="--", alpha=0.35)

    # (b) Minh họa phân phối sau chuẩn hóa StandardScaler (Mean=0, Std=1)
    axes[1].hist(X_train_logreg["loan_amnt"], bins=30, alpha=0.65, label="loan_amnt (Z-score)", color="#3182ce", edgecolor="black")
    axes[1].hist(X_train_logreg["loan_int_rate"], bins=30, alpha=0.55, label="loan_int_rate (Z-score)", color="#e53e3e", edgecolor="black")
    axes[1].axvline(0, color="black", linestyle="--", lw=1.5, label="Mean = 0")
    axes[1].set_xlim(-3, 4)
    axes[1].set_title("(b) Đưa về cùng thang đo Z-score cho Logistic", fontsize=10.5)
    axes[1].set_xlabel("Giá trị sau StandardScaler")
    axes[1].set_ylabel("Số mẫu (Tập Train)")
    axes[1].legend(fontsize=8.5)
    axes[1].grid(True, linestyle="--", alpha=0.35)

    # (c) Minh họa kết quả rời rạc hóa 4 khoảng phục vụ ID3
    bin_counts = X_train_id3["person_income"].value_counts().reindex(BIN_LABELS)
    bars = axes[2].bar(bin_counts.index, bin_counts.values, color="#48bb78", edgecolor="black", alpha=0.9)
    for bar, cnt in zip(bars, bin_counts.values):
        axes[2].text(bar.get_x() + bar.get_width() / 2, cnt + 120, f"{cnt:,}", ha="center", fontsize=9, fontweight="bold")
    axes[2].set_ylim(0, max(bin_counts.values) * 1.18)
    axes[2].set_title("(c) Rời rạc hóa person_income cho cây ID3", fontsize=10.5)
    axes[2].set_xlabel("Khoảng phân vị (Quantile Bins)")
    axes[2].set_ylabel("Số mẫu (Tập Train)")
    axes[2].grid(axis="y", linestyle="--", alpha=0.35)

    fig.suptitle("Hình 2.1: Minh họa các bước Tiền xử lý dữ liệu tại Chương II (Không rò rỉ dữ liệu)", fontsize=12.5, fontweight="bold")
    plt.tight_layout()
    out_fig = os.path.join(figures_dir, "02_minh_hoa_tien_xu_ly.png")
    plt.savefig(out_fig, dpi=180)
    plt.close()


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    df_raw = pd.read_csv(RAW_DATA_PATH)

    # 1. Làm sạch dòng trùng và ngoại lai vô lý
    df_clean, clean_stats = lam_sach_du_lieu_tho(df_raw)

    # 2. Chia Stratified Train / Test TRƯỚC TIÊN (Chống Data Leakage)
    X_train_raw, X_test_raw, y_train, y_test = chia_tap_train_test_phan_tang(df_clean)

    # 3. Điền khuyết (chỉ fit trên Train)
    X_train_imp, X_test_imp, num_imputer, cat_imputer = dien_khuyet_du_lieu(X_train_raw, X_test_raw)

    # 4. Mã hóa & Chuẩn hóa cho Logistic Regression, CART, Random Forest (chỉ fit trên Train)
    (
        X_train_tree,
        X_test_tree,
        X_train_logreg,
        X_test_logreg,
        ohe,
        scaler,
        scale_cols,
    ) = ma_hoa_va_chuan_hoa_cho_logreg_cart_rf(X_train_imp, X_test_imp)

    # 5. Rời rạc hóa dữ liệu liên tục cho thuật toán ID3 (chỉ fit ngưỡng cắt trên Train)
    X_train_id3, X_test_id3, bin_edges_dict, df_bin_table = roi_rac_hoa_cho_id3(X_train_imp, X_test_imp)

    # 6. Lưu toàn bộ tập dữ liệu đã xử lý
    X_train_tree.to_csv(os.path.join(PROCESSED_DIR, "X_train_tree.csv"), index=False)
    X_test_tree.to_csv(os.path.join(PROCESSED_DIR, "X_test_tree.csv"), index=False)
    X_train_logreg.to_csv(os.path.join(PROCESSED_DIR, "X_train_logreg.csv"), index=False)
    X_test_logreg.to_csv(os.path.join(PROCESSED_DIR, "X_test_logreg.csv"), index=False)
    X_train_id3.to_csv(os.path.join(PROCESSED_DIR, "X_train_id3.csv"), index=False)
    X_test_id3.to_csv(os.path.join(PROCESSED_DIR, "X_test_id3.csv"), index=False)
    y_train.to_csv(os.path.join(PROCESSED_DIR, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(PROCESSED_DIR, "y_test.csv"), index=False)

    # 7. Lưu gói tham số tiền xử lý để dùng lại ở Chương IV (`predict_loan_status`)
    preprocessor_bundle = {
        "num_cols": NUM_COLS,
        "cat_cols": CAT_COLS,
        "nominal_cols": NOMINAL_COLS,
        "ordinal_col": ORDINAL_COL,
        "binary_col": BINARY_COL,
        "grade_mapping": GRADE_MAPPING,
        "binary_mapping": BINARY_MAPPING,
        "num_imputer": num_imputer,
        "cat_imputer": cat_imputer,
        "onehot_encoder": ohe,
        "scaler": scaler,
        "scale_cols": scale_cols,
        "tree_feature_names": X_train_tree.columns.tolist(),
        "id3_feature_names": X_train_id3.columns.tolist(),
        "id3_bin_edges": bin_edges_dict,
    }
    bundle_path = os.path.join(MODELS_DIR, "preprocessor_bundle.joblib")
    joblib.dump(preprocessor_bundle, bundle_path)

    # 8. Xuất các bảng thông số Chương II ra CSV
    df_summary = pd.DataFrame([
        {"Buoc_Xu_Ly": "1. Dữ liệu thô ban đầu", "So_Dong": clean_stats["n_initial"], "Ghi_Chu": "Tải từ OpenML ID 43454"},
        {"Buoc_Xu_Ly": "2. Sau khi xóa dòng trùng lặp", "So_Dong": clean_stats["n_initial"] - clean_stats["n_dup_removed"],
         "Ghi_Chu": f"Đã xóa {clean_stats['n_dup_removed']} dòng trùng hoàn toàn"},
        {"Buoc_Xu_Ly": "3. Sau khi lọc ngoại lai vô lý", "So_Dong": clean_stats["n_clean"],
         "Ghi_Chu": f"Đã lọc {clean_stats['n_outlier_removed']} dòng sai logic (tuổi > 100 hoặc thâm niên > 60)"},
        {"Buoc_Xu_Ly": "4. Tập Huấn luyện (Train - 80%)", "So_Dong": len(X_train_raw),
         "Ghi_Chu": f"Tỷ lệ vỡ nợ (nhãn 1) = {y_train.mean()*100:.2f}% ({int(y_train.sum()):,} mẫu)"},
        {"Buoc_Xu_Ly": "5. Tập Kiểm tra (Test - 20%)", "So_Dong": len(X_test_raw),
         "Ghi_Chu": f"Tỷ lệ vỡ nợ (nhãn 1) = {y_test.mean()*100:.2f}% ({int(y_test.sum()):,} mẫu)"},
    ])
    df_summary.to_csv(os.path.join(TABLES_DIR, "02_tong_ket_tien_xu_ly.csv"), index=False, encoding="utf-8-sig")

    df_params = pd.DataFrame({
        "Bien_So": NUM_COLS,
        "Median_Impute_Train": np.round(num_imputer.statistics_, 4),
        "Mean_Scaler_Train": np.round(scaler.mean_[: len(NUM_COLS)], 4),
        "Std_Scaler_Train": np.round(scaler.scale_[: len(NUM_COLS)], 4),
    })
    df_params.to_csv(os.path.join(TABLES_DIR, "02_tham_so_imputation_va_scaler.csv"), index=False, encoding="utf-8-sig")
    df_bin_table.to_csv(os.path.join(TABLES_DIR, "02_nguong_roi_rac_hoa_id3.csv"), index=False, encoding="utf-8-sig")

    # 9. Vẽ biểu đồ minh họa Chương II
    ve_bieu_do_tien_xu_ly(df_raw, df_clean, X_train_tree, X_train_logreg, X_train_id3)

    # In báo cáo tổng hợp ra màn hình
    print("\n" + "=" * 78)
    print("CHƯƠNG II: KẾT QUẢ TIỀN XỬ LÝ DỮ LIỆU (CHỐNG RÒ RỈ DỮ LIỆU 100%)")
    print("=" * 78)
    print(df_summary.to_string(index=False))
    print("\n- Số chiều đặc trưng sau mã hóa cho Logistic / CART / Random Forest:", X_train_tree.shape[1], "cột")
    print("  Danh sách cột:", X_train_tree.columns.tolist())
    print("- Số chiều đặc trưng rời rạc hóa cho ID3:", X_train_id3.shape[1], "cột phân loại")
    print("\n- Bảng ngưỡng rời rạc hóa 7 biến số (học duy nhất từ tập Train) phục vụ ID3:")
    print(df_bin_table[["Bien_So", "Khoang_1_Thap", "Khoang_2_Trung_binh", "Khoang_3_Kha_cao", "Khoang_4_Cao"]].to_string(index=False))
    print(f"\n[OK] Đã lưu dữ liệu xử lý tại : {PROCESSED_DIR}")
    print(f"[OK] Đã lưu Preprocessor tại : {bundle_path}")


if __name__ == "__main__":
    main()
