"""
CHƯƠNG III (MỤC 3.1 - 3.4): HUẤN LUYỆN 4 THUẬT TOÁN HỌC MÁY
1. Logistic Regression (Hồi quy Logistic với điều chuẩn L2)
2. Decision Tree - ID3 (Cây quyết định đa nhánh tự cài đặt dựa trên Entropy & Information Gain)
3. Decision Tree - CART (Cây quyết định nhị phân dựa trên chỉ số Gini Impurity)
4. Random Forest (Rừng ngẫu nhiên - Bootstrap Aggregating + Feature Subsampling, kèm OOB Score)

Đồ án Học máy - Đề tài: Dự đoán khả năng trả nợ của khách hàng vay vốn
Thành viên phụ trách chính: Thành
Giảng viên hướng dẫn: Thầy Chiểu - Khoa CNTT, Trường Đại học Thủy Lợi
"""

import os
import sys
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

# Đảm bảo import được module `id3_tu_cai_dat` dù chạy lệnh từ bất kỳ thư mục nào
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from id3_tu_cai_dat import ID3DecisionTree  # noqa: E402

BASE_DIR = os.path.dirname(SRC_DIR)
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
FIGURES_DIR = os.path.join(BASE_DIR, "outputs", "figures")
TABLES_DIR = os.path.join(BASE_DIR, "outputs", "tables")
MODELS_DIR = os.path.join(BASE_DIR, "outputs", "models")


def khao_sat_do_sau_chong_overfitting(X_train_id3, X_train_tree, y_train):
    """
    GIẢI THÍCH 5 ĐIỂM QUAN TRỌNG:
    1. Đoạn code thực hiện việc gì?
       Dùng kiểm định chéo phân tầng 5-Fold (Stratified 5-Fold Cross-Validation) CHỈ TRÊN TẬP TRAIN
       để khảo sát ảnh hưởng của độ sâu tối đa (`max_depth`) đối với cây ID3 và cây CART.
    2. Tại sao cần thực hiện bước này?
       Cây quyết định nếu không giới hạn độ sâu sẽ chia nhỏ đến từng điểm nhiễu (Overfitting: điểm Train
       rất cao nhưng điểm Validation/Test tụt giảm). Khảo sát bằng CV trên tập Train giúp chọn `max_depth`
       có căn cứ khoa học mà tuyệt đối KHÔNG chạm vào tập Test!
    3. Dữ liệu đầu vào (Input) là gì?
       `X_train_id3`, `X_train_tree` và nhãn `y_train` (25.927 mẫu tập Train).
    4. Kết quả đầu ra (Output) là gì?
       Bảng kết quả CV và biểu đồ `03_chon_do_sau_chong_overfitting.png` chứng minh hiện tượng Overfitting.
    5. Hải Nam, Thành, Vũ cần hiểu cốt lõi điều gì để trả lời thầy Chiểu?
       Khi thầy hỏi "Tại sao em chọn max_depth=6 cho ID3 và max_depth=8 cho CART?", Thành trả lời ngay:
       "Dạ bọn em chạy Stratified 5-Fold CV trên tập Train và thấy quá ngưỡng đó điểm Train tiếp tục tăng
       nhưng ROC-AUC và F1 trên tập Validation bắt đầu giảm do cây học thuộc nhiễu (Overfitting) ạ!"
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Khảo sát độ sâu cho ID3 (11 thuộc tính rời rạc -> độ sâu tối đa của cây là 11)
    id3_depths = [2, 3, 4, 5, 6, 8, 11]
    id3_cv_rows = []
    for d in id3_depths:
        tr_aucs, val_aucs, tr_accs, val_accs = [], [], [], []
        for tr_idx, val_idx in skf.split(X_train_id3, y_train):
            clf = ID3DecisionTree(max_depth=d, min_samples_split=25)
            clf.fit(X_train_id3.iloc[tr_idx], y_train[tr_idx])

            p_tr = clf.predict_proba(X_train_id3.iloc[tr_idx])[:, 1]
            p_val = clf.predict_proba(X_train_id3.iloc[val_idx])[:, 1]
            tr_aucs.append(roc_auc_score(y_train[tr_idx], p_tr))
            val_aucs.append(roc_auc_score(y_train[val_idx], p_val))
            tr_accs.append(accuracy_score(y_train[tr_idx], (p_tr >= 0.5).astype(int)))
            val_accs.append(accuracy_score(y_train[val_idx], (p_val >= 0.5).astype(int)))

        id3_cv_rows.append({
            "Mo_Hinh": "ID3",
            "max_depth": str(d),
            "Train_Acc": round(float(np.mean(tr_accs)), 4),
            "CV_Val_Acc": round(float(np.mean(val_accs)), 4),
            "Train_ROC_AUC": round(float(np.mean(tr_aucs)), 4),
            "CV_Val_ROC_AUC": round(float(np.mean(val_aucs)), 4),
        })

    # Khảo sát độ sâu cho CART (trên 19 cột số & phân loại đã mã hóa)
    cart_depths = [3, 5, 8, 10, 14, None]
    cart_cv_rows = []
    for d in cart_depths:
        tr_aucs, val_aucs, tr_accs, val_accs = [], [], [], []
        for tr_idx, val_idx in skf.split(X_train_tree, y_train):
            clf = DecisionTreeClassifier(criterion="gini", max_depth=d, min_samples_leaf=10, random_state=42)
            clf.fit(X_train_tree.iloc[tr_idx], y_train[tr_idx])

            p_tr = clf.predict_proba(X_train_tree.iloc[tr_idx])[:, 1]
            p_val = clf.predict_proba(X_train_tree.iloc[val_idx])[:, 1]
            tr_aucs.append(roc_auc_score(y_train[tr_idx], p_tr))
            val_aucs.append(roc_auc_score(y_train[val_idx], p_val))
            tr_accs.append(accuracy_score(y_train[tr_idx], (p_tr >= 0.5).astype(int)))
            val_accs.append(accuracy_score(y_train[val_idx], (p_val >= 0.5).astype(int)))

        cart_cv_rows.append({
            "Mo_Hinh": "CART",
            "max_depth": "None (Khong gioi han)" if d is None else str(d),
            "Train_Acc": round(float(np.mean(tr_accs)), 4),
            "CV_Val_Acc": round(float(np.mean(val_accs)), 4),
            "Train_ROC_AUC": round(float(np.mean(tr_aucs)), 4),
            "CV_Val_ROC_AUC": round(float(np.mean(val_aucs)), 4),
        })

    df_cv = pd.DataFrame(id3_cv_rows + cart_cv_rows)
    df_cv.to_csv(os.path.join(TABLES_DIR, "03_khao_sat_do_sau_cv.csv"), index=False, encoding="utf-8-sig")

    # Vẽ biểu đồ chứng minh Overfitting theo độ sâu cho ID3 và CART
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Segoe UI"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))

    df_id3_cv = pd.DataFrame(id3_cv_rows)
    axes[0].plot(df_id3_cv["max_depth"], df_id3_cv["Train_ROC_AUC"], "o--", color="#2b6cb0", lw=2, label="Train ROC-AUC")
    axes[0].plot(df_id3_cv["max_depth"], df_id3_cv["CV_Val_ROC_AUC"], "s-", color="#e53e3e", lw=2.2, label="5-Fold CV ROC-AUC")
    axes[0].set_title("(a) Cây ID3: Train vs 5-Fold CV ROC-AUC theo max_depth", fontsize=10.5)
    axes[0].set_xlabel("Giới hạn độ sâu (max_depth)")
    axes[0].set_ylabel("ROC-AUC")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.35)

    df_cart_cv = pd.DataFrame(cart_cv_rows)
    x_labels_cart = ["3", "5", "8", "10", "14", "None"]
    axes[1].plot(x_labels_cart, df_cart_cv["Train_ROC_AUC"], "o--", color="#2b6cb0", lw=2, label="Train ROC-AUC")
    axes[1].plot(x_labels_cart, df_cart_cv["CV_Val_ROC_AUC"], "s-", color="#38a169", lw=2.2, label="5-Fold CV ROC-AUC")
    axes[1].set_title("(b) Cây CART: Train vs 5-Fold CV ROC-AUC theo max_depth", fontsize=10.5)
    axes[1].set_xlabel("Giới hạn độ sâu (max_depth)")
    axes[1].set_ylabel("ROC-AUC")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.35)

    fig.suptitle("Hình 3.1: Khảo sát cắt tỉa độ sâu cây bằng 5-Fold Stratified CV trên tập Train (Chống Overfitting)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "03_chon_do_sau_chong_overfitting.png"), dpi=180)
    plt.close()

    # Chọn độ sâu có CV_Val_ROC_AUC cao nhất
    best_id3_row = max(id3_cv_rows, key=lambda r: r["CV_Val_ROC_AUC"])
    best_cart_idx = int(np.argmax([r["CV_Val_ROC_AUC"] for r in cart_cv_rows]))
    best_id3_depth = int(best_id3_row["max_depth"])
    best_cart_depth = cart_depths[best_cart_idx]

    return best_id3_depth, best_cart_depth, df_cv


def huan_luyen_logistic_regression(X_train_logreg, y_train):
    """
    GIẢI THÍCH 5 ĐIỂM QUAN TRỌNG:
    1. Đoạn code thực hiện việc gì?
       Huấn luyện mô hình Logistic Regression có điều chuẩn L2 (`C = 1.0`) trên dữ liệu chuẩn hóa Z-score.
    2. Tại sao dùng điều chuẩn L2 và dữ liệu đã chuẩn hóa?
       Tổ hợp tuyến tính z = w^T x + b đi qua hàm Sigmoid sigma(z) = 1 / (1 + e^-z) và tối ưu hàm
       mất mát lồi Binary Cross-Entropy cộng thành phần phạt L2 (1/2 * ||w||^2) để giữ trọng số nhỏ,
       giảm nhạy cảm với đa cộng tuyến (như giữa `person_age` và `cred_hist_length`).
    3. Dữ liệu đầu vào (Input) là gì?
       `X_train_logreg` (25.927 mẫu x 19 đặc trưng đã StandardScaler) và `y_train`.
    4. Kết quả đầu ra (Output) là gì?
       Mô hình `logreg` đã huấn luyện, thời gian chạy và bảng trọng số `03_he_so_logistic_regression.csv`.
    5. Hải Nam, Thành, Vũ cần hiểu cốt lõi điều gì để trả lời thầy Chiểu?
       Trọng số w_j > 0 nghĩa là đặc trưng đó làm TĂNG xác suất vỡ nợ (VD: `loan_grade`, `loan_percent_income`);
       trọng số w_j < 0 nghĩa là đặc trưng đó làm GIẢM xác suất vỡ nợ (VD: sở hữu nhà `OWN`, thu nhập cao).
    """
    t0 = time.perf_counter()
    logreg = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", random_state=42)
    logreg.fit(X_train_logreg, y_train)
    elapsed = time.perf_counter() - t0

    weights = logreg.coef_.ravel()
    intercept = float(logreg.intercept_[0])
    df_coef = pd.DataFrame({
        "Dac_Trung": X_train_logreg.columns,
        "Trong_So_w": np.round(weights, 4),
        "Tri_Tuyet_Doi_w": np.round(np.abs(weights), 4),
        "Odds_Ratio_exp_w": np.round(np.exp(weights), 4),
        "Tac_Dong_Rui_Ro": np.where(weights > 0, "Làm TĂNG nguy cơ vỡ nợ (+)", "Làm GIẢM nguy cơ vỡ nợ (-)"),
    }).sort_values("Tri_Tuyet_Doi_w", ascending=False)
    df_coef.to_csv(os.path.join(TABLES_DIR, "03_he_so_logistic_regression.csv"), index=False, encoding="utf-8-sig")

    return logreg, elapsed, intercept, df_coef


def huan_luyen_id3(X_train_id3, y_train, max_depth):
    """
    GIẢI THÍCH 5 ĐIỂM QUAN TRỌNG:
    1. Đoạn code thực hiện việc gì?
       Huấn luyện mô hình `ID3DecisionTree` thuần Python trên tập dữ liệu đã rời rạc hóa 11 thuộc tính.
    2. Tại sao cần xuất bảng Information Gain tại nút gốc và sơ đồ nhánh đầu?
       Để chứng minh minh bạch 100% trước thầy Chiểu rằng cây ID3 của nhóm chọn nút gốc bằng
       Entropy & Information Gain thật sự và chia ĐA NHÁNH (multi-way split) theo từng khoảng giá trị,
       chứ không dùng nhầm cây nhị phân của `scikit-learn`.
    3. Dữ liệu đầu vào (Input) là gì?
       `X_train_id3` (11 cột phân loại rời rạc), `y_train`, độ sâu cắt tỉa `max_depth`.
    4. Kết quả đầu ra (Output) là gì?
       Mô hình `id3_model`, bảng IG nút gốc, file text sơ đồ nhánh đầu và biểu đồ IG nút gốc.
    5. Hải Nam, Thành, Vũ cần hiểu cốt lõi điều gì để trả lời thầy Chiểu?
       Biết chính xác Entropy ban đầu H(S) của tập Train (~0.7578 bit vì tỷ lệ vỡ nợ là 21.87%)
       và thuộc tính nào đạt Information Gain cao nhất để đứng ở Nút gốc (Root Node)!
    """
    t0 = time.perf_counter()
    id3_model = ID3DecisionTree(max_depth=max_depth, min_samples_split=25)
    id3_model.fit(X_train_id3, y_train)
    elapsed = time.perf_counter() - t0

    # Xuất bảng xếp hạng Information Gain của 11 thuộc tính tại nút gốc
    root_h = id3_model.tree_["entropy"]
    df_ig = pd.DataFrame([
        {
            "Thu_Tu_Uu_Tien": idx + 1,
            "Thuoc_Tinh": feat,
            "Entropy_Goc_H_S_bit": root_h,
            "Information_Gain_bit": round(ig, 4),
            "Entropy_Con_Lai_H_A_S_bit": round(root_h - ig, 4),
        }
        for idx, (feat, ig) in enumerate(id3_model.root_gains_.items())
    ])
    df_ig.to_csv(os.path.join(TABLES_DIR, "03_ig_nut_goc_id3.csv"), index=False, encoding="utf-8-sig")

    # Lưu sơ đồ chữ 2 tầng đầu tiên của cây ID3
    tree_str = id3_model.xuat_chuoi_cay(max_print_depth=2)
    with open(os.path.join(TABLES_DIR, "03_so_do_nhanh_dau_cay_id3.txt"), "w", encoding="utf-8") as f:
        f.write(tree_str + "\n")

    # Vẽ biểu đồ xếp hạng Information Gain tại nút gốc ID3
    fig, ax = plt.subplots(figsize=(11, 5.2))
    df_plot = df_ig.iloc[::-1]
    bars = ax.barh(df_plot["Thuoc_Tinh"], df_plot["Information_Gain_bit"], color="#dd6b20", edgecolor="black", alpha=0.9)
    for bar, val in zip(bars, df_plot["Information_Gain_bit"]):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2, f"{val:.4f} bit", va="center", fontsize=9, fontweight="bold")
    ax.set_xlim(0, max(df_ig["Information_Gain_bit"]) * 1.22)
    ax.set_xlabel("Độ lợi thông tin - Information Gain (bit)")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    fig.suptitle(f"Hình 3.2: Xếp hạng Information Gain của 11 thuộc tính tại Nút gốc cây ID3 (H(S) = {root_h:.4f} bit)",
                 fontsize=11.5, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "03_information_gain_nut_goc_id3.png"), dpi=180)
    plt.close()

    return id3_model, elapsed, df_ig


def huan_luyen_cart_va_random_forest(X_train_tree, y_train, cart_max_depth):
    """
    GIẢI THÍCH 5 ĐIỂM QUAN TRỌNG:
    1. Đoạn code thực hiện việc gì?
       - Huấn luyện cây nhị phân CART (`criterion="gini"`, `max_depth=cart_max_depth`, `min_samples_leaf=10`).
       - Huấn luyện rừng ngẫu nhiên Random Forest (`n_estimators=200`, `max_features="sqrt"`,
         `max_depth=12`, `min_samples_leaf=5`, `oob_score=True`).
    2. Tại sao Random Forest ổn định và chính xác hơn một cây CART đơn lẻ?
       Một cây CART có phương sai (Variance) cao: dữ liệu thay đổi nhẹ có thể làm đổi cấu trúc nhánh.
       Random Forest kết hợp 200 cây huấn luyện trên 200 mẫu Bootstrap độc lập (lấy mẫu có hoàn lại)
       cùng kỹ thuật chọn ngẫu nhiên sqrt(19) ~ 4 đặc trưng tại mỗi nút chia, làm giảm tương quan giữa
       các cây và giảm mạnh phương sai khi lấy trung bình xác suất!
    3. Dữ liệu đầu vào (Input) là gì?
       `X_train_tree` (25.927 mẫu x 19 đặc trưng giữ nguyên đơn vị gốc USD, năm, %) và `y_train`.
    4. Kết quả đầu ra (Output) là gì?
       Hai mô hình `cart_model`, `rf_model`, ảnh trực quan 3 tầng đầu cây CART và bảng Feature Importance.
    5. Hải Nam, Thành, Vũ cần hiểu cốt lõi điều gì để trả lời thầy Chiểu?
       Nhớ con số `oob_score_` của Random Forest! Theo bài giảng Chương 8 của thầy Chiểu, mỗi mẫu Bootstrap
       bỏ sót xấp xỉ 1/e ~ 36.8% dữ liệu (gọi là Out-of-Bag). Điểm `oob_score_` đánh giá trên chính 36.8%
       mẫu chưa nhìn thấy đó, đóng vai trò như một tập Validation nội tại miễn phí!
    """
    # 1. Huấn luyện CART
    t0 = time.perf_counter()
    cart_model = DecisionTreeClassifier(
        criterion="gini",
        max_depth=cart_max_depth,
        min_samples_leaf=10,
        random_state=42,
    )
    cart_model.fit(X_train_tree, y_train)
    cart_time = time.perf_counter() - t0

    # Vẽ trực quan 3 tầng đầu tiên của cây nhị phân CART (giống `BT_Chuong_5_Decision_Tree_ID3_CART.py`)
    fig, ax = plt.subplots(figsize=(21, 9))
    plot_tree(
        cart_model,
        max_depth=3,
        feature_names=X_train_tree.columns.tolist(),
        class_names=["Tra_Tot (0)", "Vo_No (1)"],
        filled=True,
        rounded=True,
        impurity=True,
        proportion=True,
        fontsize=8,
        ax=ax,
    )
    fig.suptitle("Hình 3.3: Trực quan hóa 3 tầng đầu tiên của Cây quyết định nhị phân CART (Tiêu chuẩn Gini)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "03_cay_quyet_dinh_cart_minh_hoa.png"), dpi=180)
    plt.close()

    # 2. Huấn luyện Random Forest
    t1 = time.perf_counter()
    rf_model = RandomForestClassifier(
        n_estimators=200,
        criterion="gini",
        max_features="sqrt",
        max_depth=12,
        min_samples_leaf=5,
        oob_score=True,
        random_state=42,
        n_jobs=-1,
    )
    rf_model.fit(X_train_tree, y_train)
    rf_time = time.perf_counter() - t1

    # Bảng so sánh mức độ quan trọng của đặc trưng (Feature Importance) giữa CART và Random Forest
    df_imp = pd.DataFrame({
        "Dac_Trung": X_train_tree.columns,
        "CART_Importance": np.round(cart_model.feature_importances_, 4),
        "RandomForest_Importance": np.round(rf_model.feature_importances_, 4),
    }).sort_values("RandomForest_Importance", ascending=False)
    df_imp.to_csv(os.path.join(TABLES_DIR, "03_feature_importance_cart_rf.csv"), index=False, encoding="utf-8-sig")

    # Vẽ biểu đồ Top 10 Feature Importance của Random Forest
    fig, ax = plt.subplots(figsize=(11, 5.2))
    top10 = df_imp.head(10).iloc[::-1]
    bars = ax.barh(top10["Dac_Trung"], top10["RandomForest_Importance"], color="#2b6cb0", edgecolor="black", alpha=0.9)
    for bar, val in zip(bars, top10["RandomForest_Importance"]):
        ax.text(val + 0.004, bar.get_y() + bar.get_height() / 2, f"{val:.4f}", va="center", fontsize=9, fontweight="bold")
    ax.set_xlim(0, max(top10["RandomForest_Importance"]) * 1.2)
    ax.set_xlabel("Mức độ quan trọng trung bình giảm Gini (Mean Decrease in Impurity)")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    fig.suptitle("Hình 3.4: Top 10 đặc trưng quan trọng nhất trong dự đoán rủi ro vỡ nợ (Random Forest)",
                 fontsize=11.5, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "03_top10_feature_importance_rf.png"), dpi=180)
    plt.close()

    return cart_model, cart_time, rf_model, rf_time, df_imp


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)

    # Đọc các tập Train đã tiền xử lý ở Chương II (KHÔNG đọc tập Test ở bước chọn siêu tham số & huấn luyện!)
    X_train_logreg = pd.read_csv(os.path.join(PROCESSED_DIR, "X_train_logreg.csv"))
    X_train_tree = pd.read_csv(os.path.join(PROCESSED_DIR, "X_train_tree.csv"))
    X_train_id3 = pd.read_csv(os.path.join(PROCESSED_DIR, "X_train_id3.csv")).astype(str)
    y_train = pd.read_csv(os.path.join(PROCESSED_DIR, "y_train.csv"))["loan_status"].values

    print("\n" + "=" * 78)
    print("CHƯƠNG III (3.1 - 3.4): KHẢO SÁT CẮT TỈA & HUẤN LUYỆN 4 MÔ HÌNH HỌC MÁY")
    print("=" * 78)

    # Bước 1: Khảo sát 5-Fold Stratified CV trên tập Train để chọn max_depth chống Overfitting
    print("[1/4] Đang chạy 5-Fold Stratified CV trên tập Train để chọn max_depth cho ID3 & CART...")
    best_id3_depth, best_cart_depth, df_cv = khao_sat_do_sau_chong_overfitting(
        X_train_id3, X_train_tree, y_train
    )
    print(df_cv.to_string(index=False))
    print(f"=> Độ sâu tối ưu theo 5-Fold CV ROC-AUC: ID3 max_depth = {best_id3_depth} | CART max_depth = {best_cart_depth}\n")

    # Bước 2: Huấn luyện Logistic Regression
    print("[2/4] Đang huấn luyện Mô hình 1: Logistic Regression (L2, C=1.0)...")
    logreg_model, logreg_time, intercept, df_coef = huan_luyen_logistic_regression(X_train_logreg, y_train)
    joblib.dump(logreg_model, os.path.join(MODELS_DIR, "logistic_regression.joblib"))
    print(f"   - Hoàn thành trong {logreg_time:.4f}s | Hệ số chặn b (Intercept) = {intercept:.4f}")
    print("   - Top 5 trọng số |w| ảnh hưởng mạnh nhất của Logistic Regression:")
    print(df_coef.head(5).to_string(index=False))

    # Bước 3: Huấn luyện ID3 tự cài đặt
    print(f"\n[3/4] Đang huấn luyện Mô hình 2: Decision Tree - ID3 tự cài đặt (max_depth={best_id3_depth})...")
    id3_model, id3_time, df_ig = huan_luyen_id3(X_train_id3, y_train, max_depth=best_id3_depth)
    joblib.dump(id3_model, os.path.join(MODELS_DIR, "id3_decision_tree.joblib"))
    id3_stats = id3_model.thong_ke_cau_truc_cay()
    print(f"   - Hoàn thành trong {id3_time:.4f}s | Độ sâu thực tế: {id3_stats['actual_depth']} "
          f"| Tổng số nút: {id3_stats['total_nodes']} | Số nút lá: {id3_stats['n_leaves']}")
    print("   - Top 5 thuộc tính có Information Gain cao nhất tại Nút gốc ID3:")
    print(df_ig.head(5).to_string(index=False))

    # Bước 4: Huấn luyện CART & Random Forest
    print(f"\n[4/4] Đang huấn luyện Mô hình 3 (CART, max_depth={best_cart_depth}) & Mô hình 4 (Random Forest, 200 cây)...")
    cart_model, cart_time, rf_model, rf_time, df_imp = huan_luyen_cart_va_random_forest(
        X_train_tree, y_train, cart_max_depth=best_cart_depth
    )
    joblib.dump(cart_model, os.path.join(MODELS_DIR, "cart_decision_tree.joblib"))
    joblib.dump(rf_model, os.path.join(MODELS_DIR, "random_forest.joblib"))

    print(f"   - CART hoàn thành trong {cart_time:.4f}s | Độ sâu: {cart_model.get_depth()} | Số nút lá: {cart_model.get_n_leaves()}")
    print(f"   - Random Forest hoàn thành trong {rf_time:.4f}s | OOB Accuracy Score (36.8% mẫu ngoài túi): {rf_model.oob_score_:.4f}")
    print("   - Top 5 đặc trưng quan trọng nhất (Feature Importance):")
    print(df_imp.head(5).to_string(index=False))

    # Lưu bảng cấu hình huấn luyện 4 mô hình
    df_train_meta = pd.DataFrame([
        {
            "Mo_Hinh": "1. Logistic Regression",
            "Du_Lieu_Huan_Luyen": "X_train_logreg (19 cột chuẩn hóa Z-score)",
            "Cau_Hinh_Chinh": "L2 Regularization, C=1.0, solver='lbfgs', Threshold=0.5",
            "Thoi_Gian_Train_Giay": round(logreg_time, 4),
            "Thong_Tin_Cau_Truc": f"19 trọng số w + 1 hệ số chặn b = {intercept:.4f}",
        },
        {
            "Mo_Hinh": "2. Decision Tree - ID3 (Tự cài đặt)",
            "Du_Lieu_Huan_Luyen": "X_train_id3 (11 cột rời rạc hóa Quantile)",
            "Cau_Hinh_Chinh": f"Entropy & Information Gain, đa nhánh, max_depth={best_id3_depth}, min_samples_split=25",
            "Thoi_Gian_Train_Giay": round(id3_time, 4),
            "Thong_Tin_Cau_Truc": f"Nút gốc: {df_ig.iloc[0]['Thuoc_Tinh']} | Độ sâu={id3_stats['actual_depth']}, Số lá={id3_stats['n_leaves']}",
        },
        {
            "Mo_Hinh": "3. Decision Tree - CART",
            "Du_Lieu_Huan_Luyen": "X_train_tree (19 cột giữ nguyên đơn vị gốc)",
            "Cau_Hinh_Chinh": f"criterion='gini', chia nhị phân, max_depth={best_cart_depth}, min_samples_leaf=10",
            "Thoi_Gian_Train_Giay": round(cart_time, 4),
            "Thong_Tin_Cau_Truc": f"Độ sâu={cart_model.get_depth()}, Số nút lá={cart_model.get_n_leaves()}",
        },
        {
            "Mo_Hinh": "4. Random Forest",
            "Du_Lieu_Huan_Luyen": "X_train_tree (19 cột giữ nguyên đơn vị gốc)",
            "Cau_Hinh_Chinh": "n_estimators=200, criterion='gini', max_features='sqrt', max_depth=12, min_samples_leaf=5, oob_score=True",
            "Thoi_Gian_Train_Giay": round(rf_time, 4),
            "Thong_Tin_Cau_Truc": f"200 cây độc lập | OOB Score = {rf_model.oob_score_:.4f}",
        },
    ])
    df_train_meta.to_csv(os.path.join(TABLES_DIR, "03_cau_hinh_4_mo_hinh.csv"), index=False, encoding="utf-8-sig")
    print(f"\n[OK] Đã lưu đầy đủ 4 mô hình tại: {MODELS_DIR}")


if __name__ == "__main__":
    main()
