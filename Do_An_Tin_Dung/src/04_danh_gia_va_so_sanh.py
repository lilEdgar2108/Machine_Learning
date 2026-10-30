"""
CHƯƠNG III - PHẦN 3.5: KẾT QUẢ THỰC NGHIỆM, ĐÁNH GIÁ VÀ SO SÁNH 4 THUẬT TOÁN
Trình bày đầy đủ 8 nội dung trọng tâm khi bảo vệ:
1. Confusion Matrix (TP, TN, FP, FN & ý nghĩa hậu quả tài chính FP vs FN)
2. Accuracy & nghịch lý Accuracy Paradox trên dữ liệu tín dụng mất cân bằng
3. Precision (Độ chuẩn xác - bảo vệ tập khách hàng tốt)
4. Recall / Sensitivity (Độ nhạy - bắt nợ xấu & đánh đổi Precision-Recall)
5. F1-score (Trung bình điều hòa phạt nặng khi 1 trong 2 chỉ số thấp)
6. Đường cong ROC & diện tích ROC-AUC
7. Bảng tổng hợp so sánh cả 4 mô hình (trên cả Train và Test để kiểm tra Overfitting)
8. Phân tích chuyên sâu số liệu chạy thật 100%

Đồ án Học máy - Đề tài: Dự đoán khả năng trả nợ của khách hàng vay vốn
Thành viên phụ trách chính: Vũ
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
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from id3_tu_cai_dat import ID3DecisionTree  # noqa: F401, E402

BASE_DIR = os.path.dirname(SRC_DIR)
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
FIGURES_DIR = os.path.join(BASE_DIR, "outputs", "figures")
TABLES_DIR = os.path.join(BASE_DIR, "outputs", "tables")
MODELS_DIR = os.path.join(BASE_DIR, "outputs", "models")


def tinh_bo_chi_so_danh_gia(y_true, y_pred, y_proba):
    """
    GIẢI THÍCH 5 ĐIỂM QUAN TRỌNG:
    1. Đoạn code thực hiện việc gì?
       Tính trọn bộ Confusion Matrix (TN, FP, FN, TP) và 5 độ đo chuẩn học thuật:
       Accuracy, Precision, Recall, F1-score và ROC-AUC.
    2. Tại sao phải tính đủ cả 5 độ đo mà không chỉ dùng Accuracy?
       Vì tập Test có 78.12% khách trả tốt (0) và chỉ 21.88% khách vỡ nợ (1). Một mô hình
       "vô dụng" phán tất cả mọi người đều trả nợ tốt cũng đạt Accuracy = 78.12% nhưng
       Recall = 0% (bỏ lọt 100% nợ xấu -> ngân hàng mất trắng vốn!).
    3. Dữ liệu đầu vào (Input) là gì?
       Nhãn thực tế `y_true`, nhãn dự đoán `y_pred` (ngưỡng 0.5) và xác suất vỡ nợ `y_proba`.
    4. Kết quả đầu ra (Output) là gì?
       Dictionary chứa đầy đủ các chỉ số làm tròn 4 chữ số thập phân và số đếm TN, FP, FN, TP.
    5. Hải Nam, Thành, Vũ cần hiểu cốt lõi điều gì để trả lời thầy Chiểu?
       Quy ước bắt buộc: Lớp Positive (1) = VỠ NỢ, Lớp Negative (0) = TRẢ NỢ TỐT.
       - FP (Báo động giả): Khách tốt bị phán nhầm là vỡ nợ -> Từ chối vay -> Mất cơ hội thu lãi.
       - FN (Bỏ lọt nợ xấu): Khách vỡ nợ bị phán nhầm là tốt -> Giải ngân -> Mất cả gốc lẫn lãi!
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_proba)

    fpr_single = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr_single = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    return {
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "Accuracy": round(float(acc), 4),
        "Precision": round(float(prec), 4),
        "Recall": round(float(rec), 4),
        "F1_Score": round(float(f1), 4),
        "ROC_AUC": round(float(auc), 4),
        "FPR_Tu_Choi_Nham_Pct": round(float(fpr_single * 100), 2),
        "FNR_Bo_Lot_No_Xau_Pct": round(float(fnr_single * 100), 2),
    }


def ve_confusion_matrix_4_mo_hinh(eval_results, n_test, figures_dir=FIGURES_DIR):
    """
    GIẢI THÍCH 5 ĐIỂM QUAN TRỌNG:
    1. Đoạn code thực hiện việc gì?
       Vẽ 4 ma trận nhầm lẫn (Confusion Matrix) của 4 thuật toán trên tập Test trong cùng 1 hình 2x2,
       ghi rõ tên ô (TN, FP, FN, TP), số lượng khách hàng và ý nghĩa nghiệp vụ ngân hàng.
    2. Tại sao cần thực hiện bước này?
       Giúp đối chiếu trực quan ngay lập tức: mô hình nào bắt được nhiều nợ xấu nhất (TP cao, FN thấp)
       và mô hình nào ít làm phiền/từ chối nhầm khách tốt nhất (FP thấp).
    3. Dữ liệu đầu vào (Input) là gì?
       Danh sách kết quả đánh giá `eval_results` của 4 mô hình và kích thước tập Test `n_test`.
    4. Kết quả đầu ra (Output) là gì?
       File ảnh `04_confusion_matrix_4_mo_hinh.png` phục vụ trực tiếp cho Phần 3.5 báo cáo.
    5. Hải Nam, Thành, Vũ cần hiểu cốt lõi điều gì để trả lời thầy Chiểu?
       Ô góc dưới bên trái (Thực tế = 1 Vỡ nợ, Dự đoán = 0 Trả tốt) chính là FN - ô nguy hiểm nhất
       về mặt tài chính đối với ngân hàng thương mại!
    """
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Segoe UI"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10))
    axes = axes.ravel()

    cell_titles = [
        ["TN - Duyệt đúng\nkhách tốt", "FP - Từ chối nhầm\nkhách tốt"],
        ["FN - Bỏ lọt\nnợ xấu (!)", "TP - Bắt trúng\nkhách vỡ nợ"],
    ]

    for ax, res in zip(axes, eval_results):
        cm = np.array([
            [res["test"]["TN"], res["test"]["FP"]],
            [res["test"]["FN"], res["test"]["TP"]],
        ])
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=max(cm.ravel()) * 1.08)

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Dự đoán: 0 (Trả tốt)", "Dự đoán: 1 (Vỡ nợ)"], fontsize=9.5, fontweight="bold")
        ax.set_yticklabels(["Thực tế: 0 (Trả tốt)", "Thực tế: 1 (Vỡ nợ)"], fontsize=9.5, fontweight="bold")

        for i in range(2):
            for j in range(2):
                val = cm[i, j]
                pct = val / n_test * 100
                text_color = "white" if val > (cm.max() * 0.55) else "#1a202c"
                if i == 1 and j == 0 and text_color != "white":
                    text_color = "#c53030"  # Nhấn mạnh màu đỏ cho ô FN (Bỏ lọt nợ xấu)
                ax.text(
                    j, i,
                    f"{cell_titles[i][j]}\n{val:,} mẫu ({pct:.2f}%)",
                    ha="center", va="center",
                    color=text_color, fontsize=9, fontweight="bold"
                )

        ax.set_title(
            f"{res['name']}\nAcc={res['test']['Accuracy']:.4f} | Prec={res['test']['Precision']:.4f} "
            f"| Rec={res['test']['Recall']:.4f} | F1={res['test']['F1_Score']:.4f}",
            fontsize=10.5, fontweight="bold", pad=8
        )

    fig.suptitle(f"Hình 3.5: So sánh Confusion Matrix của 4 thuật toán trên tập Test độc lập (N = {n_test:,})",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "04_confusion_matrix_4_mo_hinh.png"), dpi=180)
    plt.close()


def ve_roc_curve_va_bieu_do_cot(eval_results, y_test, figures_dir=FIGURES_DIR):
    """
    GIẢI THÍCH 5 ĐIỂM QUAN TRỌNG:
    1. Đoạn code thực hiện việc gì?
       - Vẽ chung đường cong ROC (TPR theo FPR khi quét mọi ngưỡng Threshold từ 0 đến 1) và tính AUC.
       - Vẽ biểu đồ cột nhóm so sánh trực quan 5 độ đo trên tập Test của cả 4 thuật toán.
    2. Tại sao cần đường cong ROC-AUC bên cạnh F1-score?
       Accuracy, Precision, Recall, F1 chỉ đánh giá tại 1 ngưỡng cố định (Threshold = 0.5).
       ROC-AUC đánh giá năng lực phân tách tổng thể giữa khách tốt và khách xấu trên MỌI ngưỡng cắt!
    3. Dữ liệu đầu vào (Input) là gì?
       `eval_results` chứa xác suất dự đoán trên tập Test của 4 mô hình và nhãn thật `y_test`.
    4. Kết quả đầu ra (Output) là gì?
       2 file biểu đồ: `04_roc_curve_so_sanh_4_mo_hinh.png` và `04_so_sanh_5_chi_so_test.png`.
    5. Hải Nam, Thành, Vũ cần hiểu cốt lõi điều gì để trả lời thầy Chiểu?
       - Trục tung ROC là TPR = Recall = TP/(TP+FN); trục hoành là FPR = FP/(FP+TN).
       - Đường chéo nét đứt tương ứng đoán mò ngẫu nhiên (AUC = 0.50); đường cong càng vươn sát góc
         trên bên trái (FPR -> 0, TPR -> 1) thì AUC càng gần 1.0 và mô hình phân hạng rủi ro càng giỏi!
    """
    colors = ["#e53e3e", "#dd6b20", "#38a169", "#2b6cb0"]
    linestyles = ["-.", "--", ":", "-"]

    # 1. Vẽ đường cong ROC của cả 4 mô hình
    fig, ax = plt.subplots(figsize=(8.8, 6.8))
    for res, col, ls in zip(eval_results, colors, linestyles):
        fpr, tpr, _ = roc_curve(y_test, res["y_test_proba"])
        auc_val = res["test"]["ROC_AUC"]
        ax.plot(fpr, tpr, color=col, linestyle=ls, lw=2.4,
                label=f"{res['name']} (AUC = {auc_val:.4f})")

    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1.6, label="Đoán ngẫu nhiên (AUC = 0.5000)")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.02)
    ax.set_xlabel("False Positive Rate (FPR = Tỷ lệ từ chối nhầm khách tốt)", fontsize=10.5)
    ax.set_ylabel("True Positive Rate (TPR = Recall = Tỷ lệ bắt trúng nợ xấu)", fontsize=10.5)
    ax.legend(loc="lower right", fontsize=10, frameon=True)
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.suptitle("Hình 3.6: Đường cong ROC và chỉ số AUC của 4 thuật toán trên tập Test",
                 fontsize=12.5, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "04_roc_curve_so_sanh_4_mo_hinh.png"), dpi=180)
    plt.close()

    # 2. Vẽ biểu đồ cột nhóm so sánh 5 chỉ số trên tập Test
    metrics = ["Accuracy", "Precision", "Recall", "F1_Score", "ROC_AUC"]
    metric_labels = ["Accuracy", "Precision", "Recall\n(Bắt nợ xấu)", "F1-Score", "ROC-AUC"]
    x = np.arange(len(metrics))
    width = 0.19

    fig, ax = plt.subplots(figsize=(12.5, 5.8))
    for idx, (res, col) in enumerate(zip(eval_results, colors)):
        vals = [res["test"][m] for m in metrics]
        offset = (idx - 1.5) * width
        bars = ax.bar(x + offset, vals, width=width, label=res["name"], color=col, edgecolor="black", alpha=0.9)
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v + 0.012,
                f"{v:.3f}",
                ha="center", va="bottom", fontsize=8, fontweight="bold", rotation=0
            )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=10.5, fontweight="bold")
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Giá trị độ đo (0.0 -> 1.0)", fontsize=10.5)
    ax.legend(loc="upper center", ncol=4, fontsize=9.2, frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.suptitle("Hình 3.7: Tổng hợp so sánh 5 độ đo đánh giá của 4 mô hình trên tập Test (N = 6,482)",
                 fontsize=12.5, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "04_so_sanh_5_chi_so_test.png"), dpi=180)
    plt.close()


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)

    # Đọc dữ liệu Train & Test
    X_train_logreg = pd.read_csv(os.path.join(PROCESSED_DIR, "X_train_logreg.csv"))
    X_test_logreg = pd.read_csv(os.path.join(PROCESSED_DIR, "X_test_logreg.csv"))

    X_train_tree = pd.read_csv(os.path.join(PROCESSED_DIR, "X_train_tree.csv"))
    X_test_tree = pd.read_csv(os.path.join(PROCESSED_DIR, "X_test_tree.csv"))

    X_train_id3 = pd.read_csv(os.path.join(PROCESSED_DIR, "X_train_id3.csv")).astype(str)
    X_test_id3 = pd.read_csv(os.path.join(PROCESSED_DIR, "X_test_id3.csv")).astype(str)

    y_train = pd.read_csv(os.path.join(PROCESSED_DIR, "y_train.csv"))["loan_status"].values
    y_test = pd.read_csv(os.path.join(PROCESSED_DIR, "y_test.csv"))["loan_status"].values

    # Tải 4 mô hình đã huấn luyện ở Bước 03
    logreg_model = joblib.load(os.path.join(MODELS_DIR, "logistic_regression.joblib"))
    id3_model = joblib.load(os.path.join(MODELS_DIR, "id3_decision_tree.joblib"))
    cart_model = joblib.load(os.path.join(MODELS_DIR, "cart_decision_tree.joblib"))
    rf_model = joblib.load(os.path.join(MODELS_DIR, "random_forest.joblib"))

    model_specs = [
        ("1. Logistic Regression", logreg_model, X_train_logreg, X_test_logreg),
        ("2. Decision Tree - ID3", id3_model, X_train_id3, X_test_id3),
        ("3. Decision Tree - CART", cart_model, X_train_tree, X_test_tree),
        ("4. Random Forest", rf_model, X_train_tree, X_test_tree),
    ]

    eval_results = []
    comparison_rows = []
    cm_detail_rows = []

    for name, model, X_tr, X_te in model_specs:
        tr_proba = model.predict_proba(X_tr)[:, 1]
        tr_pred = (tr_proba >= 0.5).astype(int)

        te_proba = model.predict_proba(X_te)[:, 1]
        te_pred = (te_proba >= 0.5).astype(int)

        tr_metrics = tinh_bo_chi_so_danh_gia(y_train, tr_pred, tr_proba)
        te_metrics = tinh_bo_chi_so_danh_gia(y_test, te_pred, te_proba)

        eval_results.append({
            "name": name,
            "train": tr_metrics,
            "test": te_metrics,
            "y_test_proba": te_proba,
        })

        comparison_rows.append({
            "Mo_Hinh": name,
            "Train_Accuracy": tr_metrics["Accuracy"],
            "Test_Accuracy": te_metrics["Accuracy"],
            "Test_Precision": te_metrics["Precision"],
            "Test_Recall": te_metrics["Recall"],
            "Test_F1_Score": te_metrics["F1_Score"],
            "Train_ROC_AUC": tr_metrics["ROC_AUC"],
            "Test_ROC_AUC": te_metrics["ROC_AUC"],
            "Chenh_Lech_AUC_Train_Test": round(tr_metrics["ROC_AUC"] - te_metrics["ROC_AUC"], 4),
        })

        cm_detail_rows.append({
            "Mo_Hinh": name,
            "TN_Duyet_Dung_Khach_Tot": te_metrics["TN"],
            "FP_Tu_Choi_Nham_Khach_Tot": te_metrics["FP"],
            "FN_Bo_Lot_No_Xau": te_metrics["FN"],
            "TP_Bat_Trung_No_Xau": te_metrics["TP"],
            "Ty_Le_Tu_Choi_Nham_FPR_Pct": te_metrics["FPR_Tu_Choi_Nham_Pct"],
            "Ty_Le_Bo_Lot_No_Xau_FNR_Pct": te_metrics["FNR_Bo_Lot_No_Xau_Pct"],
        })

    df_comp = pd.DataFrame(comparison_rows)
    df_cm = pd.DataFrame(cm_detail_rows)

    df_comp.to_csv(os.path.join(TABLES_DIR, "04_bang_so_sanh_4_mo_hinh.csv"), index=False, encoding="utf-8-sig")
    df_cm.to_csv(os.path.join(TABLES_DIR, "04_chi_tiet_confusion_matrix_test.csv"), index=False, encoding="utf-8-sig")

    # Vẽ các biểu đồ đánh giá Phần 3.5
    ve_confusion_matrix_4_mo_hinh(eval_results, len(y_test))
    ve_roc_curve_va_bieu_do_cot(eval_results, y_test)

    # Minh họa nghịch lý Accuracy Paradox bằng mô hình ngây thơ (luôn đoán 0)
    naive_acc = np.mean(y_test == 0)

    print("\n" + "=" * 86)
    print("PHẦN 3.5: BẢNG TỔNG HỢP KẾT QUẢ THỰC NGHIỆM TRÊN TẬP TEST ĐỘC LẬP (N = 6,482)")
    print("=" * 86)
    print(f"[Tham chiếu Accuracy Paradox] Nếu duyệt bừa 100% hồ sơ (luôn đoán nhãn 0): "
          f"Accuracy vẫn đạt {naive_acc*100:.2f}%, nhưng Recall = 0.00% (Bỏ lọt toàn bộ {int(y_test.sum()):,} ca vỡ nợ!)\n")

    print("1. Bảng so sánh tổng hợp 4 mô hình (Ngưỡng phân lớp chuẩn Threshold = 0.5):")
    print(df_comp.to_string(index=False))

    print("\n2. Bảng phân tích chi tiết Confusion Matrix & Rủi ro tài chính (FP vs FN) trên tập Test:")
    print(df_cm.to_string(index=False))
    print(f"\n[OK] Đã xuất 2 bảng tổng hợp tại: {TABLES_DIR}")
    print(f"[OK] Đã xuất 3 biểu đồ Phần 3.5 tại: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
