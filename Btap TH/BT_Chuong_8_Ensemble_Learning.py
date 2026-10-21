"""
Machine Learning - Chương 8: Học Kết Hợp (Ensemble Learning)
Giải đầy đủ toàn bộ 5 bài tập tự luận, tính tay & lập trình:
- Bài 1: Bootstrap tự cài đặt (2000 mẫu bootstrap) tính khoảng tin cậy 95% (percentile) & vẽ histogram
- Bài 2: Tính tay Bagging với 3 mẫu bootstrap & Decision Stump (Gini tại ngưỡng 3.5), biểu quyết cho x = 4.0
- Bài 3: Random Forest OOB score (500 cây) trên load_digits so sánh với 5-Fold Cross Validation
- Bài 4: Tự cài đặt thuật toán AdaBoost từ đầu với Decision Stump (T=3 vòng) trên tập dữ liệu 1D
- Bài 5: So sánh 4 phương pháp Ensemble (Bagging, Random Forest, AdaBoost, Gradient Boosting)
"""

import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits, make_classification
from sklearn.ensemble import (BaggingClassifier, RandomForestClassifier,
                              AdaBoostClassifier, GradientBoostingClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =====================================================================
# BÀI 1: BOOTSTRAP CHO TRUNG BÌNH & KHOẢNG TIN CẬY 95%
# =====================================================================
def solve_bai_1():
    print("=" * 70)
    print("BÀI 1: ƯỚC LƯỢNG KHOẢNG TIN CẬY 95% BẰNG BOOTSTRAP (2000 MẪU)")
    print("=" * 70)
    scores = np.array([58, 70, 82, 95, 63, 77, 88, 69, 73, 85], dtype=float)
    rng = np.random.default_rng(42)
    n = len(scores)
    boot_means = np.array([np.mean(rng.choice(scores, size=n, replace=True)) for _ in range(2000)])
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])

    print(f"- Mẫu gốc (10 điểm)      : {scores.tolist()}")
    print(f"- Trung bình mẫu gốc     : {np.mean(scores):.2f}")
    print(f"- Trung bình Bootstrap   : {np.mean(boot_means):.2f}")
    print(f"- Khoảng tin cậy 95% (CI): [{ci_low:.2f}, {ci_high:.2f}]\n")

    plt.figure(figsize=(7, 4.5))
    plt.hist(boot_means, bins=30, color="skyblue", edgecolor="black", alpha=0.8)
    plt.axvline(ci_low, color="red", linestyle="--", lw=2, label=f"2.5% Percentile ({ci_low:.2f})")
    plt.axvline(ci_high, color="red", linestyle="--", lw=2, label=f"97.5% Percentile ({ci_high:.2f})")
    plt.axvline(np.mean(scores), color="navy", lw=2, label=f"Mean gốc ({np.mean(scores):.2f})")
    plt.title("Bài 1: Phân phối Bootstrap của Trung bình (2000 mẫu)")
    plt.xlabel("Giá trị trung bình mẫu Bootstrap")
    plt.ylabel("Tần số")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "chuong8_bootstrap_hist.png"), dpi=150)
    plt.close()


# =====================================================================
# BÀI 2: TÍNH TAY BAGGING VỚI 3 MẪU BOOTSTRAP & DECISION STUMP (GINI)
# =====================================================================
def gini_impurity(y):
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return float(1.0 - np.sum(probs ** 2))


def solve_bai_2():
    print("=" * 70)
    print("BÀI 2: TÍNH TAY BAGGING VỚI 3 CÂY DECISION STUMP (NGƯỠNG CHIA 3.5)")
    print("=" * 70)
    # Ánh xạ x -> y từ bảng gốc: x <= 3 có y = 0, x >= 4 có y = 1
    samples = [
        ("Mẫu Bootstrap 1", np.array([1.0, 1.0, 4.0, 5.0, 5.0, 6.0])),
        ("Mẫu Bootstrap 2", np.array([2.0, 3.0, 3.0, 4.0, 6.0, 6.0])),
        ("Mẫu Bootstrap 3", np.array([1.0, 2.0, 4.0, 4.0, 5.0, 6.0])),
    ]
    threshold = 3.5
    x_test = 4.0
    votes = []

    for idx, (name, x_boot) in enumerate(samples, 1):
        y_boot = (x_boot >= 4.0).astype(int)
        g_parent = gini_impurity(y_boot)
        left_y = y_boot[x_boot <= threshold]
        right_y = y_boot[x_boot > threshold]
        g_left = gini_impurity(left_y)
        g_right = gini_impurity(right_y)
        g_split = (len(left_y) / len(y_boot)) * g_left + (len(right_y) / len(y_boot)) * g_right

        pred_left = int(np.bincount(left_y).argmax())
        pred_right = int(np.bincount(right_y).argmax())
        pred_x = pred_right if x_test > threshold else pred_left
        votes.append(pred_x)

        print(f"- {name}: x = {x_boot.tolist()}, y = {y_boot.tolist()}")
        print(f"  + Gini trước khi chia = {g_parent:.4f} | Chia tại 3.5: Gini_trái = {g_left:.1f}, Gini_phải = {g_right:.1f} => Gini_split = {g_split:.1f}")
        print(f"  + Luật Cây {idx}: Nếu x <= 3.5 đoán {pred_left}, nếu x > 3.5 đoán {pred_right} => Với x = 4.0, Cây {idx} đoán: {pred_x}")

    final_vote = int(np.bincount(votes).argmax())
    print(f"=> Biểu quyết đa số của 3 cây cho x = 4.0: {votes} => Dự đoán cuối cùng y_hat = {final_vote}\n")


# =====================================================================
# BÀI 3: RANDOM FOREST OOB SCORE SO SÁNH VỚI 5-FOLD CROSS VALIDATION
# =====================================================================
def solve_bai_3():
    print("=" * 70)
    print("BÀI 3: RANDOM FOREST (500 CÂY) OOB SCORE VS 5-FOLD CV TRÊN LOAD_DIGITS")
    print("=" * 70)
    digits = load_digits()
    rf = RandomForestClassifier(n_estimators=500, oob_score=True, random_state=42, n_jobs=-1)
    rf.fit(digits.data, digits.target)

    cv_scores = cross_val_score(
        RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1),
        digits.data, digits.target, cv=5
    )
    print(f"- Điểm OOB Score (Out-of-Bag Accuracy) : {rf.oob_score_:.4f}")
    print(f"- Điểm 5-Fold Cross-Validation trung bình: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
    print("=> Giải thích: Với mỗi cây, ~36.8% mẫu không nằm trong mẫu bootstrap (OOB) đóng vai trò")
    print("   như tập validation độc lập, nên OOB score xấp xỉ rất sát kết quả Cross-Validation!\n")


# =====================================================================
# BÀI 4: TỰ CÀI ĐẶT ADABOOST VỚI DECISION STUMP (T = 3 VÒNG)
# =====================================================================
def solve_bai_4():
    print("=" * 70)
    print("BÀI 4: TỰ CÀI ĐẶT THUẬT TOÁN ADABOOST TỪ ĐẦU (T = 3 VÒNG)")
    print("=" * 70)
    X = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)
    y = np.array([-1, -1, -1, 1, 1, 1, 1, 1], dtype=float)

    N = len(X)
    w = np.full(N, 1.0 / N)
    # Duyệt các ngưỡng chia ứng viên (bao gồm cả ngoài biên để tạo các stump khác nhau sau khi mẫu đúng hết)
    thresholds = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5]
    stumps = []

    for t in range(1, 4):
        best_err = float("inf")
        best_cfg = None
        for th in thresholds:
            for pol in [1, -1]:
                preds = np.where(X > th, pol, -pol)
                err = float(np.sum(w[preds != y]))
                if err < best_err:
                    best_err = err
                    best_cfg = (th, pol, preds)

        th, pol, preds = best_cfg
        # Tránh chia cho 0 khi decision stump đầu tiên đã phân tách hoàn hảo 100% (err = 0)
        eps = np.clip(best_err, 1e-6, 1.0 - 1e-6)
        alpha = 0.5 * np.log((1.0 - eps) / eps)
        w = w * np.exp(-alpha * y * preds)
        w /= np.sum(w)
        stumps.append((th, pol, alpha))
        print(f"- Vòng {t}: Ngưỡng tốt nhất = {th:.1f}, Hướng = {pol:+d} | Sai số epsilon_{t} = {best_err:.4f} | Trọng số alpha_{t} = {alpha:.4f}")

    ensemble_score = sum(alpha * np.where(X > th, pol, -pol) for th, pol, alpha in stumps)
    ensemble_pred = np.sign(ensemble_score).astype(int)
    print(f"=> Dự đoán của Ensemble trên toàn tập X={X.astype(int).tolist()}:")
    print(f"   Nhãn thật y     = {y.astype(int).tolist()}")
    print(f"   Dự đoán H(x)    = {ensemble_pred.tolist()} (Chính xác 100%!)\n")


# =====================================================================
# BÀI 5: SO SÁNH CÁC PHƯƠNG PHÁP ENSEMBLE
# =====================================================================
def solve_bai_5():
    print("=" * 70)
    print("BÀI 5: SO SÁNH 4 PHƯƠNG PHÁP ENSEMBLE (100 BỘ ƯỚC LƯỢNG)")
    print("=" * 70)
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=5, n_redundant=2, random_state=42
    )
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    models = {
        "1. Bagging (DecisionTree) ": BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=100, random_state=42),
        "2. Random Forest          ": RandomForestClassifier(n_estimators=100, random_state=42),
        "3. AdaBoost (Stump d=1)   ": AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1), n_estimators=100, random_state=42),
        "4. Gradient Boosting      ": GradientBoostingClassifier(n_estimators=100, random_state=42),
    }

    print(f"{'Phương pháp Ensemble':28s} | {'Train Acc':>10s} | {'Test Acc':>10s} | {'Thời gian (s)':>13s}")
    print("-" * 70)
    for name, clf in models.items():
        t0 = time.perf_counter()
        clf.fit(X_tr, y_tr)
        elapsed = time.perf_counter() - t0
        tr_acc = accuracy_score(y_tr, clf.predict(X_tr))
        te_acc = accuracy_score(y_te, clf.predict(X_te))
        print(f"{name:28s} | {tr_acc:10.4f} | {te_acc:10.4f} | {elapsed:12.4f}s")
    print()


if __name__ == "__main__":
    solve_bai_1()
    solve_bai_2()
    solve_bai_3()
    solve_bai_4()
    solve_bai_5()
