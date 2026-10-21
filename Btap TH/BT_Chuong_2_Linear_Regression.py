"""
Machine Learning - Chương 2: Hồi Quy Tuyến Tính, Quá Khớp & Regularization
Giải đầy đủ các bài tập tự luận & thực hành:
- Bài 2.21: Minh họa Overfitting & Underfitting qua Training/Test Error
- Bài 2.22: Đánh giá chéo K-Fold Cross Validation
- Bài 2.23: Minh họa kỹ thuật Early Stopping (Dừng sớm)
- Bài 2.24: So sánh LASSO (L1) và RIDGE (L2) Regularization (trọng số & chọn đặc trưng)
- Bài 2.25: Dự báo hồi quy thực tế đánh giá qua 4 độ đo R2, NSE, MAE, RMSE
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, SGDRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def nse_score(y_true, y_pred):
    """Tính chỉ số hiệu quả Nash-Sutcliffe (NSE)."""
    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - (numerator / denominator)


# =====================================================================
# BÀI 2.21 & 2.22: OVERFITTING, UNDERFITTING & K-FOLD CROSS VALIDATION
# =====================================================================
def solve_2_21_and_2_22():
    print("=" * 70)
    print("BÀI 2.21 & 2.22: MINH HỌA UNDERFITTING, OVERFITTING & K-FOLD CV")
    print("=" * 70)

    np.random.seed(42)
    X = np.sort(np.random.uniform(0, 5, 40)).reshape(-1, 1)
    y = np.sin(X).ravel() + 0.3 * X.ravel() + np.random.normal(0, 0.2, size=len(X))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    degrees = [1, 3, 12]
    labels = ["Bậc 1 (Underfitting)", "Bậc 3 (Phù hợp - Good Fit)", "Bậc 12 (Overfitting)"]

    for deg, label in zip(degrees, labels):
        pipe = Pipeline([
            ("poly", PolynomialFeatures(degree=deg)),
            ("scaler", StandardScaler()),
            ("lr", LinearRegression())
        ])
        pipe.fit(X_train, y_train)
        tr_mse = mean_squared_error(y_train, pipe.predict(X_train))
        te_mse = mean_squared_error(y_test, pipe.predict(X_test))

        # K-Fold Cross Validation (K=5)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_mses = []
        for tr_idx, val_idx in kf.split(X):
            pipe.fit(X[tr_idx], y[tr_idx])
            cv_mses.append(mean_squared_error(y[val_idx], pipe.predict(X[val_idx])))

        print(f"- {label:28s} | Train MSE: {tr_mse:.4f} | Test MSE: {te_mse:.4f} | 5-Fold CV MSE: {np.mean(cv_mses):.4f}")
    print()


# =====================================================================
# BÀI 2.23: MINH HỌA KỸ THUẬT EARLY STOPPING
# =====================================================================
def solve_2_23_early_stopping():
    print("=" * 70)
    print("BÀI 2.23: MINH HỌA EARLY STOPPING TRONG HUẤN LUYỆN HỒI QUY")
    print("=" * 70)

    np.random.seed(42)
    X = np.linspace(-3, 3, 60).reshape(-1, 1)
    y = 0.5 * X.ravel() ** 2 + X.ravel() + 2 + np.random.normal(0, 0.8, size=60)
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

    poly_scaler = Pipeline([
        ("poly", PolynomialFeatures(degree=10, include_bias=False)),
        ("scaler", StandardScaler())
    ])
    X_tr_p = poly_scaler.fit_transform(X_tr)
    X_val_p = poly_scaler.transform(X_val)

    sgd = SGDRegressor(max_iter=1, tol=None, warm_start=True, penalty=None,
                       learning_rate="constant", eta0=0.005, random_state=42)

    best_val_mse = float("inf")
    best_epoch = 0
    for epoch in range(1, 301):
        sgd.fit(X_tr_p, y_tr)
        val_mse = mean_squared_error(y_val, sgd.predict(X_val_p))
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_epoch = epoch

    print(f"Validation MSE đạt cực tiểu tại Epoch = {best_epoch} (Val MSE = {best_val_mse:.4f}).")
    print("=> Dừng sớm (Early Stopping) tại epoch này giúp ngăn mô hình khớp vào nhiễu!\n")


# =====================================================================
# BÀI 2.24: SO SÁNH LASSO (L1) VÀ RIDGE (L2) REGULARIZATION
# =====================================================================
def solve_2_24_lasso_vs_ridge():
    print("=" * 70)
    print("BÀI 2.24: SO SÁNH LASSO (L1) VÀ RIDGE (L2) VỀ LỰA CHỌN ĐẶC TRƯNG")
    print("=" * 70)

    np.random.seed(42)
    # Tạo dữ liệu 10 đặc trưng nhưng chỉ có 3 đặc trưng đầu thực sự có ý nghĩa
    X = np.random.randn(150, 10)
    true_w = np.array([3.5, -2.0, 1.5, 0, 0, 0, 0, 0, 0, 0])
    y = X @ true_w + np.random.normal(0, 0.5, size=150)

    ridge = Ridge(alpha=10.0).fit(X, y)
    lasso = Lasso(alpha=0.2).fit(X, y)

    print("Trọng số thực tế :", np.round(true_w, 2).tolist())
    print("Trọng số Ridge L2:", np.round(ridge.coef_, 3).tolist())
    print("Trọng số LASSO L1:", np.round(lasso.coef_, 3).tolist())
    print(f"=> Số trọng số bằng 0 của Ridge: {np.sum(ridge.coef_ == 0)}/10")
    print(f"=> Số trọng số bằng 0 của LASSO: {np.sum(lasso.coef_ == 0)}/10 (Tự động loại bỏ đặc trưng thừa!)\n")


# =====================================================================
# BÀI 2.25: BÀI TOÁN HỒI QUY VỚI CÁC ĐỘ ĐO R2, NSE, MAE, RMSE
# =====================================================================
def solve_2_25_regression_project():
    print("=" * 70)
    print("BÀI 2.25: DỰ BÁO HỒI QUY & ĐÁNH GIÁ QUA R2, NSE, MAE, RMSE")
    print("=" * 70)

    # Tạo tập dữ liệu dự báo điểm thi cuối kỳ từ: [Giờ tự học, Điểm chuyên cần, Số bài tập đã làm, Giờ ngủ]
    np.random.seed(42)
    n_samples = 300
    study_hours = np.random.uniform(1, 10, n_samples)
    attendance = np.random.uniform(5, 10, n_samples)
    exercises = np.random.uniform(0, 20, n_samples)
    noise_feat = np.random.uniform(0, 10, n_samples)

    X = np.column_stack([study_hours, attendance, exercises, noise_feat])
    y = 0.45 * study_hours + 0.30 * attendance + 0.12 * exercises + 0.5 + np.random.normal(0, 0.35, n_samples)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge (alpha=1.0)": Ridge(alpha=1.0),
        "LASSO (alpha=0.05)": Lasso(alpha=0.05),
    }

    best_name, best_r2, best_preds = None, -float("inf"), None
    print(f"{'Mô hình':22s} | {'R2':>8s} | {'NSE':>8s} | {'MAE':>8s} | {'RMSE':>8s}")
    print("-" * 66)
    for name, model in models.items():
        model.fit(X_train_s, y_train)
        preds = model.predict(X_test_s)
        r2 = r2_score(y_test, preds)
        nse = nse_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        print(f"{name:22s} | {r2:8.4f} | {nse:8.4f} | {mae:8.4f} | {rmse:8.4f}")
        if r2 > best_r2:
            best_r2 = r2
            best_name = name
            best_preds = preds

    # Lưu biểu đồ Thực tế vs Dự báo
    plt.figure(figsize=(7, 5))
    plt.scatter(y_test, best_preds, alpha=0.7, edgecolors="k", label=f"Dự báo ({best_name})")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2, label="Đường lý tưởng (y = y_hat)")
    plt.title(f"Bài 2.25: Giá trị Thực tế vs Dự báo ({best_name}, R²={best_r2:.4f})")
    plt.xlabel("Điểm thi thực tế")
    plt.ylabel("Điểm thi dự báo")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "chuong2_regression_actual_vs_pred.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\n=> Đã lưu biểu đồ đánh giá hồi quy tại: {out_path}\n")


if __name__ == "__main__":
    solve_2_21_and_2_22()
    solve_2_23_early_stopping()
    solve_2_24_lasso_vs_ridge()
    solve_2_25_regression_project()
