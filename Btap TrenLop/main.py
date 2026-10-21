import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score


rng = np.random.default_rng(42)

# Tạo dữ liệu giả lập về diện tích nhà (m²) và giá bán (tỷ VNĐ)
X = np.linspace(20, 220, 40).reshape(-1, 1)
y = (
    1.2
    + 0.08 * X.ravel()
    + 0.0008 * X.ravel() ** 2
    + rng.normal(0, 2.5, size=X.shape[0])
)

# Chia tập train/test để minh họa overfitting
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# 1. Mô hình overfitting: Polynomial degree 10 + LinearRegression
poly_overfit = Pipeline(
    steps=[
        ("poly", PolynomialFeatures(degree=10, include_bias=False)),
        ("linear", LinearRegression()),
    ]
)

poly_overfit.fit(X_train, y_train)
train_r2 = r2_score(y_train, poly_overfit.predict(X_train))
test_r2 = r2_score(y_test, poly_overfit.predict(X_test))

print("=== Mô hình Overfitting ===")
print(f"Train R²: {train_r2:.6f}")
print(f"Test R²: {test_r2:.6f}")
print(f"Khoảng cách train - test: {train_r2 - test_r2:.6f}")

# 2. Đánh giá bằng K-Fold Cross-Validation
cv = KFold(n_splits=5, shuffle=True, random_state=42)
train_scores = []
val_scores = []

for train_idx, val_idx in cv.split(X):
    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    model = Pipeline(
        steps=[
            ("poly", PolynomialFeatures(degree=10, include_bias=False)),
            ("linear", LinearRegression()),
        ]
    )
    model.fit(X_tr, y_tr)

    train_scores.append(r2_score(y_tr, model.predict(X_tr)))
    val_scores.append(r2_score(y_val, model.predict(X_val)))

print("\n=== K-Fold Cross-Validation (degree=10, LinearRegression) ===")
print(f"Train score trung bình: {np.mean(train_scores):.6f}")
print(f"Validation score trung bình: {np.mean(val_scores):.6f}")
print(f"Std Validation score: {np.std(val_scores):.6f}")
print(f"Validation scores từng fold: {[round(v, 4) for v in val_scores]}")

# 3. Khắc phục overfitting bằng Ridge + GridSearchCV
ridge_pipe = Pipeline(
    steps=[
        ("poly", PolynomialFeatures(degree=10, include_bias=False)),
        ("ridge", Ridge()),
    ]
)

param_grid = {"ridge__alpha": np.logspace(-5, 3, 25)}
search = GridSearchCV(
    estimator=ridge_pipe,
    param_grid=param_grid,
    cv=5,
    scoring="r2",
    n_jobs=None,
)
search.fit(X_train, y_train)

best_model = search.best_estimator_
best_alpha = search.best_params_["ridge__alpha"]
train_r2_ridge = r2_score(y_train, best_model.predict(X_train))
test_r2_ridge = r2_score(y_test, best_model.predict(X_test))

print("\n=== Ridge + GridSearchCV ===")
print(f"Best alpha: {best_alpha:.6f}")
print(f"Train R²: {train_r2_ridge:.6f}")
print(f"Test R²: {test_r2_ridge:.6f}")
print(f"Validation score tốt nhất: {search.best_score_:.6f}")

# 4. Vẽ đồ thị so sánh
x_line = np.linspace(X.min(), X.max(), 500).reshape(-1, 1)
y_true = 1.2 + 0.08 * x_line.ravel() + 0.0008 * x_line.ravel() ** 2

y_overfit = poly_overfit.predict(x_line)
y_ridge = best_model.predict(x_line)

plt.figure(figsize=(10, 6))
plt.scatter(X, y, color="blue", s=40, alpha=0.7, label="Dữ liệu thực tế")
plt.plot(x_line.ravel(), y_true, color="black", linewidth=2, label="Hàm gốc")
plt.plot(x_line.ravel(), y_overfit, color="red", linewidth=2, label="Polynomial degree=10 + LinearRegression")
plt.plot(x_line.ravel(), y_ridge, color="green", linewidth=2, label=f"Ridge (alpha={best_alpha:.3g})")
plt.title("So sánh Overfitting và Regularization trong dự báo giá nhà")
plt.xlabel("Diện tích nhà (m²)")
plt.ylabel("Giá bán (tỷ VNĐ)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("house_pricing_overfitting.png", dpi=200)
print("\nĐã lưu biểu đồ: house_pricing_overfitting.png")
