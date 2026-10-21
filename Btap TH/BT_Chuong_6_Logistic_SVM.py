"""
Machine Learning - Chương 6: Hồi Quy Logistic & Máy Vector Hỗ Trợ (SVM)
Giải đầy đủ các bài tập tính tay & lập trình:
- Bài 1: Tính tay Logistic Regression (xác suất y_hat, Cross-Entropy loss, gradient, cập nhật w, b)
- Bài 2: Cài đặt Gradient Descent cho Logistic Regression trên tập Iris (vẽ Loss & đường biên)
- Bài 3: Tính tay SVM Hard-Margin cho 4 điểm 2D: lớp +1 {(0,0),(1,0)}, lớp -1 {(2,1),(3,1)}
- Bài 4: Thực hành SVM trên Breast Cancer (C = 0.1, 1, 10 & RBF kernel với PCA 2D)
- Bài 5: Tính tay SVM Hard-Margin cho 3 điểm A(1,1), B(2,2) thuộc +1 và C(2,0) thuộc -1
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# =====================================================================
# BÀI 1: TÍNH TAY LOGISTIC REGRESSION (3 ĐIỂM, 1 EPOCH)
# =====================================================================
def solve_bai_1():
    print("=" * 70)
    print("BÀI 1: TÍNH TAY HỒI QUY LOGISTIC QUA 1 EPOCH")
    print("=" * 70)
    x = np.array([1.0, 2.0, 4.0])
    y = np.array([0.0, 0.0, 1.0])
    w, b, eta = 0.5, -1.5, 0.5

    z = w * x + b
    y_hat = sigmoid(z)
    losses = -(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
    mean_loss = np.mean(losses)

    dw = np.mean((y_hat - y) * x)
    db = np.mean(y_hat - y)
    w_new = w - eta * dw
    b_new = b - eta * db

    print("a) Xác suất dự đoán y_hat = sigmoid(w*x + b) cho từng điểm:")
    for i in range(3):
        print(f"   - Điểm (x={x[i]:.0f}, y={y[i]:.0f}): z_{i+1} = {z[i]:+.1f} => y_hat_{i+1} = {y_hat[i]:.4f} (Loss l_{i+1} = {losses[i]:.4f})")
    print(f"b) Hàm mất mát Cross-Entropy trung bình: J(w, b) = {mean_loss:.4f}")
    print(f"c) Gradient trên toàn tập: dJ/dw = {dw:.4f}, dJ/db = {db:.4f}")
    print(f"d) Cập nhật sau 1 epoch (eta = {eta}): w_new = {w_new:.4f}, b_new = {b_new:.4f}\n")


# =====================================================================
# BÀI 2: CÀI ĐẶT LOGISTIC REGRESSION GD TRÊN TẬP IRIS (KHÔNG DÙNG SKLEARN)
# =====================================================================
def solve_bai_2():
    print("=" * 70)
    print("BÀI 2: CÀI ĐẶT LOGISTIC REGRESSION BẰNG GRADIENT DESCENT (TẬP IRIS)")
    print("=" * 70)
    iris = load_iris()
    mask = iris.target < 2  # Lấy 2 lớp setosa (0) và versicolor (1)
    X = iris.data[mask, :2] # 2 thuộc tính đầu
    y = iris.target[mask].astype(float)

    # Chuẩn hóa để hội tụ mượt và vẽ đồ thị
    X_mean, X_std = X.mean(axis=0), X.std(axis=0)
    X_norm = (X - X_mean) / X_std

    learning_rates = [0.01, 0.1, 0.8]
    epochs = 200
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    best_w, best_b = None, None
    for lr in learning_rates:
        w = np.zeros(2)
        b = 0.0
        loss_hist = []
        for _ in range(epochs):
            y_hat = sigmoid(X_norm @ w + b)
            eps = 1e-15
            loss = -np.mean(y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps))
            loss_hist.append(loss)
            w -= lr * (X_norm.T @ (y_hat - y)) / len(y)
            b -= lr * np.mean(y_hat - y)
        preds = (sigmoid(X_norm @ w + b) >= 0.5).astype(int)
        acc = np.mean(preds == y)
        print(f"- Learning rate = {lr:4.2f} | Loss cuối: {loss_hist[-1]:.4f} | Accuracy: {acc*100:.2f}%")
        axes[0].plot(loss_hist, lw=2, label=f"lr = {lr}")
        if lr == 0.8:
            best_w, best_b = w, b

    axes[0].set_title("Đồ thị Loss theo Epoch với 3 Learning Rate")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Binary Cross-Entropy Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Vẽ đường biên phân lớp trên mặt phẳng 2D (với lr=0.8)
    axes[1].scatter(X_norm[y == 0, 0], X_norm[y == 0, 1], color="blue", label="Setosa (0)")
    axes[1].scatter(X_norm[y == 1, 0], X_norm[y == 1, 1], color="red", label="Versicolor (1)")
    x1_vals = np.linspace(X_norm[:, 0].min() - 0.5, X_norm[:, 0].max() + 0.5, 100)
    x2_vals = -(best_w[0] * x1_vals + best_b) / best_w[1]
    axes[1].plot(x1_vals, x2_vals, "k--", lw=2, label="Đường biên phân lớp (w^T x + b = 0)")
    axes[1].set_title("Đường biên phân lớp Logistic Regression (Iris 2D)")
    axes[1].set_xlabel("Sepal length (chuẩn hóa)")
    axes[1].set_ylabel("Sepal width (chuẩn hóa)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "chuong6_logistic_iris.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"=> Đã lưu đồ thị Logistic Regression tại: {out_path}\n")


# =====================================================================
# BÀI 3 & BÀI 5: TÍNH TAY SVM LỀ CỨNG (HARD-MARGIN SVM)
# =====================================================================
def solve_bai_3_and_bai_5():
    print("=" * 70)
    print("BÀI 3: TÍNH TAY SVM HARD-MARGIN (4 ĐIỂM 2D)")
    print("=" * 70)
    w3 = np.array([-1.0, -1.0])
    b3 = 2.0
    margin_total_3 = 2.0 / np.linalg.norm(w3)
    print("Dữ liệu: Lớp +1: (0,0), (1,0) | Lớp -1: (2,1), (3,1)")
    print(f"- Hai điểm gần nhau nhất giữa 2 lớp (Support Vectors): P2(1, 0) và Q1(2, 1)")
    print(f"- Vector trọng số tối ưu: w = {w3.tolist()}, b = {b3:.1f}")
    print(f"- Phương trình siêu phẳng tối ưu: -x1 - x2 + 2 = 0 (hay x1 + x2 - 2 = 0)")
    print(f"- Độ rộng lề (Margin = 2/||w||): 2/sqrt(2) = sqrt(2) = {margin_total_3:.4f}\n")

    print("=" * 70)
    print("BÀI 5: TÍNH TAY SVM HARD-MARGIN CHO 3 ĐIỂM A(1,1), B(2,2), C(2,0)")
    print("=" * 70)
    w5 = np.array([-1.0, 1.0])
    b5 = 1.0
    margin_total_5 = 2.0 / np.linalg.norm(w5)
    B = np.array([2.0, 2.0])
    val_B = np.dot(w5, B) + b5
    print("a) Support Vectors & Siêu phẳng tối ưu:")
    print(f"   - Vector trọng số tối ưu: w = {w5.tolist()}, b = {b5:.1f}")
    print(f"   - Phương trình siêu phẳng: -x1 + x2 + 1 = 0")
    print(f"b) Độ rộng lề (Margin = 2/||w||): 2/sqrt(2) = sqrt(2) = {margin_total_5:.4f}")
    print(f"c) Kiểm tra điểm B(2, 2): w^T * B + b = (-1)*2 + 1*2 + 1 = +{val_B:.1f}")
    print("   => B(2, 2) nằm đúng trên đường biên lề +1 (-x1 + x2 + 1 = 1) (trên góc độ đối ngẫu alpha_B = 0).\n")


# =====================================================================
# BÀI 4: THỰC HÀNH SVM TRÊN TẬP BREAST CANCER (LINEAR & RBF KERNEL)
# =====================================================================
def solve_bai_4():
    print("=" * 70)
    print("BÀI 4: THỰC HÀNH SVM TRÊN TẬP DỮ LIỆU BREAST CANCER")
    print("=" * 70)
    cancer = load_breast_cancer()
    X_tr, X_te, y_tr, y_te = train_test_split(
        cancer.data, cancer.target, test_size=0.2, random_state=42, stratify=cancer.target
    )
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    print("a) So sánh SVM tuyến tính với C = 0.1, 1, 10:")
    for C_val in [0.1, 1.0, 10.0]:
        clf = SVC(kernel="linear", C=C_val, random_state=42).fit(X_tr_s, y_tr)
        n_sv = int(np.sum(clf.n_support_))
        acc = accuracy_score(y_te, clf.predict(X_te_s))
        print(f"   - C = {C_val:4.1f} | Tổng số Support Vectors: {n_sv:3d} | Test Accuracy: {acc:.4f}")

    # b) SVM với RBF Kernel trên 2 thành phần PCA đầu tiên
    pca = PCA(n_components=2, random_state=42)
    X_tr_2d = pca.fit_transform(X_tr_s)
    X_te_2d = pca.transform(X_te_s)

    configs = [(1.0, 0.05, "Vừa khớp (C=1, gamma=0.05)"), (100.0, 5.0, "Overfitting (C=100, gamma=5.0)")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    x_min, x_max = X_tr_2d[:, 0].min() - 1, X_tr_2d[:, 0].max() + 1
    y_min, y_max = X_tr_2d[:, 1].min() - 1, X_tr_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))

    print("\nb) Thử nghiệm SVM RBF Kernel trên không gian PCA 2 chiều:")
    for ax, (C_v, gam, title) in zip(axes, configs):
        rbf = SVC(kernel="rbf", C=C_v, gamma=gam).fit(X_tr_2d, y_tr)
        tr_acc = accuracy_score(y_tr, rbf.predict(X_tr_2d))
        te_acc = accuracy_score(y_te, rbf.predict(X_te_2d))
        print(f"   - {title:30s} | Train Acc: {tr_acc:.4f} | Test Acc: {te_acc:.4f}")

        Z = rbf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        ax.contourf(xx, yy, Z, alpha=0.25, cmap="coolwarm")
        ax.scatter(X_tr_2d[:, 0], X_tr_2d[:, 1], c=y_tr, s=15, cmap="coolwarm", edgecolors="k")
        ax.set_title(f"{title}\nTrain={tr_acc:.3f}, Test={te_acc:.3f}")
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "chuong6_svm_rbf_pca.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"=> Đã lưu biểu đồ biên quyết định SVM RBF tại: {out_path}\n")


if __name__ == "__main__":
    solve_bai_1()
    solve_bai_2()
    solve_bai_3_and_bai_5()
    solve_bai_4()
