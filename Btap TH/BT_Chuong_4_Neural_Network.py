"""
Machine Learning - Chương 4: Mạng Nơ-ron Nhân Tạo (ANN / MLP)
Giải đầy đủ các bài tập tính tay & lập trình:
- Bài 4.19, 4.20, 4.24 & 4.25: Tính tay Lan truyền xuôi (Forward) & Lan truyền ngược (Backpropagation)
- Bài 4.26: Cài đặt mạng Nơ-ron đa tầng (MLP) từ đầu với số tầng ẩn & số nơ-ron tùy ý
- Bài 4.27: Ứng dụng MLP cho bài toán Hồi quy (đánh giá qua R2, NSE, MAE, RMSE)
- Bài 4.28: Ứng dụng MLP cho bài toán Phân lớp (đánh giá qua Accuracy, Precision, Recall, F1)
"""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (r2_score, mean_absolute_error, mean_squared_error,
                             accuracy_score, precision_score, recall_score, f1_score)


def nse_score(y_true, y_pred):
    return 1.0 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)


# =====================================================================
# BÀI 4.19, 4.20, 4.24 & 4.25: TÍNH TAY LAN TRUYỀN XUÔI & NGƯỢC
# =====================================================================
def solve_4_24_and_4_25():
    print("=" * 70)
    print("BÀI 4.24 & 4.25 (KÈM 4.19 & 4.20): TÍNH TAY FORWARD & BACKPROPAGATION")
    print("=" * 70)

    x = np.array([[1.0], [2.0]])            # (2, 1)
    W1 = np.array([[1.0, 0.0], [0.0, 1.0]]) # (2, 2)
    W2 = np.array([[1.0, 1.0]])             # (1, 2)
    y = 5.0

    # 1. Lan truyền xuôi (Bài 4.19 & 4.24)
    z1 = W1 @ x
    a1 = np.maximum(0, z1) # ReLU
    z2 = W2 @ a1
    y_hat = np.maximum(0, z2).item()
    J = 0.5 * (y - y_hat) ** 2

    print("--- KẾT QUẢ BÀI 4.24 (LAN TRUYỀN XUÔI) ---")
    print(f"z^(1) = W^(1) * x = {z1.ravel().tolist()}^T")
    print(f"a^(1) = ReLU(z^(1)) = {a1.ravel().tolist()}^T")
    print(f"z^(2) = W^(2) * a^(1) = {z2.item():.1f}")
    print(f"Đầu ra cuối cùng y_hat = ReLU(z^(2)) = {y_hat:.1f}  (Đáp án câu 4.19: C. 3)")
    print(f"Hàm mất mát J = 1/2 * (y - y_hat)^2 = 1/2 * (5 - 3)^2 = {J:.1f}\n")

    # 2. Lan truyền ngược (Bài 4.20 & 4.25)
    delta2 = (y_hat - y) * (1.0 if z2.item() > 0 else 0.0) # dJ/dz^(2) = -2
    dJ_dW2 = delta2 * a1.T                                 # (1, 2) -> [-2, -4]
    relu_grad_z1 = (z1 > 0).astype(float)
    delta1 = (W2.T * delta2) * relu_grad_z1                # (2, 1) -> [-2, -2]^T
    dJ_dW1 = delta1 @ x.T                                  # (2, 2)

    print("--- KẾT QUẢ BÀI 4.25 (LAN TRUYỀN NGƯỢC) ---")
    print(f"Sai số tầng ra delta^(2) = y_hat - y = {delta2:.1f}")
    print(f"dJ / dW^(2) = delta^(2) * (a^(1))^T = {dJ_dW2.ravel().tolist()}  (Đáp án câu 4.20: D. [-2, -4])")
    print(f"Sai số tầng ẩn delta^(1) = (W^(2))^T * delta^(2) (x) ReLU'(z^(1)) = {delta1.ravel().tolist()}^T")
    print("Gradient dJ / dW^(1) = delta^(1) * x^T =")
    print(dJ_dW1)
    print()


# =====================================================================
# BÀI 4.26: CÀI ĐẶT MẠNG NƠ-RON ĐA TẦNG TỪ ĐẦU (MLP FROM SCRATCH)
# =====================================================================
class CustomMLP:
    """
    Mạng Nơ-ron đa tầng từ đầu:
    - hidden_layers: tuple/list chứa số nơ-ron của từng tầng ẩn (VD: [16, 8])
    - task: 'regression' hoặc 'classification'
    """
    def __init__(self, input_dim, hidden_layers, output_dim=1, task="regression", lr=0.01, epochs=800):
        self.layer_sizes = [input_dim] + list(hidden_layers) + [output_dim]
        self.task = task
        self.lr = lr
        self.epochs = epochs
        self.weights = []
        self.biases = []

        rng = np.random.default_rng(42)
        for i in range(len(self.layer_sizes) - 1):
            in_d, out_d = self.layer_sizes[i], self.layer_sizes[i + 1]
            W = rng.normal(0, np.sqrt(2.0 / in_d), size=(in_d, out_d))
            b = np.zeros((1, out_d))
            self.weights.append(W)
            self.biases.append(b)

    @staticmethod
    def _relu(z):
        return np.maximum(0, z)

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -250, 250)))

    def forward(self, X):
        activations = [X]
        zs = []
        A = X
        L = len(self.weights)
        for l in range(L):
            Z = A @ self.weights[l] + self.biases[l]
            zs.append(Z)
            if l < L - 1:
                A = self._relu(Z)
            else:
                A = Z if self.task == "regression" else self._sigmoid(Z)
            activations.append(A)
        return activations, zs

    def fit(self, X, y):
        y_col = y.reshape(-1, 1)
        N = X.shape[0]
        L = len(self.weights)

        for _ in range(self.epochs):
            activations, zs = self.forward(X)
            y_hat = activations[-1]
            delta = (y_hat - y_col) / N

            for l in range(L - 1, -1, -1):
                dW = activations[l].T @ delta
                db = np.sum(delta, axis=0, keepdims=True)
                if l > 0:
                    delta = (delta @ self.weights[l].T) * (zs[l - 1] > 0).astype(float)
                self.weights[l] -= self.lr * dW
                self.biases[l] -= self.lr * db
        return self

    def predict(self, X):
        activations, _ = self.forward(X)
        out = activations[-1].ravel()
        if self.task == "regression":
            return out
        return (out >= 0.5).astype(int)


# =====================================================================
# BÀI 4.27 & 4.28: ỨNG DỤNG MLP CHO HỒI QUY VÀ PHÂN LỚP
# =====================================================================
def solve_4_27_and_4_28():
    print("=" * 70)
    print("BÀI 4.27: HUẤN LUYỆN MẠNG NƠ-RON CHO BÀI TOÁN HỒI QUY (GIÁ NHÀ GIẢ LẬP)")
    print("=" * 70)
    rng = np.random.default_rng(42)
    X_reg = rng.uniform(0, 10, size=(300, 3))
    # Quan hệ phi tuyến: giá nhà phụ thuộc diện tích, khoảng cách trung tâm, số phòng
    y_reg = 2.5 * X_reg[:, 0] + 0.4 * (X_reg[:, 1] ** 2) - 1.2 * X_reg[:, 2] + 5.0 + rng.normal(0, 0.5, 300)

    X_tr, X_te, y_tr, y_te = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    mlp_reg = CustomMLP(input_dim=3, hidden_layers=[16, 8], task="regression", lr=0.02, epochs=1200)
    mlp_reg.fit(X_tr_s, y_tr)
    preds_reg = mlp_reg.predict(X_te_s)

    print(f"- Kiến trúc mạng : 3 -> 16 (ReLU) -> 8 (ReLU) -> 1 (Linear)")
    print(f"- R2 score       : {r2_score(y_te, preds_reg):.4f}")
    print(f"- NSE score      : {nse_score(y_te, preds_reg):.4f}")
    print(f"- MAE            : {mean_absolute_error(y_te, preds_reg):.4f}")
    print(f"- RMSE           : {np.sqrt(mean_squared_error(y_te, preds_reg)):.4f}\n")

    print("=" * 70)
    print("BÀI 4.28: HUẤN LUYỆN MẠNG NƠ-RON CHO BÀI TOÁN PHÂN LỚP (BREAST CANCER)")
    print("=" * 70)
    cancer = load_breast_cancer()
    X_tr, X_te, y_tr, y_te = train_test_split(
        cancer.data, cancer.target, test_size=0.2, random_state=42, stratify=cancer.target
    )
    scaler_c = StandardScaler()
    X_tr_s = scaler_c.fit_transform(X_tr)
    X_te_s = scaler_c.transform(X_te)

    mlp_clf = CustomMLP(input_dim=X_tr.shape[1], hidden_layers=[16, 8], task="classification", lr=0.05, epochs=1000)
    mlp_clf.fit(X_tr_s, y_tr)
    preds_clf = mlp_clf.predict(X_te_s)

    print(f"- Kiến trúc mạng : 30 -> 16 (ReLU) -> 8 (ReLU) -> 1 (Sigmoid)")
    print(f"- Accuracy       : {accuracy_score(y_te, preds_clf):.4f}")
    print(f"- Precision      : {precision_score(y_te, preds_clf):.4f}")
    print(f"- Recall         : {recall_score(y_te, preds_clf):.4f}")
    print(f"- F1-score       : {f1_score(y_te, preds_clf):.4f}\n")


if __name__ == "__main__":
    solve_4_24_and_4_25()
    solve_4_27_and_4_28()
