"""
Machine Learning - Chương 3: Gradient Descent & Thuật Toán Perceptron (PLA)
Giải đầy đủ các bài tập tính tay & lập trình:
- Bài 3.26: Tính tay 4 bước Gradient Descent cho f(x) = x^2 - 4x + 5, x^(0) = 5, eta = 0.2
- Bài 3.27: Tính w^T x, nhãn dự đoán và kiểm tra phân lớp sai trong Perceptron
- Bài 3.28: Kiểm tra phân lớp sai, cập nhật 1 bước Perceptron và tính lại w^T x
- Bài 3.29: Xây dựng lớp Perceptron (fit, predict) từ đầu
- Bài 3.30: Huấn luyện và đánh giá Perceptron qua Accuracy, Precision, Recall, F1-score
"""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


# =====================================================================
# BÀI 3.26: TÍNH TAY 4 BƯỚC GRADIENT DESCENT
# =====================================================================
def solve_3_26():
    print("=" * 70)
    print("BÀI 3.26: GRADIENT DESCENT CHO f(x) = x^2 - 4x + 5 (x0=5, eta=0.2)")
    print("=" * 70)
    f = lambda x: x ** 2 - 4 * x + 5
    df = lambda x: 2 * x - 4

    x = 5.0
    eta = 0.2
    print("1. Đạo hàm: f'(x) = 2x - 4 (Nghiệm cực tiểu thực tế: x* = 2, f(2) = 1)")
    print(f"   Khởi tạo (k=0): x^(0) = {x:.4f} | f'(x^(0)) = {df(x):.4f} | f(x^(0)) = {f(x):.4f}")
    for k in range(1, 5):
        grad = df(x)
        x = x - eta * grad
        print(f"   Bước {k}   (k={k}): x^({k}) = {x:.4f} | f'(x^({k})) = {df(x):.4f} | f(x^({k})) = {f(x):.6f}")
    print("4. Nhận xét: x^(k) giảm đơn điệu tiến dần về x* = 2, f(x^(k)) giảm dần về 1.\n")


# =====================================================================
# BÀI 3.27 & 3.28: TÍNH TAY THUẬT TOÁN PERCEPTRON (PLA)
# =====================================================================
def solve_3_27_and_3_28():
    print("=" * 70)
    print("BÀI 3.27: KIỂM TRA DỰ ĐOÁN PERCEPTRON")
    print("=" * 70)
    w = np.array([1, 2, -10])
    x = np.array([3, 4, 1])
    y_true = -1
    wTx = int(np.dot(w, x))
    y_hat = 1 if wTx >= 0 else -1
    print(f"1. Tích vô hướng w^T x = 1*3 + 2*4 + (-10)*1 = {wTx}")
    print(f"2. Vì w^T x = {wTx} > 0 => Nhãn dự đoán y_hat = +{y_hat}")
    print(f"3. Nhãn thực tế y = {y_true} != y_hat (+1) => Điểm dữ liệu BỊ PHÂN LỚP SAI.\n")

    print("=" * 70)
    print("BÀI 3.28: CẬP NHẬT TRỌNG SỐ PERCEPTRON (1 BƯỚC)")
    print("=" * 70)
    w = np.array([-2, 1, 0])
    x = np.array([2, 3, 1])
    y = 1
    wTx_before = int(np.dot(w, x))
    y_hat_before = 1 if wTx_before >= 0 else -1
    print(f"1. Trước cập nhật: w^T x = (-2)*2 + 1*3 + 0*1 = {wTx_before} < 0")
    print(f"   => Dự đoán y_hat = {y_hat_before} != y = +{y} => Mẫu BỊ PHÂN LỚP SAI.")
    w_new = w + y * x
    wTx_after = int(np.dot(w_new, x))
    print(f"2. Cập nhật Perceptron: w_new = w + y*x = [-2, 1, 0]^T + 1*[2, 3, 1]^T = {w_new.tolist()}^T")
    print(f"3. Tính lại w_new^T x = 0*2 + 4*3 + 1*1 = {wTx_after} > 0 (Đã phân lớp đúng thành +1!).\n")


# =====================================================================
# BÀI 3.29: LỚP PERCEPTRON TỰ CÀI ĐẶT
# =====================================================================
class Perceptron:
    def __init__(self, learning_rate=0.1, max_epochs=1000):
        self.lr = learning_rate
        self.max_epochs = max_epochs
        self.w = None
        self.b = 0.0

    def fit(self, X, y):
        y_bin = np.where(y <= 0, -1, 1)
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0.0

        for _ in range(self.max_epochs):
            errors = 0
            for xi, yi in zip(X, y_bin):
                if yi * (np.dot(self.w, xi) + self.b) <= 0:
                    self.w += self.lr * yi * xi
                    self.b += self.lr * yi
                    errors += 1
            if errors == 0:
                break
        return self

    def predict(self, X):
        scores = np.dot(X, self.w) + self.b
        return np.where(scores >= 0, 1, 0)


# =====================================================================
# BÀI 3.30: PHÂN LỚP NHỊ PHÂN & ĐÁNH GIÁ ACCURACY, PRECISION, RECALL, F1
# =====================================================================
def solve_3_30():
    print("=" * 70)
    print("BÀI 3.30: ĐÁNH GIÁ PERCEPTRON TRÊN BÀI TOÁN PHÁT HIỆN UNG THƯ VÚ")
    print("=" * 70)
    data = load_breast_cancer()
    X_tr, X_te, y_tr, y_te = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
    )
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    clf = Perceptron(learning_rate=0.05, max_epochs=1000).fit(X_tr_s, y_tr)
    y_pred = clf.predict(X_te_s)

    print(f"- Accuracy : {accuracy_score(y_te, y_pred):.4f}")
    print(f"- Precision: {precision_score(y_te, y_pred):.4f}")
    print(f"- Recall   : {recall_score(y_te, y_pred):.4f}")
    print(f"- F1-score : {f1_score(y_te, y_pred):.4f}\n")
    print(classification_report(y_te, y_pred, target_names=["Ác tính (0)", "Lành tính (1)"]))


if __name__ == "__main__":
    solve_3_26()
    solve_3_27_and_3_28()
    solve_3_30()
