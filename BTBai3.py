"""
Machine Learning - Chuong 3
File tong hop loi giai thuc thi cac bai tap:
- Bai 3.26: Mo phong thuat toan Gradient Descent
- Bai 3.27 & 3.28: Kiem tra va cap nhat Perceptron
- Bai 3.29: Cai dat lop Custom Perceptron
- Bai 3.30: Huan luyen va danh gia mo hinh tren tap du lieu nhi phan (Breast Cancer)
"""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


# ==========================================
# BÀI 3.26: MÔ PHỎNG GRADIENT DESCENT
# ==========================================
def solve_exercise_3_26():
    print("=" * 60)
    print("BÀI 3.26: MÔ PHỎNG GRADIENT DESCENT CHO f(x) = x^2 - 4x + 5")
    print("=" * 60)

    def f(x):
        return x**2 - 4*x + 5

    def df(x):
        return 2*x - 4

    x = 5.0
    lr = 0.2
    steps = 4

    print(f"Bước 0: x = {x:.4f}, f'(x) = {df(x):.4f}, f(x) = {f(x):.4f}")
    for t in range(1, steps + 1):
        grad = df(x)
        x = x - lr * grad
        print(f"Bước {t}: x = {x:.4f}, f'(x) = {df(x):.4f}, f(x) = {f(x):.4f}")
    print("\nNhận xét: x tiến dần về nghiệm tối ưu x* = 2, f(x) tiến về 1.\n")


# ==========================================
# BÀI 3.27 & 3.28: TÍNH TOÁN PERCEPTRON CƠ BẢN
# ==========================================
def solve_exercise_3_27_and_3_28():
    print("=" * 60)
    print("BÀI 3.27: TÍNH TOÁN NHÃN DỰ ĐOÁN PERCEPTRON")
    print("=" * 60)
    w_27 = np.array([1, 2, -10])
    x_27 = np.array([3, 4, 1])
    y_true_27 = -1

    score_27 = np.dot(w_27, x_27)
    pred_27 = 1 if score_27 >= 0 else -1
    is_misclassified_27 = (pred_27 != y_true_27)

    print(f"1. w^T * x = {score_27}")
    print(f"2. Nhãn dự đoán y_hat = {pred_27}")
    print(f"3. Nhãn thực tế y = {y_true_27} => Phân lớp sai: {is_misclassified_27}\n")

    print("=" * 60)
    print("BÀI 3.28: CẬP NHẬT TRỌNG SỐ PERCEPTRON")
    print("=" * 60)
    w_28 = np.array([-2, 1, 0])
    x_28 = np.array([2, 3, 1])
    y_28 = 1

    score_28_before = np.dot(w_28, x_28)
    print(f"1. w^T * x (trước cập nhật) = {score_28_before}")
    print(f"   Nhãn dự đoán: {-1 if score_28_before < 0 else 1} != {y_28} => BỊ SAI")

    # Cập nhật w_new = w + y * x (eta = 1)
    w_28_new = w_28 + 1.0 * y_28 * x_28
    score_28_after = np.dot(w_28_new, x_28)
    print(f"2. Trọng số mới w_new = {w_28_new.tolist()}")
    print(f"3. w_new^T * x = {score_28_after} (Dương => Đã phân loại đúng!)\n")


# ==========================================
# BÀI 3.29: XÂY DỰNG LỚP PERCEPTRON
# ==========================================
class CustomPerceptron:
    """
    Lớp cài đặt thuật toán Perceptron học từ đầu (from scratch).
    """
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.w = None
        self.b = None

    def fit(self, X, y):
        # Đưa nhãn về dạng chuẩn {-1, 1}
        y_binary = np.where(y <= 0, -1, 1)
        n_samples, n_features = X.shape

        self.w = np.zeros(n_features)
        self.b = 0.0

        for _ in range(self.n_iterations):
            misclassified = False
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.w) + self.b
                y_pred = 1 if linear_output >= 0 else -1

                if y_binary[idx] != y_pred:
                    self.w += self.learning_rate * y_binary[idx] * x_i
                    self.b += self.learning_rate * y_binary[idx]
                    misclassified = True

            if not misclassified:
                break

    def predict(self, X):
        linear_output = np.dot(X, self.w) + self.b
        return np.where(linear_output >= 0, 1, 0)


# ==========================================
# BÀI 3.30: THỰC NGHIỆM PHÂN LỚP & ĐÁNH GIÁ
# ==========================================
def solve_exercise_3_30():
    print("=" * 60)
    print("BÀI 3.30: THỰC NGHIỆM TRÊN TẬP DỮ LIỆU BREAST CANCER")
    print("=" * 60)

    # 1. Tải tập dữ liệu
    dataset = load_breast_cancer()
    X, y = dataset.data, dataset.target

    # 2. Phân chia Train/Test 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Chuẩn hóa đặc trưng
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Huấn luyện mô hình CustomPerceptron đã xây dựng ở 3.29
    model = CustomPerceptron(learning_rate=0.1, n_iterations=1000)
    model.fit(X_train_scaled, y_train)

    # 5. Dự báo trên tập kiểm thử
    y_pred = model.predict(X_test_scaled)

    # 6. Tính toán 4 độ đo yêu cầu
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Tổng số mẫu kiểm thử: {len(y_test)}")
    print(f"- Accuracy:  {acc * 100:.2f}%")
    print(f"- Precision: {prec * 100:.2f}%")
    print(f"- Recall:    {rec * 100:.2f}%")
    print(f"- F1-score:  {f1 * 100:.2f}%")
    print("\nChi tiết báo cáo phân lớp:")
    print(classification_report(y_test, y_pred, target_names=["Malignant (0)", "Benign (1)"]))


if __name__ == "__main__":
    solve_exercise_3_26()
    solve_exercise_3_27_and_3_28()
    solve_exercise_3_30()
