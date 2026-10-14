import numpy as np
import matplotlib.pyplot as plt

# =====================================================================
# 1. NHẬP DỮ LIỆU TỪ BẢNG TRANG 7 & TRANG 20 CỦA SLIDE (20 sinh viên)
# =====================================================================
# Số giờ ôn thi (Hours)
X_hours = np.array([
    0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50, 
    2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50
])

# Kết quả thi (Pass: 0 là Trượt, 1 là Đỗ)
y = np.array([
    0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 
    1, 0, 1, 0, 1, 1, 1, 1, 1, 1
])

# =====================================================================
# 2. ĐỊNH NGHĨA HÀM SIGMOID VÀ THUẬT TOÁN GRADIENT DESCENT (TRANG 12 & 19)
# =====================================================================
def sigmoid(s):
    return 1 / (1 + np.exp(-s))

def logistic_regression_gd(X, y, w_init, eta=0.05, epochs=10000):
    """
    X: Ma trận dữ liệu đã thêm cột 1 (ứng với w0)
    y: Nhãn thực tế (0 hoặc 1)
    w_init: Hệ số khởi tạo [w0, w1]
    eta: Tốc độ học (learning rate)
    epochs: Số vòng lặp huấn luyện
    """
    w = w_init.copy()
    N = X.shape[0] # Số lượng sinh viên (20)
    
    np.random.seed(2) # Cố định ngẫu nhiên để kết quả luôn giống nhau
    
    for epoch in range(epochs):
        # Trộn ngẫu nhiên thứ tự các điểm dữ liệu trong mỗi vòng lặp (Stochastic GD)
        mix_ids = np.random.permutation(N)
        for i in mix_ids:
            xi = X[i]               # Lấy điểm dữ liệu thứ i: [1, số_giờ]
            yi = y[i]               # Lấy kết quả thực tế thứ i: 0 hoặc 1
            zi = sigmoid(np.dot(w, xi)) # Tính xác suất dự đoán zi = f(w^T * xi)
            
            # Công thức cập nhật ở TRANG 19: w = w + eta * (yi - zi) * xi
            w = w + eta * (yi - zi) * xi
            
    return w

# =====================================================================
# 3. CHẠY HUẤN LUYỆN MÔ HÌNH
# =====================================================================
# Thêm số 1 vào trước mỗi điểm dữ liệu để tính hệ số tự do w0 (w0*1 + w1*x1)
N = X_hours.shape[0]
X_bar = np.column_stack((np.ones(N), X_hours))

# Khởi tạo w = [w0, w1] ban đầu bằng [0, 0]
w_init = np.array([0.0, 0.0])

# Huấn luyện tìm w
w = logistic_regression_gd(X_bar, y, w_init, eta=0.05, epochs=10000)
w0, w1 = w[0], w[1]

print("=" * 60)
print("KẾT QUẢ TÌM HỆ SỐ w BẰNG THUẬT TOÁN GRADIENT DESCENT:")
print("=" * 60)
print(f"Hệ số w0 = {w0:.4f}")
print(f"Hệ số w1 = {w1:.4f}")
print(f"=> Hàm dự đoán xác suất đỗ: y_hat = sigmoid({w0:.4f} + {w1:.4f} * Hours)")

# Tìm ngưỡng số giờ học để đỗ (khi xác suất y_hat > 0.5 <=> w0 + w1*x > 0)
threshold_hours = -w0 / w1
print(f"=> Ngưỡng đỗ/trượt (xác suất = 0.5): Học trên {threshold_hours:.2f} giờ sẽ được dự đoán là ĐỖ.")

# =====================================================================
# 4. DỰ ĐOÁN THỬ CHO MỘT VÀI SINH VIÊN (ÁP DỤNG CÔNG THỨC TRANG 18)
# =====================================================================
print("\n" + "=" * 60)
print("DỰ ĐOÁN THỬ (Theo trang 18: Nếu xác suất > 0.5 -> Lớp 1 (Đỗ)):")
print("=" * 60)
test_hours = [1.5, 2.5, 2.8, 3.5, 5.0]
for h in test_hours:
    prob = sigmoid(w0 + w1 * h)
    ket_luan = "1 (ĐỖ)" if prob > 0.5 else "0 (TRƯỢT)"
    print(f"- Ôn thi {h} giờ -> Xác suất đỗ = {prob*100:5.1f}%  ===> Dự đoán: {ket_luan}")

# =====================================================================
# 5. VẼ ĐỒ THỊ BIỂU DIỄN KẾT QUẢ (GIỐNG SLIDE TRANG 8 & TRANG 11)
# =====================================================================
plt.figure(figsize=(9, 5))

# Tách các điểm Trượt (y=0) và Đỗ (y=1) để vẽ 2 màu khác nhau
X_fail = X_hours[y == 0]
X_pass = X_hours[y == 1]

plt.plot(X_fail, np.zeros_like(X_fail), 'ro', markersize=8, label='Trượt (Pass = 0)')
plt.plot(X_pass, np.ones_like(X_pass), 'bs', markersize=8, label='Đỗ (Pass = 1)')

# Vẽ đường cong Sigmoid màu xanh lá
xx = np.linspace(0, 6, 200)
yy = sigmoid(w0 + w1 * xx)
plt.plot(xx, yy, 'g-', linewidth=2.5, label='Đường cong Sigmoid: P(Pass=1)')

# Vẽ đường phân chia tại xác suất 0.5
plt.axhline(0.5, color='gray', linestyle='--', alpha=0.7)
plt.axvline(threshold_hours, color='orange', linestyle='--', label=f'Ngưỡng đỗ (~{threshold_hours:.2f} giờ)')

plt.xlabel('Số giờ học (Hours)', fontsize=12)
plt.ylabel('Xác suất đỗ P(Pass = 1)', fontsize=12)
plt.title('Mô hình Logistic Regression dự đoán kết quả thi', fontsize=14)
plt.xlim(0, 6)
plt.ylim(-0.1, 1.1)
plt.legend(loc='center right')
plt.grid(True, alpha=0.3)
plt.show()