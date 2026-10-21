"""
Machine Learning - Chương 5: Phương Pháp Học Theo Mô Hình Cây (ID3 & CART)
Giải đầy đủ các bài tập tính tay & lập trình:
- Bài 5.21: Tính tay ID3 (Entropy, Information Gain, Cây quyết định) cho bảng "Đi du lịch" (5 mẫu)
- Bài 5.22: Cài đặt thuật toán ID3 bằng Python (build_tree, predict) cho bảng "Mua máy tính" (14 mẫu)
- Bài 5.23: Thực hành CART phân lớp với scikit-learn trên tập Iris (so sánh max_depth)
- Bài 5.24: Thực hành CART hồi quy y = 2x + sin(x) + noise với max_depth = 2, 5, 10
- Bài 5.25: Đọc dữ liệu khách hàng từ CSV, cắt tỉa cây (min_samples_leaf), in Feature Importance & lưu ảnh cây
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def calc_entropy(y):
    _, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return float(-np.sum(probs * np.log2(probs)))


def calc_information_gain(df, feature, target):
    total_h = calc_entropy(df[target].values)
    weighted_h = 0.0
    for _, sub in df.groupby(feature):
        weighted_h += (len(sub) / len(df)) * calc_entropy(sub[target].values)
    return total_h - weighted_h


def build_tree(df, features, target):
    labels = df[target].values
    if len(np.unique(labels)) == 1:
        return str(labels[0])
    if len(features) == 0:
        vals, counts = np.unique(labels, return_counts=True)
        return str(vals[np.argmax(counts)])

    gains = [calc_information_gain(df, f, target) for f in features]
    best_feat = features[int(np.argmax(gains))]
    tree = {best_feat: {}}
    rem_features = [f for f in features if f != best_feat]

    for val, sub in df.groupby(best_feat):
        tree[best_feat][str(val)] = build_tree(sub, rem_features, target)
    return tree


def predict(tree, sample, default="Có"):
    if not isinstance(tree, dict):
        return tree
    root_feat = next(iter(tree))
    val = sample.get(root_feat)
    if val not in tree[root_feat]:
        return default
    return predict(tree[root_feat][val], sample, default)


# =====================================================================
# BÀI 5.21: TÍNH TAY ID3 TRÊN BẢNG "ĐI DU LỊCH" (5 MẪU)
# =====================================================================
def solve_5_21():
    print("=" * 70)
    print("BÀI 5.21: TÍNH TOÁN ID3 CHO BẢNG DỮ LIỆU 'ĐI DU LỊCH' (5 MẪU)")
    print("=" * 70)
    df_travel = pd.DataFrame([
        {"Thời tiết": "Nắng", "Gió": "Mạnh", "Đi du lịch": "Không"},
        {"Thời tiết": "Nắng", "Gió": "Nhẹ",  "Đi du lịch": "Có"},
        {"Thời tiết": "Mưa",  "Gió": "Mạnh", "Đi du lịch": "Không"},
        {"Thời tiết": "Mưa",  "Gió": "Nhẹ",  "Đi du lịch": "Có"},
        {"Thời tiết": "Âm u", "Gió": "Mạnh", "Đi du lịch": "Có"},
    ])
    h_s = calc_entropy(df_travel["Đi du lịch"].values)
    ig_weather = calc_information_gain(df_travel, "Thời tiết", "Đi du lịch")
    ig_wind = calc_information_gain(df_travel, "Gió", "Đi du lịch")

    print(f"a) Entropy toàn tập S (3 Có, 2 Không): H(S) = {h_s:.4f} bit")
    print(f"b) Information Gain:")
    print(f"   - Gain(S, Thời tiết) = {ig_weather:.4f} bit")
    print(f"   - Gain(S, Gió)       = {ig_wind:.4f} bit  (Lớn nhất => Chọn 'Gió' làm nút gốc)")
    tree_5_21 = build_tree(df_travel, ["Thời tiết", "Gió"], "Đi du lịch")
    print("c) Cây quyết định ID3:")
    print(json.dumps(tree_5_21, ensure_ascii=False, indent=2))
    print()


# =====================================================================
# BÀI 5.22: CÀI ĐẶT ID3 TRÊN BẢNG "MUA MÁY TÍNH" (14 MẪU)
# =====================================================================
def solve_5_22():
    print("=" * 70)
    print("BÀI 5.22: CÀI ĐẶT ID3 CHO BẢNG 'MUA MÁY TÍNH' (14 MẪU)")
    print("=" * 70)
    df_comp = pd.DataFrame([
        {"Tuổi": "Trẻ", "Thu nhập": "Cao", "Sinh viên": "Không", "Đánh giá tín dụng": "Tốt", "Mua máy tính": "Không"},
        {"Tuổi": "Trẻ", "Thu nhập": "Cao", "Sinh viên": "Không", "Đánh giá tín dụng": "Xuất sắc", "Mua máy tính": "Không"},
        {"Tuổi": "Trung niên", "Thu nhập": "Cao", "Sinh viên": "Không", "Đánh giá tín dụng": "Tốt", "Mua máy tính": "Có"},
        {"Tuổi": "Già", "Thu nhập": "Trung bình", "Sinh viên": "Không", "Đánh giá tín dụng": "Tốt", "Mua máy tính": "Có"},
        {"Tuổi": "Già", "Thu nhập": "Thấp", "Sinh viên": "Có", "Đánh giá tín dụng": "Tốt", "Mua máy tính": "Có"},
        {"Tuổi": "Già", "Thu nhập": "Thấp", "Sinh viên": "Có", "Đánh giá tín dụng": "Xuất sắc", "Mua máy tính": "Không"},
        {"Tuổi": "Trung niên", "Thu nhập": "Thấp", "Sinh viên": "Có", "Đánh giá tín dụng": "Xuất sắc", "Mua máy tính": "Có"},
        {"Tuổi": "Trẻ", "Thu nhập": "Trung bình", "Sinh viên": "Không", "Đánh giá tín dụng": "Tốt", "Mua máy tính": "Không"},
        {"Tuổi": "Trẻ", "Thu nhập": "Thấp", "Sinh viên": "Có", "Đánh giá tín dụng": "Tốt", "Mua máy tính": "Có"},
        {"Tuổi": "Già", "Thu nhập": "Trung bình", "Sinh viên": "Có", "Đánh giá tín dụng": "Tốt", "Mua máy tính": "Có"},
        {"Tuổi": "Trẻ", "Thu nhập": "Trung bình", "Sinh viên": "Có", "Đánh giá tín dụng": "Xuất sắc", "Mua máy tính": "Có"},
        {"Tuổi": "Trung niên", "Thu nhập": "Trung bình", "Sinh viên": "Không", "Đánh giá tín dụng": "Xuất sắc", "Mua máy tính": "Có"},
        {"Tuổi": "Trung niên", "Thu nhập": "Cao", "Sinh viên": "Có", "Đánh giá tín dụng": "Tốt", "Mua máy tính": "Có"},
        {"Tuổi": "Già", "Thu nhập": "Trung bình", "Sinh viên": "Không", "Đánh giá tín dụng": "Xuất sắc", "Mua máy tính": "Không"},
    ])
    features = ["Tuổi", "Thu nhập", "Sinh viên", "Đánh giá tín dụng"]
    for f in features:
        print(f"   Gain(S, {f:18s}) = {calc_information_gain(df_comp, f, 'Mua máy tính'):.4f} bit")

    tree = build_tree(df_comp, features, "Mua máy tính")
    print("\nCây quyết định ID3 (Dictionary):")
    print(json.dumps(tree, ensure_ascii=False, indent=2))

    test_samples = [
        {"Tuổi": "Trẻ", "Thu nhập": "Trung bình", "Sinh viên": "Có", "Đánh giá tín dụng": "Tốt"},
        {"Tuổi": "Trẻ", "Thu nhập": "Cao", "Sinh viên": "Không", "Đánh giá tín dụng": "Tốt"},
        {"Tuổi": "Già", "Thu nhập": "Thấp", "Sinh viên": "Không", "Đánh giá tín dụng": "Xuất sắc"},
    ]
    print("\nKiểm tra hàm predict trên các mẫu thử:")
    for s in test_samples:
        print(f"   Mẫu {s} => Dự đoán: {predict(tree, s)}")
    print()


# =====================================================================
# BÀI 5.23 & 5.24: CART PHÂN LỚP (IRIS) & CART HỒI QUY
# =====================================================================
def solve_5_23_and_5_24():
    print("=" * 70)
    print("BÀI 5.23: CART PHÂN LỚP TRÊN TẬP IRIS VỚI ĐỘ SÂU KHÁC NHAU")
    print("=" * 70)
    iris = load_iris()
    X_tr, X_te, y_tr, y_te = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42, stratify=iris.target)

    for depth in [None, 2, 3]:
        clf = DecisionTreeClassifier(criterion="gini", max_depth=depth, random_state=42)
        clf.fit(X_tr, y_tr)
        tr_acc = accuracy_score(y_tr, clf.predict(X_tr))
        te_acc = accuracy_score(y_te, clf.predict(X_te))
        print(f"- max_depth = {str(depth):4s} | Số nút lá: {clf.get_n_leaves():2d} | Train Acc: {tr_acc:.4f} | Test Acc: {te_acc:.4f}")

    print("\n" + "=" * 70)
    print("BÀI 5.24: CART HỒI QUY y = 2x + sin(x) + nhiễu (50 ĐIỂM)")
    print("=" * 70)
    rng = np.random.default_rng(42)
    X_reg = np.sort(rng.uniform(0, 6, 50)).reshape(-1, 1)
    y_reg = 2 * X_reg.ravel() + np.sin(X_reg.ravel()) + rng.normal(0, 0.4, 50)
    x_grid = np.linspace(0, 6, 300).reshape(-1, 1)

    plt.figure(figsize=(9, 5))
    plt.scatter(X_reg, y_reg, color="black", s=25, alpha=0.7, label="Dữ liệu gốc (có nhiễu)")
    colors = ["blue", "green", "red"]
    for depth, col in zip([2, 5, 10], colors):
        reg = DecisionTreeRegressor(max_depth=depth, random_state=42).fit(X_reg, y_reg)
        plt.plot(x_grid, reg.predict(x_grid), color=col, lw=1.8, label=f"max_depth={depth}")

    plt.title("Bài 5.24: So sánh Cây Hồi Quy CART với max_depth = 2, 5, 10")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_plot = os.path.join(OUTPUT_DIR, "chuong5_cart_regression.png")
    plt.savefig(out_plot, dpi=150)
    plt.close()
    print(f"=> Đã lưu biểu đồ CART Hồi quy tại: {out_plot}\n")


# =====================================================================
# BÀI 5.25: ĐỌC FILE CSV KHÁCH HÀNG, CẮT TỈA CÂY & LƯU ẢNH TRỰC QUAN
# =====================================================================
def solve_5_25():
    print("=" * 70)
    print("BÀI 5.25: PHÂN TÍCH DỮ LIỆU KHÁCH HÀNG CSV, CẮT TỈA & FEATURE IMPORTANCE")
    print("=" * 70)
    csv_path = os.path.join(OUTPUT_DIR, "customers_data.csv")
    rng = np.random.default_rng(42)
    n = 200
    df_cust = pd.DataFrame({
        "Age": rng.integers(18, 65, n),
        "Annual_Income_kUSD": rng.uniform(15, 120, n).round(1),
        "Spending_Score": rng.integers(1, 100, n),
        "Membership_Years": rng.integers(0, 10, n),
    })
    df_cust["Purchased_VIP"] = (
        (df_cust["Annual_Income_kUSD"] > 55) & (df_cust["Spending_Score"] > 45)
    ).astype(int)
    df_cust.to_csv(csv_path, index=False)

    df_loaded = pd.read_csv(csv_path)
    feat_names = ["Age", "Annual_Income_kUSD", "Spending_Score", "Membership_Years"]
    X = df_loaded[feat_names].values
    y = df_loaded["Purchased_VIP"].values

    for min_leaf in [1, 5, 15]:
        clf = DecisionTreeClassifier(min_samples_leaf=min_leaf, random_state=42).fit(X, y)
        print(f"- min_samples_leaf = {min_leaf:2d} | Độ sâu cây: {clf.get_depth()} | Số nút lá: {clf.get_n_leaves()}")

    best_tree = DecisionTreeClassifier(min_samples_leaf=5, random_state=42).fit(X, y)
    print("\nTầm quan trọng của các đặc trưng (Feature Importances):")
    for fname, imp in zip(feat_names, best_tree.feature_importances_):
        print(f"   + {fname:20s}: {imp:.4f}")

    plt.figure(figsize=(10, 6))
    plot_tree(best_tree, feature_names=feat_names, class_names=["Standard", "VIP"], filled=True, rounded=True)
    plt.title("Bài 5.25: Cây quyết định phân loại Khách hàng (min_samples_leaf=5)")
    plt.tight_layout()
    tree_img = os.path.join(OUTPUT_DIR, "chuong5_customer_tree.png")
    plt.savefig(tree_img, dpi=150)
    plt.close()
    print(f"=> Đã lưu file CSV ({csv_path}) và ảnh cây ({tree_img}).\n")


if __name__ == "__main__":
    solve_5_21()
    solve_5_22()
    solve_5_23_and_5_24()
    solve_5_25()
