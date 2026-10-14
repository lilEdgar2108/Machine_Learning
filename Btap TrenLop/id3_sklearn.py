import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# 1. Tạo bảng dữ liệu bằng Pandas
df = pd.DataFrame([
    ["<=30", "high", "no", "fair", "no"],
    ["<=30", "high", "no", "excellent", "no"],
    ["31...40", "high", "no", "fair", "yes"],
    [">40", "medium", "no", "fair", "yes"],
    [">40", "low", "yes", "fair", "yes"],
    [">40", "low", "yes", "excellent", "no"],
    ["31...40", "low", "yes", "excellent", "yes"],
    ["<=30", "medium", "no", "fair", "no"],
    ["<=30", "low", "yes", "fair", "yes"],
    [">40", "medium", "yes", "fair", "yes"],
    ["<=30", "medium", "yes", "excellent", "yes"],
    ["31...40", "medium", "no", "excellent", "yes"],
    ["31...40", "high", "yes", "fair", "yes"],
    [">40", "medium", "no", "excellent", "no"]
], columns=["age", "income", "student", "credit_rating", "buys_computer"])

# 2. Chuyển chữ (categorical) thành số vì thư viện sklearn chỉ nhận số
X = df[["age", "income", "student", "credit_rating"]]
y = df["buys_computer"]

encoder = OrdinalEncoder()
X_encoded = encoder.fit_transform(X)

# 3. Huấn luyện mô hình Decision Tree với tiêu chuẩn 'entropy' (ID3)
# (Nếu muốn đổi sang thuật toán CART ở file 4.2 thì chỉ cần sửa criterion="gini")
clf = DecisionTreeClassifier(criterion="entropy", random_state=42)
clf.fit(X_encoded, y)

# 4. Vẽ hình cây quyết định
plt.figure(figsize=(12, 8))
plot_tree(
    clf, 
    feature_names=["age", "income", "student", "credit_rating"],
    class_names=["no", "yes"],
    filled=True, 
    rounded=True,
    fontsize=11
)
plt.title("Cây quyết định dự đoán mua máy tính (buys_computer)")
plt.show()