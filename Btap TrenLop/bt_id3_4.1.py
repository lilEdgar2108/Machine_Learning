import math

# 1. NHẬP DỮ LIỆU TỪ BẢNG TRANG 31 CỦA SLIDE
data = [
    {"id": 1,  "age": "<=30",    "income": "high",   "student": "no",  "credit_rating": "fair",      "buys_computer": "no"},
    {"id": 2,  "age": "<=30",    "income": "high",   "student": "no",  "credit_rating": "excellent", "buys_computer": "no"},
    {"id": 3,  "age": "31...40", "income": "high",   "student": "no",  "credit_rating": "fair",      "buys_computer": "yes"},
    {"id": 4,  "age": ">40",     "income": "medium", "student": "no",  "credit_rating": "fair",      "buys_computer": "yes"},
    {"id": 5,  "age": ">40",     "income": "low",    "student": "yes", "credit_rating": "fair",      "buys_computer": "yes"},
    {"id": 6,  "age": ">40",     "income": "low",    "student": "yes", "credit_rating": "excellent", "buys_computer": "no"},
    {"id": 7,  "age": "31...40", "income": "low",    "student": "yes", "credit_rating": "excellent", "buys_computer": "yes"},
    {"id": 8,  "age": "<=30",    "income": "medium", "student": "no",  "credit_rating": "fair",      "buys_computer": "no"},
    {"id": 9,  "age": "<=30",    "income": "low",    "student": "yes", "credit_rating": "fair",      "buys_computer": "yes"},
    {"id": 10, "age": ">40",     "income": "medium", "student": "yes", "credit_rating": "fair",      "buys_computer": "yes"},
    {"id": 11, "age": "<=30",    "income": "medium", "student": "yes", "credit_rating": "excellent", "buys_computer": "yes"},
    {"id": 12, "age": "31...40", "income": "medium", "student": "no",  "credit_rating": "excellent", "buys_computer": "yes"},
    {"id": 13, "age": "31...40", "income": "high",   "student": "yes", "credit_rating": "fair",      "buys_computer": "yes"},
    {"id": 14, "age": ">40",     "income": "medium", "student": "no",  "credit_rating": "excellent", "buys_computer": "no"}
]

attributes = ["age", "income", "student", "credit_rating"]
target = "buys_computer"

# 2. HÀM TÍNH ENTROPY H(S) (Công thức trang 15)
def calc_entropy(subset):
    total = len(subset)
    if total == 0:
        return 0
    
    # Đếm số lượng 'yes' và 'no'
    yes_count = sum(1 for row in subset if row[target] == "yes")
    no_count = total - yes_count
    
    entropy = 0
    for count in [yes_count, no_count]:
        if count > 0:
            p = count / total
            # Trong slide thầy dùng log tự nhiên (math.log)
            entropy -= p * math.log(p) 
    return entropy

# 3. HÀM TÍNH H(x, S) CỦA MỘT THUỘC TÍNH (Công thức trang 16)
def calc_attribute_entropy(subset, attr, show_detail=False):
    total = len(subset)
    # Lấy các giá trị khác nhau của thuộc tính (VD: age có <=30, 31...40, >40)
    values = set(row[attr] for row in subset)
    
    attr_entropy = 0
    for val in sorted(values):
        # Lọc ra các dòng có giá trị bằng val
        sub_data = [row for row in subset if row[attr] == val]
        sub_h = calc_entropy(sub_data)
        weight = len(sub_data) / total
        attr_entropy += weight * sub_h
        
        if show_detail:
            yes_c = sum(1 for r in sub_data if r[target] == "yes")
            no_c = len(sub_data) - yes_c
            print(f"   + Nhánh [{attr} = {val}]: có {len(sub_data)} mẫu ({yes_c} yes, {no_c} no) -> H = {sub_h:.3f}")
            
    return attr_entropy

# 4. THUẬT TOÁN ĐỆ QUY XÂY DỰNG CÂY ID3 (Công thức trang 10)
def build_id3_tree(subset, avail_attrs, step_name="Root"):
    yes_count = sum(1 for row in subset if row[target] == "yes")
    no_count = len(subset) - yes_count
    
    # Điều kiện dừng 1: Tất cả cùng 1 lớp (Entropy = 0)
    if yes_count == len(subset):
        return "YES"
    if no_count == len(subset):
        return "NO"
        
    # Điều kiện dừng 2: Hết thuộc tính để chia -> lấy nhãn chiếm đa số
    if not avail_attrs:
        return "YES" if yes_count >= no_count else "NO"
        
    print(f"\n================ XÉT TẠI: {step_name} ================")
    current_H = calc_entropy(subset)
    print(f"Tổng số mẫu: {len(subset)} ({yes_count} yes, {no_count} no) => Entropy H(S) = {current_H:.3f}")
    
    # Tìm thuộc tính có H(x, S) nhỏ nhất (tương đương Gain lớn nhất)
    best_attr = None
    min_h = float("inf")
    
    for attr in avail_attrs:
        print(f"\n* Xét thuộc tính '{attr}':")
        h_val = calc_attribute_entropy(subset, attr, show_detail=True)
        gain = current_H - h_val
        print(f"  => H({attr}, S) = {h_val:.3f} | Information Gain = {gain:.3f}")
        
        if h_val < min_h:
            min_h = h_val
            best_attr = attr
            
    print(f"\n>>> KẾT LUẬN: Chọn thuộc tính '{best_attr}' để phân nhánh (vì H({best_attr}, S) = {min_h:.3f} nhỏ nhất)")
    
    # Tạo node và chia nhánh đệ quy
    tree = {best_attr: {}}
    remaining_attrs = [a for a in avail_attrs if a != best_attr]
    
    values = sorted(set(row[best_attr] for row in subset))
    for val in values:
        sub_data = [row for row in subset if row[best_attr] == val]
        tree[best_attr][val] = build_id3_tree(sub_data, remaining_attrs, step_name=f"Nhánh ({best_attr} = {val})")
        
    return tree

# 5. HÀM IN CÂY QUYẾT ĐỊNH CHO DỄ NHÌN
def print_tree(tree, indent=""):
    if not isinstance(tree, dict):
        print(f" ===> Kết quả: {tree}")
        return
    for attr, branches in tree.items():
        print(f"\n{indent}[Hỏi thuộc tính: {attr}?]")
        for val, subtree in branches.items():
            print(f"{indent}  ├── Nếu là '{val}':", end="")
            if isinstance(subtree, dict):
                print()
                print_tree(subtree, indent + "  │     ")
            else:
                print(f" ===> Kết luận: {subtree}")

# CHẠY CHƯƠNG TRÌNH
if __name__ == "__main__":
    decision_tree = build_id3_tree(data, attributes)
    print("\n" + "="*50)
    print("CÂY QUYẾT ĐỊNH HOÀN CHỈNH SAU KHI CHẠY ID3:")
    print("="*50)
    print_tree(decision_tree)