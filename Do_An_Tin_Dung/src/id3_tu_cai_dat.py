"""
CHƯƠNG III (MỤC 3.2): TỰ CÀI ĐẶT THUẬT TOÁN CÂY QUYẾT ĐỊNH ID3 THUẦN PYTHON
Đồ án Học máy - Đề tài: Dự đoán khả năng trả nợ của khách hàng vay vốn
Thành viên phụ trách chính: Thành
Giảng viên hướng dẫn: Thầy Chiểu - Khoa CNTT, Trường Đại học Thủy Lợi

Đồng bộ ký hiệu với bài giảng Chương 5 của thầy Chiểu (`bt_id3_4.1.py` & Sách bài tập):
- Entropy toàn tập S : H(S) = - sum_{c} p_c * log2(p_c)  (đơn vị: bit)
  (Lưu ý: Slide trên lớp dùng ln - đơn vị nat; do log2(x) = ln(x)/ln(2) chỉ khác hằng số
   dương 1/ln(2) nên thứ tự Information Gain và cấu trúc cây ID3 giống hệt nhau 100%).
- Entropy có trọng số khi chia theo thuộc tính A: H(A, S) = sum_{k} (|S_k| / |S|) * H(S_k)
- Độ lợi thông tin (Information Gain): Gain(S, A) = H(S) - H(A, S)
- Phân nhánh ĐA HƯỚNG (Multi-way split) trên tập dữ liệu đã rời rạc hóa (Discretized).
"""

import sys
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


class ID3DecisionTree:
    """
    Cài đặt thuật toán ID3 chuẩn học thuật cho bài toán phân loại nhị phân (0 / 1),
    có cắt tỉa trước (`max_depth`, `min_samples_split`) để chống Overfitting và
    lưu xác suất lớp 1 tại mỗi nút để hỗ trợ `predict_proba` (vẽ đường cong ROC-AUC).
    """

    def __init__(self, max_depth=6, min_samples_split=25):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree_ = None
        self.root_gains_ = {}
        self.features_ = []

    @staticmethod
    def tinh_entropy(y):
        """
        GIẢI THÍCH 5 ĐIỂM QUAN TRỌNG:
        1. Đoạn code thực hiện việc gì?
           Tính độ hỗn loạn thông tin Entropy H(S) của vector nhãn nhị phân `y` (theo cơ số 2 - bit).
        2. Tại sao cần thực hiện bước này?
           Entropy đo mức độ "lẫn lộn" giữa khách trả tốt (0) và khách vỡ nợ (1) trong một nút.
           Nếu H(S) = 0 thì nút thuần khiết 100%; nếu tỷ lệ 50%-50% thì H(S) đạt cực đại = 1.0 bit.
        3. Dữ liệu đầu vào (Input) là gì?
           Mảng NumPy 1 chiều `y` chứa các nhãn 0 và 1 tại nút đang xét.
        4. Kết quả đầu ra (Output) là gì?
           Số thực float `H(S) >= 0.0` (đơn vị: bit).
        5. Hải Nam, Thành, Vũ cần hiểu cốt lõi điều gì để trả lời thầy Chiểu?
           Chỉ lấy các lớp có `counts > 0` để tránh lỗi toán học `0 * log2(0)` (quy ước 0*log2(0) = 0).
        """
        n = len(y)
        if n == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        probs = counts / n
        probs = probs[probs > 0]
        return float(-np.sum(probs * np.log2(probs)))

    @classmethod
    def tinh_information_gain(cls, col_values, y, parent_entropy=None):
        """
        GIẢI THÍCH 5 ĐIỂM QUAN TRỌNG:
        1. Đoạn code thực hiện việc gì?
           Tính Độ lợi thông tin Gain(S, A) = H(S) - H(A, S) khi chia đa nhánh theo thuộc tính A.
        2. Tại sao cần thực hiện bước này?
           ID3 dùng chiến lược tham lam từ trên xuống: tại mỗi bước, thuộc tính nào làm giảm
           Entropy mạnh nhất (tức Information Gain lớn nhất) sẽ được chọn làm câu hỏi phân nhánh.
        3. Dữ liệu đầu vào (Input) là gì?
           Mảng giá trị rời rạc `col_values` của thuộc tính A và mảng nhãn `y`.
        4. Kết quả đầu ra (Output) là gì?
           Giá trị float `gain = H(S) - H(A, S)` (bit).
        5. Hải Nam, Thành, Vũ cần hiểu cốt lõi điều gì để trả lời thầy Chiểu?
           Khác với CART chỉ chia 2 nhánh (Đúng/Sai), vòng lặp `unique_vals` ở đây chia thành K nhánh
           ứng với toàn bộ K giá trị rời rạc của thuộc tính A (Multi-way split).
        """
        n_total = len(y)
        if n_total == 0:
            return 0.0
        if parent_entropy is None:
            parent_entropy = cls.tinh_entropy(y)

        unique_vals, inverse_idx, counts = np.unique(
            col_values, return_inverse=True, return_counts=True
        )
        weighted_child_entropy = 0.0
        for idx, count in enumerate(counts):
            sub_y = y[inverse_idx == idx]
            weighted_child_entropy += (count / n_total) * cls.tinh_entropy(sub_y)

        return float(parent_entropy - weighted_child_entropy)

    def _xay_dung_cay_de_quy(self, X_df, y, avail_features, depth):
        """
        GIẢI THÍCH 5 ĐIỂM QUAN TRỌNG:
        1. Đoạn code thực hiện việc gì?
           Xây dựng cây quyết định ID3 đệ quy dưới dạng cấu trúc Dictionary lồng nhau.
        2. Tại sao cần lưu `prob_1` và giới hạn `max_depth`, `min_samples_split`?
           - Dữ liệu tín dụng có nhiễu và chồng lấn (2 khách hàng hồ sơ giống nhau nhưng 1 người trả tốt,
             1 người vỡ nợ). Nếu chia đến tận cùng sẽ gây Overfitting.
           - Lưu xác suất vỡ nợ `prob_1 = tỷ lệ nhãn 1 tại nút` giúp mô hình xuất được xác suất
             để vẽ đường cong ROC-AUC và dự phòng an toàn khi gặp tổ hợp giá trị mới ở tập Test.
        3. Dữ liệu đầu vào (Input) là gì?
           Bảng `X_df` (rời rạc), mảng nhãn `y`, danh sách thuộc tính còn lại `avail_features`, độ sâu `depth`.
        4. Kết quả đầu ra (Output) là gì?
           Dictionary biểu diễn nút lá (`is_leaf=True`) hoặc nút phân nhánh (`is_leaf=False`, `branches={...}`).
        5. Hải Nam, Thành, Vũ cần hiểu cốt lõi điều gì để trả lời thầy Chiểu?
           Trong ID3 nguyên bản, sau khi chọn `best_feat` để phân nhánh, thuộc tính đó được loại bỏ khỏi
           danh sách `rem_features` trên nhánh con (vì đã chia hết mọi giá trị rời rạc của nó rồi!).
        """
        n_samples = len(y)
        n_pos = int(np.sum(y == 1))
        prob_1 = float(n_pos / n_samples) if n_samples > 0 else 0.0
        pred_class = 1 if prob_1 >= 0.5 else 0
        current_h = self.tinh_entropy(y)

        # Kiểm tra 4 điều kiện dừng (Pre-pruning + Điều kiện dừng chuẩn ID3)
        if (
            len(np.unique(y)) == 1
            or len(avail_features) == 0
            or (self.max_depth is not None and depth >= self.max_depth)
            or n_samples < self.min_samples_split
        ):
            return {
                "is_leaf": True,
                "pred_class": pred_class,
                "prob_1": prob_1,
                "n_samples": n_samples,
                "n_pos": n_pos,
                "entropy": round(current_h, 4),
            }

        # Tính Information Gain cho từng thuộc tính còn lại
        gains = {}
        for feat in avail_features:
            gains[feat] = self.tinh_information_gain(X_df[feat].values, y, parent_entropy=current_h)

        # Lưu bảng Information Gain tại Nút gốc (depth == 0) để đưa vào Chương III báo cáo
        if depth == 0:
            self.root_gains_ = dict(sorted(gains.items(), key=lambda item: item[1], reverse=True))

        best_feat = max(gains, key=gains.get)
        best_gain = gains[best_feat]

        # Điều kiện dừng 5: Nếu chia nhánh không làm giảm thêm Entropy (IG <= 0)
        if best_gain <= 1e-6:
            return {
                "is_leaf": True,
                "pred_class": pred_class,
                "prob_1": prob_1,
                "n_samples": n_samples,
                "n_pos": n_pos,
                "entropy": round(current_h, 4),
            }

        rem_features = [f for f in avail_features if f != best_feat]
        branches = {}

        for val, sub_X in X_df.groupby(best_feat, sort=True):
            sub_y = y[sub_X.index.values]
            sub_X_reset = sub_X.reset_index(drop=True)
            branches[str(val)] = self._xay_dung_cay_de_quy(
                sub_X_reset, sub_y, rem_features, depth + 1
            )

        return {
            "is_leaf": False,
            "feature": best_feat,
            "info_gain": round(best_gain, 4),
            "entropy": round(current_h, 4),
            "prob_1": prob_1,
            "pred_class": pred_class,
            "n_samples": n_samples,
            "n_pos": n_pos,
            "branches": branches,
        }

    def fit(self, X_df, y):
        """Huấn luyện cây ID3 trên DataFrame rời rạc `X_df` và nhãn `y`."""
        X_reset = X_df.reset_index(drop=True).astype(str)
        y_arr = np.asarray(y, dtype=int).ravel()
        self.features_ = list(X_reset.columns)
        self.tree_ = self._xay_dung_cay_de_quy(X_reset, y_arr, self.features_, depth=0)
        return self

    def _du_doan_mot_mau_proba(self, node, row_dict):
        """Đi từ nút gốc xuống nút lá để lấy xác suất vỡ nợ P(y = 1 | x)."""
        if node["is_leaf"]:
            return node["prob_1"]

        feat = node["feature"]
        val = str(row_dict.get(feat, ""))
        if val in node["branches"]:
            return self._du_doan_mot_mau_proba(node["branches"][val], row_dict)
        # Nếu nhánh con chưa xuất hiện lúc huấn luyện -> trả về xác suất thực nghiệm của nút cha hiện tại
        return node["prob_1"]

    def predict_proba(self, X_df):
        """
        GIẢI THÍCH 5 ĐIỂM QUAN TRỌNG:
        1. Đoạn code thực hiện việc gì?
           Dự đoán xác suất thuộc lớp 0 và lớp 1 `[P(y=0|x), P(y=1|x)]` cho toàn bộ mẫu trong `X_df`.
        2. Tại sao cần hàm `predict_proba` bên cạnh `predict`?
           Để vẽ đường cong ROC và tính diện tích ROC-AUC ở Phần 3.5, cũng như xuất phần trăm
           rủi ro vỡ nợ cụ thể cho cán bộ ngân hàng ở hàm `predict_loan_status` Chương IV.
        3. Dữ liệu đầu vào (Input) là gì?
           DataFrame `X_df` đã được rời rạc hóa theo đúng ngưỡng của tập Train.
        4. Kết quả đầu ra (Output) là gì?
           Mảng NumPy 2 chiều kích thước `(N, 2)`: cột 0 là xác suất trả tốt, cột 1 là xác suất vỡ nợ.
        5. Hải Nam, Thành, Vũ cần hiểu cốt lõi điều gì để trả lời thầy Chiểu?
           Xác suất ở nút lá của cây quyết định chính là tỷ lệ mẫu lớp 1 rơi vào nút lá đó trên tập Train:
           `P(y=1 | lá) = (số khách vỡ nợ trong lá) / (tổng số khách trong lá)`.
        """
        records = X_df.astype(str).to_dict(orient="records")
        p1 = np.array([self._du_doan_mot_mau_proba(self.tree_, r) for r in records], dtype=float)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def predict(self, X_df, threshold=0.5):
        """Dự đoán nhãn phân lớp 0 hoặc 1 dựa trên ngưỡng xác suất (mặc định 0.5)."""
        proba_1 = self.predict_proba(X_df)[:, 1]
        return (proba_1 >= threshold).astype(int)

    def thong_ke_cau_truc_cay(self):
        """Đếm tổng số nút, số nút lá và độ sâu thực tế của cây ID3 đã học."""
        def _traverse(node, d):
            if node["is_leaf"]:
                return 1, 1, d
            total_nodes = 1
            total_leaves = 0
            max_d = d
            for child in node["branches"].values():
                n_c, l_c, d_c = _traverse(child, d + 1)
                total_nodes += n_c
                total_leaves += l_c
                if d_c > max_d:
                    max_d = d_c
            return total_nodes, total_leaves, max_d

        if self.tree_ is None:
            return {"total_nodes": 0, "n_leaves": 0, "actual_depth": 0}
        n_nodes, n_leaves, max_d = _traverse(self.tree_, 0)
        return {"total_nodes": n_nodes, "n_leaves": n_leaves, "actual_depth": max_d}

    def xuat_chuoi_cay(self, max_print_depth=2):
        """Xuất sơ đồ chữ các tầng đầu tiên của cây ID3 (giống phong cách `bt_id3_4.1.py` trên lớp)."""
        lines = []

        def _fmt(node, indent="", depth=0):
            if node["is_leaf"]:
                nhan = "1 - VỠ NỢ" if node["pred_class"] == 1 else "0 - TRẢ TỐT"
                lines.append(
                    f" -> [LÁ: {nhan} | P(Vỡ nợ)={node['prob_1']*100:.1f}% | n={node['n_samples']}]"
                )
                return
            if depth >= max_print_depth:
                lines.append(
                    f" -> [Nhánh tiếp theo hỏi `{node['feature']}` (IG={node['info_gain']:.4f}) "
                    f"| P(Vỡ nợ)={node['prob_1']*100:.1f}% | n={node['n_samples']} ...]"
                )
                return

            lines.append(
                f"\n{indent}[Nút hỏi `{node['feature']}` | H={node['entropy']:.4f} bit | "
                f"IG={node['info_gain']:.4f} bit | n={node['n_samples']}]"
            )
            for val, child in node["branches"].items():
                prefix = f"{indent}  ├── Nếu `{node['feature']}` = '{val}':"
                if child["is_leaf"] or (depth + 1 >= max_print_depth):
                    sub_lines = []
                    # Tạm thu kết quả 1 dòng
                    old_len = len(lines)
                    lines.append(prefix)
                    _fmt(child, indent + "  │     ", depth + 1)
                    merged = lines[old_len] + lines[old_len + 1]
                    lines[old_len:] = [merged]
                else:
                    lines.append(prefix)
                    _fmt(child, indent + "  │     ", depth + 1)

        if self.tree_ is not None:
            _fmt(self.tree_, indent="", depth=0)
        return "\n".join(lines).strip()


if __name__ == "__main__":
    import os

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_x_path = os.path.join(base_dir, "data", "processed", "X_train_id3.csv")
    test_x_path = os.path.join(base_dir, "data", "processed", "X_test_id3.csv")
    train_y_path = os.path.join(base_dir, "data", "processed", "y_train.csv")
    test_y_path = os.path.join(base_dir, "data", "processed", "y_test.csv")

    print("=" * 72)
    print("KIỂM THỬ TRỰC TIẾP MODULE CÂY QUYẾT ĐỊNH ĐA NHÁNH ID3 TỰ CÀI ĐẶT")
    print("=" * 72)
    if os.path.exists(train_x_path) and os.path.exists(train_y_path):
        X_tr = pd.read_csv(train_x_path)
        y_tr = pd.read_csv(train_y_path)["loan_status"].values
        X_te = pd.read_csv(test_x_path)
        y_te = pd.read_csv(test_y_path)["loan_status"].values

        id3_demo = ID3DecisionTree(max_depth=4, min_samples_split=25)
        id3_demo.fit(X_tr, y_tr)
        stats = id3_demo.thong_ke_cau_truc_cay()
        y_pred_te = id3_demo.predict(X_te)
        acc_te = float(np.mean(y_pred_te == y_te))

        print(f"- Entropy gốc H(S) trên tập Train (N={len(y_tr)}): {id3_demo.tinh_entropy(y_tr):.4f} bit")
        print("- Top 5 thuộc tính có Information Gain cao nhất tại Nút gốc:")
        for rank, (feat_name, ig_val) in enumerate(list(id3_demo.root_gains_.items())[:5], start=1):
            print(f"    {rank}. {feat_name:28s}: IG = {ig_val:.4f} bit")
        print(f"- Cấu trúc cây ID3 (max_depth=4): Tổng số nút = {stats['total_nodes']}, Số nút lá = {stats['n_leaves']}")
        print(f"- Độ chính xác trên tập Test độc lập (N={len(y_te)}): {acc_te*100:.2f}%")
        print("\n- Sơ đồ nhánh đầu tiên của cây ID3:")
        print(id3_demo.xuat_chuoi_cay(max_print_depth=1))
    else:
        print("Hãy chạy `01_tai_va_kham_pha_du_lieu.py` và `02_tien_xu_ly_du_lieu.py` trước để tạo dữ liệu.")


