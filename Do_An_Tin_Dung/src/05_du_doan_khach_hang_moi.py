"""
CHƯƠNG IV: XÂY DỰNG CHƯƠNG TRÌNH DỰ ĐOÁN KHẢ NĂNG TRẢ NỢ CHO KHÁCH HÀNG MỚI
Cung cấp hàm chuẩn `predict_loan_status(customer_data)` đi qua đúng Pipeline tiền xử lý
đã học từ tập Train (`preprocessor_bundle.joblib`) để tránh tuyệt đối sai lệch dữ liệu
(Training-Serving Skew) và chạy minh họa với các hồ sơ khách hàng thực tế.

Đồ án Học máy - Đề tài: Dự đoán khả năng trả nợ của khách hàng vay vốn
Thành viên phụ trách chính: Vũ
Giảng viên hướng dẫn: Thầy Chiểu - Khoa CNTT, Trường Đại học Thủy Lợi
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from id3_tu_cai_dat import ID3DecisionTree  # noqa: F401, E402

BASE_DIR = os.path.dirname(SRC_DIR)
MODELS_DIR = os.path.join(BASE_DIR, "outputs", "models")
TABLES_DIR = os.path.join(BASE_DIR, "outputs", "tables")

MODEL_FILES = {
    "logistic_regression": ("Logistic Regression", "logistic_regression.joblib"),
    "id3": ("Decision Tree - ID3", "id3_decision_tree.joblib"),
    "cart": ("Decision Tree - CART", "cart_decision_tree.joblib"),
    "random_forest": ("Random Forest", "random_forest.joblib"),
}


def tien_xu_ly_mot_khach_hang(customer_data, bundle):
    """
    GIẢI THÍCH 5 ĐIỂM QUAN TRỌNG:
    1. Đoạn code thực hiện việc gì?
       Nhận dictionary thông tin thô của 1 khách hàng mới và biến đổi qua ĐÚNG các bộ tham số
       đã học trên tập Train (`num_imputer`, `cat_imputer`, `onehot_encoder`, `scaler`, `id3_bin_edges`).
    2. Tại sao không được gọi `.fit()` lại hay tính lại trung bình/phân vị trên khách hàng mới?
       Một khách hàng đơn lẻ không có độ lệch chuẩn hay tứ phân vị; hơn nữa mô hình học máy chỉ
       hiểu đúng hệ quy chiếu (thang đo Z-score, thứ tự cột One-Hot, mốc chia khoảng ID3) của tập Train!
    3. Dữ liệu đầu vào (Input) là gì?
       `customer_data` (dict thông tin 11 thuộc tính khách hàng) và `bundle` (`preprocessor_bundle.joblib`).
    4. Kết quả đầu ra (Output) là gì?
       Bộ 3 DataFrame 1 dòng: `(X_logreg, X_tree, X_id3)` khớp chính xác định dạng của 4 mô hình.
    5. Hải Nam, Thành, Vũ cần hiểu cốt lõi điều gì để trả lời thầy Chiểu?
       Nếu hồ sơ khách hàng mới bị bỏ trống lãi suất hay số năm làm việc (`None` / `NaN`), hàm này
       tự động điền đúng Median của tập Train (`10.99%` và `4.0` năm) mà không bao giờ báo lỗi!
    """
    data = dict(customer_data)

    # Tự động tính tỷ lệ vay / thu nhập nếu người dùng chỉ nhập số tiền vay & thu nhập
    if data.get("loan_percent_income") is None and data.get("loan_amnt") and data.get("person_income"):
        data["loan_percent_income"] = round(float(data["loan_amnt"]) / float(data["person_income"]), 4)

    num_cols = bundle["num_cols"]
    cat_cols = bundle["cat_cols"]
    nominal_cols = bundle["nominal_cols"]
    ordinal_col = bundle["ordinal_col"]
    binary_col = bundle["binary_col"]

    # Tạo DataFrame 1 dòng theo đúng thứ tự cột
    df_single = pd.DataFrame([data])
    for col in num_cols:
        if col not in df_single.columns:
            df_single[col] = np.nan
        df_single[col] = pd.to_numeric(df_single[col], errors="coerce")

    for col in cat_cols:
        if col not in df_single.columns or pd.isna(df_single.loc[0, col]):
            df_single[col] = np.nan
        else:
            df_single[col] = str(df_single.loc[0, col]).strip().upper()

    # Kiểm tra hợp lệ cơ bản
    age_val = df_single.loc[0, "person_age"]
    if pd.notna(age_val) and not (18 <= age_val <= 100):
        raise ValueError(f"Tuổi khách hàng ({age_val}) nằm ngoài phạm vi hợp lệ [18, 100]!")

    # Bước 1: Điền khuyết bằng Imputer đã fit trên tập Train
    X_num = pd.DataFrame(bundle["num_imputer"].transform(df_single[num_cols]), columns=num_cols)
    X_cat = pd.DataFrame(bundle["cat_imputer"].transform(df_single[cat_cols]), columns=cat_cols)
    X_imp = pd.concat([X_num, X_cat], axis=1)

    # Bước 2: Mã hóa cho CART, Random Forest và chuẩn hóa cho Logistic Regression
    ohe_arr = bundle["onehot_encoder"].transform(X_imp[nominal_cols])
    ohe_cols = bundle["onehot_encoder"].get_feature_names_out(nominal_cols).tolist()

    X_tree = X_imp[num_cols].copy()
    X_tree[ordinal_col] = X_imp[ordinal_col].map(bundle["grade_mapping"]).fillna(2.0).astype(float)
    X_tree[binary_col] = X_imp[binary_col].map(bundle["binary_mapping"]).fillna(0.0).astype(float)
    X_ohe = pd.DataFrame(ohe_arr, columns=ohe_cols)
    X_tree = pd.concat([X_tree, X_ohe], axis=1)[bundle["tree_feature_names"]]

    X_logreg = X_tree.copy()
    scale_cols = bundle["scale_cols"]
    X_logreg[scale_cols] = bundle["scaler"].transform(X_tree[scale_cols])

    # Bước 3: Rời rạc hóa theo đúng ngưỡng tứ phân vị tập Train cho cây ID3
    X_id3 = pd.DataFrame(index=[0])
    for col in num_cols:
        edges, labels = bundle["id3_bin_edges"][col]
        X_id3[col] = pd.cut(X_imp[col], bins=edges, labels=labels).astype(str)
    for col in cat_cols:
        X_id3[col] = X_imp[col].astype(str)
    X_id3 = X_id3[bundle["id3_feature_names"]]

    return X_logreg, X_tree, X_id3


def predict_loan_status(customer_data, model_name="random_forest", threshold=0.5):
    """
    GIẢI THÍCH 5 ĐIỂM QUAN TRỌNG:
    1. Đoạn code thực hiện việc gì?
       Dự đoán trạng thái khoản vay (0 = Trả nợ tốt, 1 = Vỡ nợ) và xác suất rủi ro vỡ nợ
       cho 1 khách hàng mới bằng mô hình đã lưu (mặc định là mô hình tốt nhất `random_forest`).
    2. Tại sao hàm trả về cả `default_probability` bên cạnh `predicted_label`?
       Trong thực tế ngân hàng, hai khách hàng cùng có nhãn 0 nhưng người có rủi ro 3% sẽ được
       duyệt tự động với hạn mức tốt, còn người có rủi ro 46% cần thẩm định viên xem xét thêm!
    3. Dữ liệu đầu vào (Input) là gì?
       `customer_data` (dict hồ sơ khách hàng), `model_name` ('random_forest', 'cart', 'id3',
       hoặc 'logistic_regression'), và ngưỡng quyết định `threshold` (mặc định 0.5).
    4. Kết quả đầu ra (Output) là gì?
       Dictionary chứa nhãn dự đoán (0/1), xác suất vỡ nợ (%), mức độ rủi ro và khuyến nghị tín dụng.
    5. Hải Nam, Thành, Vũ cần hiểu cốt lõi điều gì để trả lời thầy Chiểu?
       Nếu ngân hàng muốn thắt chặt rủi ro mùa suy thoái kinh tế (tăng Recall để bớt bỏ lọt nợ xấu FN),
       chỉ cần hạ tham số `threshold` từ `0.5` xuống `0.35` mà không cần huấn luyện lại mô hình!
    """
    if model_name not in MODEL_FILES:
        raise ValueError(f"model_name phải thuộc {list(MODEL_FILES.keys())}")

    bundle = joblib.load(os.path.join(MODELS_DIR, "preprocessor_bundle.joblib"))
    display_name, filename = MODEL_FILES[model_name]
    model = joblib.load(os.path.join(MODELS_DIR, filename))

    X_logreg, X_tree, X_id3 = tien_xu_ly_mot_khach_hang(customer_data, bundle)
    if model_name == "logistic_regression":
        X_in = X_logreg
    elif model_name == "id3":
        X_in = X_id3
    else:
        X_in = X_tree

    prob_default = float(model.predict_proba(X_in)[0, 1])
    pred_label = int(prob_default >= threshold)

    if prob_default < 0.20:
        risk_tier = "Rủi ro THẤP"
    elif prob_default < 0.50:
        risk_tier = "Rủi ro TRUNG BÌNH"
    else:
        risk_tier = "Rủi ro CAO"

    recommendation = (
        "0 - TRẢ NỢ TỐT (Đề xuất: DUYỆT KHOẢN VAY)"
        if pred_label == 0
        else "1 - NGUY CƠ VỠ NỢ (Đề xuất: TỪ CHỐI / YÊU CẦU TÀI SẢN ĐẢM BẢO)"
    )

    return {
        "model_used": display_name,
        "predicted_label": pred_label,
        "default_probability": round(prob_default, 4),
        "default_probability_pct": f"{prob_default * 100:.2f}%",
        "risk_level": risk_tier,
        "decision_text": recommendation,
    }


def main():
    os.makedirs(TABLES_DIR, exist_ok=True)

    # 3 Kịch bản hồ sơ khách hàng thực tế để minh họa trong Chương IV báo cáo
    sample_customers = [
        (
            "Hồ sơ 1: Khách hàng tốt (Thu nhập ổn định, sở hữu nhà, hạng A)",
            {
                "person_age": 30,
                "person_income": 85000,
                "person_home_ownership": "OWN",
                "person_emp_length": 7.0,
                "loan_intent": "EDUCATION",
                "loan_grade": "A",
                "loan_amnt": 8000,
                "loan_int_rate": 7.50,
                "loan_percent_income": 0.09,
                "cb_person_default_on_file": "N",
                "cb_person_cred_hist_length": 8,
            },
        ),
        (
            "Hồ sơ 2: Khách hàng rủi ro cao (Nhà thuê, vay 42% thu nhập để đảo nợ, hạng E)",
            {
                "person_age": 22,
                "person_income": 24000,
                "person_home_ownership": "RENT",
                "person_emp_length": 1.0,
                "loan_intent": "DEBTCONSOLIDATION",
                "loan_grade": "E",
                "loan_amnt": 10000,
                "loan_int_rate": 16.80,
                "loan_percent_income": 0.42,
                "cb_person_default_on_file": "Y",
                "cb_person_cred_hist_length": 2,
            },
        ),
        (
            "Hồ sơ 3: Hồ sơ khuyết thông tin lãi suất & năm đi làm (Kiểm chứng tự động Impute)",
            {
                "person_age": 26,
                "person_income": 48000,
                "person_home_ownership": "MORTGAGE",
                "person_emp_length": np.nan,  # Bị bỏ trống -> Pipeline tự điền Median = 4.0 năm
                "loan_intent": "MEDICAL",
                "loan_grade": "B",
                "loan_amnt": 9600,
                "loan_int_rate": np.nan,      # Bị bỏ trống -> Pipeline tự điền Median = 10.99%
                "loan_percent_income": 0.20,
                "cb_person_default_on_file": "N",
                "cb_person_cred_hist_length": 4,
            },
        ),
    ]

    print("\n" + "=" * 86)
    print("CHƯƠNG IV: CHẠY THỬ CHƯƠNG TRÌNH DỰ ĐOÁN KHÁCH HÀNG MỚI (`predict_loan_status`)")
    print("=" * 86)

    demo_rows = []
    for title, cust in sample_customers:
        print(f"\n>>> {title}")
        print(f"    Thông tin đầu vào: {cust}")
        for m_key in ["logistic_regression", "id3", "cart", "random_forest"]:
            res = predict_loan_status(cust, model_name=m_key)
            print(
                f"    - [{res['model_used']:22s}] => Xác suất vỡ nợ: {res['default_probability_pct']:>6s} "
                f"({res['risk_level']:18s}) | Kết luận: {res['decision_text']}"
            )
            demo_rows.append({
                "Ho_So_Khach_Hang": title,
                "Mo_Hinh": res["model_used"],
                "Nhan_Du_Doan": res["predicted_label"],
                "Xac_Suat_Vo_No_Pct": res["default_probability_pct"],
                "Muc_Do_Rui_Ro": res["risk_level"],
                "Khuyen_Nghi_Tin_Dung": res["decision_text"],
            })

    df_demo = pd.DataFrame(demo_rows)
    out_csv = os.path.join(TABLES_DIR, "05_ket_qua_du_doan_ho_so_mau.csv")
    df_demo.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\n[OK] Đã xuất bảng kết quả chạy thử Chương IV tại: {out_csv}")


if __name__ == "__main__":
    main()

