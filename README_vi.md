<p align="center">
  <img src="https://raw.githubusercontent.com/nguyen-thinh15/mlxplain/main/docs/images/hero_banner.png" alt="mlxplain — Trí tuệ Nhân tạo Giải thích được cho các Quyết định Tín dụng" width="700"/>
</p>

<h1 align="center">mlxplain</h1>

<p align="center">
  <strong>Chuyển đổi bất kỳ mô hình Học máy nào thành một biên bản thẩm định tín dụng tuân thủ quy chuẩn — chỉ với 4 dòng mã.</strong>
</p>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Phiên bản Python"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="Giấy phép: MIT"></a>
  <a href="#-các-mô-hình-được-hỗ-trợ--bộ-dịch-xai"><img src="https://img.shields.io/badge/models-LogReg%20%7C%20Trees%20%7C%20XGBoost-teal.svg" alt="Mô hình Hỗ trợ"></a>
</p>

<p align="center">
  <a href="README.md">🇺🇸 English</a> • 🇻🇳 <strong>Tiếng Việt</strong>
</p>

---

## 🔥 Những gì bạn nhận được

**Chỉ một lệnh gọi hàm. Kết quả đầu ra chuyên nghiệp đa chiều.** Không cần cấu hình phức tạp.

<p align="center">
  <img src="https://raw.githubusercontent.com/nguyen-thinh15/mlxplain/main/docs/images/workflow_diagram.png" alt="Quy trình hoạt động mlxplain: Huấn luyện → Giải thích → Kết quả" width="650"/>
</p>

### 💻 Điểm nổi bật: Hồ sơ Rủi ro HTML Cao cấp

Bên cạnh các biên bản dạng văn bản và biểu đồ tiêu chuẩn, `mlxplain` tạo ra một **Hồ sơ Rủi ro Tín dụng HTML (Credit Risk Dossier)** hoàn toàn độc lập, mang phong cách kính mờ (glassmorphism) sang trọng với các hình vẽ vector SVG độ phân giải cao được nhúng trực tiếp. Điều này cung cấp một bảng điều khiển (dashboard) tương tác di động, sẵn sàng đáp ứng mọi yêu cầu tuân thủ quy chuẩn ngay tức thì!

### 📊 Biểu đồ Chẩn đoán Tự động tạo

Mỗi cuộc gọi tới `explain_risk()` đều tạo ra ba biểu đồ `matplotlib` sắc nét sẵn sàng để xuất bản:

<table>
  <tr>
    <td align="center"><strong>Thước đo Xác suất</strong><br/><em>Xem quyết định trong nháy mắt</em></td>
    <td align="center"><strong>Nhân tố tác động (Thác nước)</strong><br/><em>Điều gì đã thúc đẩy dự đoán?</em></td>
    <td align="center"><strong>Thanh phản thực tế</strong><br/><em>Những gì cần phải thay đổi?</em></td>
  </tr>
  <tr>
    <td><img src="https://raw.githubusercontent.com/nguyen-thinh15/mlxplain/main/docs/images/logistic_gauge.png" alt="Thước đo Xác suất" width="350"/></td>
    <td><img src="https://raw.githubusercontent.com/nguyen-thinh15/mlxplain/main/docs/images/logistic_drivers.png" alt="Nhân tố tác động" width="350"/></td>
    <td><img src="https://raw.githubusercontent.com/nguyen-thinh15/mlxplain/main/docs/images/logistic_counterfactuals.png" alt="Thay đổi Phản thực tế" width="350"/></td>
  </tr>
</table>

### 📝 Biên bản Thẩm định Tín dụng Tự động tạo

Đối với các hồ sơ bị từ chối, **mlxplain** tạo ra một văn bản tóm tắt sẵn sàng đáp ứng yêu cầu tuân thủ với quyết định cụ thể, xác suất tương ứng, xếp hạng các yếu tố rủi ro và các phương án khắc phục khả thi (cure paths):

```text
============================================================
BIÊN BẢN THẨM ĐỊNH TÍN DỤNG (XGBoost SHAP / RandomForest XAI)
============================================================
QUYẾT ĐỊNH TÍN DỤNG: Từ chối (Declined)
Xác suất Nợ xấu: 65.7% (ngưỡng: 45.0%)

YẾU TỐ RỦI RO (RISK FACTORS):
  - Lịch sử tín dụng xấu: 2 (mức độ ảnh hưởng: 0.1789)
  - Điểm tín dụng (FICO): 612.7 (mức độ ảnh hưởng: 0.1597)
  - Tỷ lệ nợ trên thu nhập (%): 42.54 (mức độ ảnh hưởng: 0.1022)
  - Tỷ lệ khoản vay trên giá trị tài sản (%): 92.13 (mức độ ảnh hưởng: 0.08839)
  - Thu nhập hàng năm ($k): 68.41 (mức độ ảnh hưởng: 0.05231)
  - Tỷ lệ sử dụng thẻ tín dụng (%): 25.97 (mức độ ảnh hưởng: 0.03185)
  - Thời gian làm việc (năm): 5.625 (mức độ ảnh hưởng: 0.005195)
  - Số lượng hạn mức tín dụng đang mở: 6 (mức độ ảnh hưởng: 0.001935)
  - Số tiền vay yêu cầu ($k): 31.06 (mức độ ảnh hưởng: 0.0003766)

YẾU TỐ GIẢM THIỂU RỦI RO (MITIGATING FACTORS):
  + Số dư tiết kiệm ($k): 27.25 (mức độ ảnh hưởng: 0.06112)
  + Tỷ lệ thanh toán đúng hạn (%): 99.89 (mức độ ảnh hưởng: 0.01372)
  + Số năm lịch sử tín dụng: 10.78 (mức độ ảnh hưởng: 8.391e-05)

PHƯƠNG ÁN KHẮC PHỤC (CURE PATHS - thay đổi cần thiết để được duyệt):
  → Lịch sử tín dụng xấu: giảm từ 2 xuống 0.92
  → Số dư tiết kiệm ($k): tăng từ 27.25 lên 30.52
  → Tỷ lệ sử dụng thẻ tín dụng (%): giảm từ 25.97 xuống 16.62
  → Tỷ lệ nợ trên thu nhập (%): giảm từ 42.54 xuống 29.78
  → Tỷ lệ khoản vay trên giá trị tài sản (%): giảm từ 92.13 xuống 75.54
  → Điểm tín dụng (FICO): tăng từ 612.7 lên 686.2
============================================================
```

---

## ⚡ Khởi đầu nhanh (4 Dòng mã)

```python
from sklearn.ensemble import RandomForestClassifier
from mlxplain import explain_risk

# 1. Huấn luyện mô hình học máy tiêu chuẩn (ví dụ: mô hình chấm điểm tín dụng 12 đặc trưng)
model = RandomForestClassifier().fit(X_train, y_train)

# 2. Định nghĩa 12 thuộc tính dữ liệu toàn diện
feature_names = [
    "Thu nhập hàng năm ($k)", "Tỷ lệ nợ trên thu nhập (%)", "Điểm tín dụng (FICO)",
    "Thời gian làm việc (năm)", "Số dư tiết kiệm ($k)", "Số tiền vay yêu cầu ($k)",
    "Tỷ lệ khoản vay trên giá trị tài sản (%)", "Số hạn mức tín dụng đang mở", "Tỷ lệ thanh toán đúng hạn (%)",
    "Lịch sử tín dụng xấu", "Tỷ lệ sử dụng thẻ tín dụng (%)", "Số năm lịch sử tín dụng"
]

# 3. Tạo báo cáo giải thích trực quan & tuân thủ quy chuẩn chỉ với 1 dòng lệnh!
report = explain_risk(model, X_train, idx=10, feature_names=feature_names, threshold=0.45)

# 4. In biên bản thẩm định tín dụng chuyên nghiệp ra màn hình
print(report.summary)

# 5. Lưu biểu đồ chẩn đoán định dạng vector SVG sắc nét
report.figures["gauge"].savefig("gauge.svg")
report.figures["drivers"].savefig("drivers.svg")
report.figures["counterfactuals"].savefig("counterfactuals.svg")
```

---

## 💻 Các mô hình được hỗ trợ & Bộ dịch XAI

**mlxplain** hoạt động tương thích với các họ mô hình phổ biến nhất hiện nay mà không cần bất kỳ thay đổi cấu hình nào:

| Họ mô hình | Phương pháp trích xuất XAI | Chiến thuật Phản thực tế (Counterfactual) |
| :--- | :--- | :--- |
| **Hồi quy Logistic** | Trọng số hệ số × giá trị đặc trưng | **Phân tích (Analytical):** Nghịch đảo toán học chính xác |
| **Cây quyết định & Rừng ngẫu nhiên** | Chênh lệch xác suất lớp ở cấp độ phân tách dọc theo các đường đi quyết định | **Nhiễu loạn (Perturbation):** Tìm kiếm không gian ranh giới phân tách tuần tự |
| **Ensemble Boosting** *(XGBoost & LightGBM)* | Giá trị SHAP (Shapley Additive exPlanations) | **Nhiễu loạn (Perturbation):** Tìm kiếm ranh giới giới hạn trong tập mẫu |
| **Phát hiện Bất thường** *(Isolation Forest)* | SHAP `TreeExplainer` trên cây cô lập | **Nhiễu loạn (Perturbation):** Tìm kiếm ranh giới phân tách xác suất bất thường |
| **Phân cụm** *(K-Means)* | Sự khác biệt khoảng cách không gian so với trọng tâm mục tiêu | **Phân tích (Analytical):** Chiếu hình học nửa không gian L2 chính xác |
| **Học sâu** *(Mạng thần kinh)* | LIME / Integrated Gradients | *(Lên kế hoạch / Tạm hoãn)* |

### Biểu đồ trên tất cả các loại mô hình

<table>
  <tr>
    <th></th>
    <th align="center">Hồi quy Logistic</th>
    <th align="center">Cây quyết định</th>
    <th align="center">XGBoost (SHAP)</th>
  </tr>
  <tr>
    <td><strong>Nhân tố tác động</strong></td>
    <td><img src="https://raw.githubusercontent.com/nguyen-thinh15/mlxplain/main/docs/images/logistic_drivers.png" alt="Nhân tố hồi quy logistic" width="280"/></td>
    <td><img src="https://raw.githubusercontent.com/nguyen-thinh15/mlxplain/main/docs/images/tree_drivers.png" alt="Nhân tố cây quyết định" width="280"/></td>
    <td><img src="https://raw.githubusercontent.com/nguyen-thinh15/mlxplain/main/docs/images/ensemble_drivers.png" alt="Nhân tố Ensemble" width="280"/></td>
  </tr>
</table>

---

## 🎯 Các tính năng cốt lõi

Đối với bất kỳ mô hình phân loại nhị phân nào được hỗ trợ, **mlxplain** mang lại ba trụ cột giải thích:

* **Quyết định (The Decision):** Phân loại dự đoán thành nhãn rõ ràng (ví dụ: *Approved/Declined*) dựa trên ngưỡng xác suất có thể tùy biến cấu hình.
* **Nhân tố tác động (The Drivers):** Xác định và xếp hạng các đặc trưng đã thúc đẩy dự đoán theo từng hướng, sắp xếp độc lập dựa trên mức độ tác động toán học.
* **Phản thực tế (The Counterfactuals):** Đối với các kết quả bất lợi, tính toán mức điều chỉnh đặc trưng tối thiểu cần thiết để đảo ngược dự đoán (ví dụ: *"Giảm số tiền yêu cầu vay bớt $5,000"*).
* **Hiệu năng thông minh (Smart Performance):** Các phản thực tế chỉ được tính toán đối với các dự đoán bất lợi (xác suất ≥ ngưỡng) để tiết kiệm chu kỳ CPU cho các quyết định thuận lợi.

---

## 🧩 Kiến trúc miền có thể cắm (Pluggable Domain)

Lõi của **mlxplain** nhận biết được mô hình nhưng hoàn toàn độc lập với các bài toán nghiệp vụ kinh doanh. Nó tách biệt rõ ràng giữa **những gì mô hình nhìn thấy** và **ý nghĩa của chúng trong kinh doanh**:

```
mlxplain/
├── core/              # Cấu trúc dữ liệu, logic phân ngưỡng, phản thực tế
├── translators/       # Trích xuất đặc trưng đặc thù mô hình (độc lập nghiệp vụ)
├── translators_vi/    # Phiên bản tiếng Việt của bộ dịch đặc thù mô hình (docstrings/comments Việt hóa)
├── visualizations/    # Các biểu đồ matplotlib tiêu chuẩn
├── domains/           # Bộ giải thích miền nghiệp vụ có thể cắm tự do
│   └── credit_risk/   # Nghiệp vụ rủi ro tín dụng: Approved/Declined, Risk Factors, Cure Paths
└── engine.py          # Điểm truy cập API hợp nhất
```

Để hỗ trợ một miền nghiệp vụ mới (ví dụ: y tế `healthcare` hoặc phát hiện gian lận `fraud_detection`), bạn chỉ cần tạo một bộ giải thích miền mới kế thừa từ `BaseDomain`. **Các bộ dịch, công cụ toán học và mã vẽ biểu đồ sẽ hoàn toàn được giữ nguyên 100% không bị ảnh hưởng.**

### 🏦 Ánh xạ miền Rủi ro Tín dụng

Bộ giải thích miền rủi ro tín dụng ánh xạ các khái niệm toán học trừu tượng thành các thuật ngữ tài chính ngân hàng:

| Khái niệm Toán học | Thuật ngữ Kinh doanh Rủi ro Tín dụng |
| :--- | :--- |
| Dự đoán Tích cực (Positive) | **Declined** (Từ chối - Xác suất nợ xấu cao) |
| Dự đoán Tiêu cực (Negative) | **Approved** (Duyệt - Xác suất nợ xấu thấp) |
| Nhân tố tác động Tích cực | **Risk Factors** (Yêu tố rủi ro / Điểm yếu) |
| Nhân tố tác động Tiêu cực | **Mitigating Factors** (Yếu tố giảm thiểu rủi ro / Điểm mạnh) |
| Phản thực tế | **Cure Paths** (Phương án khắc phục để được duyệt) |

---

## 🚀 Các ví dụ thực thi được

Thư mục `examples/` chứa các mã nguồn hoàn chỉnh, sẵn sàng chạy để thể hiện khả năng của **mlxplain** từ đầu đến cuối. Chúng tạo ra dữ liệu tín dụng giả lập, huấn luyện mô hình, xuất báo cáo và lưu lại các đồ thị chẩn đoán.

Chạy trực tiếp các mã nguồn này bằng `uv`:

```bash
# 1. Chạy ví dụ Hồi quy Logistic Rủi ro Tín dụng
uv run python examples/01_logistic_credit_risk.py

# 2. Chạy ví dụ Cây quyết định Rủi ro Tín dụng
uv run python examples/02_decision_tree_credit_risk.py

# 3. Chạy ví dụ XGBoost dựa trên SHAP Rủi ro Tín dụng
uv run python examples/03_ensemble_credit_risk.py

# 4. Chạy ví dụ Nâng cao 12 Đặc trưng sản sinh Hồ sơ HTML
uv run python examples/04_advanced_credit_risk.py

# 5. Chạy ví dụ Phát hiện Bất thường không giám sát
uv run python examples/05_anomaly_detection.py

# 6. Chạy ví dụ Phân cụm KMeans phân khúc khách hàng
uv run python examples/06_kmeans_clustering.py
```

Tất cả các ví dụ sẽ lưu biểu đồ được tạo vào thư mục `examples/output/`. Ví dụ nâng cao cũng sẽ tạo tệp `dossier.html` trong thư mục đó — hãy mở nó bằng trình duyệt của bạn để trải nghiệm bảng điều khiển tương tác kính mờ cao cấp!

---

## 🌀 Giải thích Mô hình Học máy Không giám sát

**mlxplain** là thư viện giải thích tổng quát đầu tiên hợp nhất các mô hình phân loại có giám sát với **học máy không giám sát XAI** (Phát hiện Bất thường & Phân cụm) dưới cùng một tiêu chuẩn trình bày biểu đồ trực quan và báo cáo cấu trúc cao cấp!

### 1. Phát hiện Bất thường (qua `IsolationForest`)
* **Chuẩn hóa điểm số**: Chúng tôi chuẩn hóa điểm số bất thường của Isolation Forest từ scikit-learn về đoạn $[0, 1]$, xử lý chính xác như một xác suất với ngưỡng mặc định `0.5`. Tích hợp hoàn toàn qua API `explain()` thống nhất.
* **SHAP Drivers**: Sử dụng SHAP `TreeExplainer` trên các cây cô lập để trích xuất các đặc trưng đẩy mẫu dữ liệu vào trạng thái bất thường hay bình thường.
* **Cải thiện Phản thực tế**: Sử dụng thuật toán tìm kiếm nhiễu loạn giới hạn mẫu để xác định các thay đổi thuộc tính tối thiểu giúp khôi phục hệ thống từ trạng thái bất thường về bình thường.

### 2. Phân cụm (qua `KMeans`)
* **Endpoint riêng biệt**: Cung cấp hàm `explain_cluster()` để giải thích quyết định phân cụm của K-Means so với cụm á quân (cận kề thứ 2) hoặc một cụm mục tiêu do người dùng chỉ định.
* **Nhân tố khoảng cách**: Đo lường đóng góp của từng đặc trưng trong việc giữ mẫu dữ liệu gần với trọng tâm được gán $c$ hơn trọng tâm mục tiêu $t$:
  $$\text{impact}_i = (x_i - t_i)^2 - (x_i - c_i)^2$$
* **Lộ trình nâng cấp Phản thực tế đóng**: Sử dụng công thức toán học **chiếu hình học L2 chính xác** để tính toán tức thì các thay đổi thuộc tính tối thiểu giúp nâng cấp/chuyển dịch phân khúc khách hàng mà không cần vòng lặp!
* **Thước đo cụm tương đồng**: Biến đổi vectơ khoảng cách Euclidean thành điểm số tương đồng $[0, 1]$ ($p = d_t^2 / (d_c^2 + d_t^2)$) giúp tái sử dụng hoàn toàn biểu đồ thước đo trực quan tiêu chuẩn.

---

## 📦 Cài đặt

Dự án này tương thích hoàn toàn với [uv](https://github.com/astral-sh/uv) để quản lý gói nhanh như chớp.

### Dành cho người sử dụng

Cài đặt `mlxplain` trực tiếp vào môi trường ảo của bạn:

```bash
uv pip install mlxplain
# hoặc sử dụng pip tiêu chuẩn
pip install mlxplain
```

Hoặc cài đặt các thư viện phụ thuộc bằng tệp `requirements.txt` có sẵn:

```bash
pip install -r requirements.txt
pip install -e .
```

### Dành cho nhà phát triển

Thiết lập dự án cục bộ để phát triển:

1. Nhân bản kho chứa:
   ```bash
   git clone https://github.com/nguyen-thinh15/mlxplain.git
   cd mlxplain
   ```
2. Tạo và đồng bộ hóa môi trường ảo:
   ```bash
   uv venv --python 3.10
   source .venv/bin/activate
   uv sync --all-extras
   ```
3. Chạy bộ kiểm thử:
   ```bash
   uv run pytest tests/ -v
   ```

**Các thư viện phụ thuộc chính:** `numpy`, `scikit-learn`, `shap`, `matplotlib`
**Các thư viện phụ thuộc tùy chọn (cho mô hình ensemble):** `xgboost`, `lightgbm`
