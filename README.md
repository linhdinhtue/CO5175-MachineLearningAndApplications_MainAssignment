# Interpretable Regression & Explainability

Code thực nghiệm và kết quả của Thủy đã được merge về dự án. Repository hiện phục vụ ba việc: **đọc hiểu phương pháp, chạy tái hiện độc lập và viết báo cáo theo mục 4 của đề**. Phần proposal lịch sử được giữ bên dưới; trạng thái triển khai không còn là “chưa có kết quả”.

| Bắt đầu ở đâu? | Tài liệu / lệnh |
| --- | --- |
| Học code theo luồng dữ liệu và mục 4 | [Learning guide](docs/learning_guide.md) |
| Cấu trúc subfolder và thay đổi đường dẫn | [Reorganisation](docs/reorganisation.md) |
| Phân tích chi tiết của nhóm | [Analysis and discussion](docs/analysis_and_discussion.md) |
| Bản thảo report theo cấu trúc đề | [Report](reports/report.md) |
| Bảng số liệu có kiểm tra dự đoán | [Evidence](reports/evidence.md), sinh bằng `python -m reports.build_evidence` |
| Kiểm tra code nhanh | `python -m unittest discover -s tests -v` |
| Lập kế hoạch chạy lại | `python -m experiments.run_suite --suite baseline --run-id my-baseline-01` |
| Chạy kế hoạch | Thêm `--execute`; kết quả vào `results/reproductions/<run-id>/` |

Giữ nguyên artifact của nhóm trong `results/*-<run-number>/`. Chạy lại với ID mới, so config/checksum/metric trước khi đưa số liệu mới vào báo cáo. Các config `explain`, `stability`, `analysis` dùng hyperparameter cố định từ lần chạy cũ; xem hướng dẫn khi tuning lại chọn cấu hình khác. `configs/suites.json` chứa ba bộ `baseline`, `core`, `full`; `configs/ci_runs.json` đang rỗng nên CI push hiện không chạy thí nghiệm nào.

## Proposal và protocol của nhóm

**Học phần:** Machine Learning and Applications · **Chủ đề:** Topic 1
**Trạng thái:** Nội dung proposal/protocol được lưu để đối chiếu với code. Đã có artifact thực nghiệm; việc proposal đã được nộp/duyệt cần nhóm xác nhận riêng, không suy ra từ trạng thái code.
**Thành viên / MSSV / trưởng nhóm:**

| Mã số ID | Họ |Tên | Email | Thành viên |
| --- | --- | --- | --- | --- |
| 2570441 | Đinh Trương Tuệ  | Linh | linh.dinhtruongtue@hcmut.edu.vn | Trưởng Nhóm |
| 2570508 | Đinh Thị Thu | Thủy | thuy.dinhthithu273@hcmut.edu.vn |  Thành viên |
| 2580869 | Hứa Phước | Trường | truong.huaphuoc@hcmut.edu.vn | Thành viên |

## 1. Bài toán và mục tiêu

Nghiên cứu đánh đổi giữa hiệu năng dự đoán và khả năng diễn giải khi so sánh Logistic Regression với regularization L1, L2, Elastic Net và Gradient Boosting Decision Trees (GBDT). Dataset đề xuất là **UCI Adult**, với nhãn nhị phân thu nhập trên 50.000 USD/năm. Đây là bài toán phân loại dùng Logistic Regression, phù hợp mô tả Topic 1; không dùng metric hồi quy như RMSE.

Ba câu hỏi nghiên cứu: (RQ1) GBDT cải thiện hiệu năng bao nhiêu so với Logistic Regression? (RQ2) Regularization ảnh hưởng thế nào đến hiệu năng và độ thưa của hệ số? (RQ3) Hệ số và SHAP cung cấp thông tin gì về dự đoán, và giới hạn của mỗi cách giải thích là gì? Không đặt trước kết luận mô hình nào thắng.

## 2. Paper chính và phương pháp

**Paper chính:** Lundberg & Lee (2017), *A Unified Approach to Interpreting Model Predictions* [1]. Phạm vi tái hiện đề xuất là mô hình giải thích cộng tính và một thí nghiệm kiểm tra local accuracy trên bài toán nhỏ có thể tính Shapley chính xác; sau đó áp dụng giải thích lên GBDT trên Adult. Phần Adult là thực nghiệm ứng dụng, không được gọi là tái hiện nguyên vẹn benchmark của paper. Khi đọc paper ở tuần 2, phải chốt hình/bảng/thí nghiệm gốc để đối chiếu, thiết lập tương ứng và số liệu tham chiếu trước khi chạy; nếu không tái lập được, báo cáo rõ giới hạn này.

Baseline ban đầu: Logistic Regression L2, `C=1`. Kết quả chính: Logistic Regression L1/L2/Elastic Net được tuning và `GradientBoostingClassifier`. GAM/EBM là mở rộng sau yêu cầu tối thiểu.

Với xác suất $p_i=\sigma(b+x_i^T\beta)$, mục tiêu lý thuyết là:

$$
\min_{b,\beta}\; -\frac{1}{n}\sum_i
\left[y_i\log p_i+(1-y_i)\log(1-p_i)\right]
+\lambda\left[
\alpha\|\beta\|_1+
\frac{1-\alpha}{2}\|\beta\|_2^2
\right].
$$

L1 ứng với \(\alpha=1\), L2 ứng với \(\alpha=0\); intercept không bị phạt. Dự kiến dùng solver SAGA; `C` điều khiển nghịch độ mạnh regularization, cần đối chiếu chuẩn hóa loss của phiên bản thư viện khi liên hệ với \(\lambda\). GBDT học cộng dồn cây để giảm log-loss. SHAP phân rã đầu ra thành giá trị nền và đóng góp đặc trưng; giải thích không đồng nghĩa với quan hệ nhân quả.

## 3. Dataset và thiết kế thực nghiệm

Dùng các tệp train/test chính thức của Adult [2], lưu nguồn, checksum, kích thước, phân bố nhãn và quy tắc xử lý bản ghi trùng. Test được giữ riêng. Trên train, dùng **Stratified 5-fold CV**: bốn fold để fit, một fold validation, luân phiên; không cần thêm một validation holdout. Mọi mô hình dùng cùng fold trong mỗi seed. Chạy **3 seed: 42, 2026, 5175**; test cố định, seed thay đổi CV và thành phần ngẫu nhiên của thuật toán.

Pipeline: chuẩn hóa nhãn, chuyển `?` thành missing; impute số bằng median, biến phân loại bằng mode; one-hot với nhóm tham chiếu được lưu. StandardScaler cho biến số của Logistic Regression. Fit imputer/encoder/scaler chỉ trong training fold. Cố định chính sách đối với `fnlwgt` và hai biến giáo dục trùng thông tin trước khi chạy: dự kiến bỏ `fnlwgt`, giữ `education-num` và bỏ `education`; ghi rõ đây là lựa chọn thiết kế. Không dùng test để chọn biến hoặc siêu tham số.

| Mô hình | Không gian tuning dự kiến | Ngân sách mỗi seed |
| --- | --- | --- |
| Logistic L1 | 12 giá trị `C` log-spaced từ 0.001 đến 100 | 12 cấu hình × 5 fold |
| Logistic L2 | Cùng dãy `C` | 12 cấu hình × 5 fold |
| Logistic Elastic Net | `C` ∈ {0.01, 0.1, 1, 10}; `l1_ratio` ∈ {0.2, 0.5, 0.8} | 12 cấu hình × 5 fold |
| GBDT | `n_estimators` ∈ {100, 200}; `learning_rate` ∈ {0.03, 0.1}; `max_depth` ∈ {2, 3, 4} | 12 cấu hình × 5 fold |

Ngân sách bằng nhau theo **số cấu hình/fold cho mỗi biến thể**, không bảo đảm thời gian bằng nhau; tổng ngân sách của ba biến thể Logistic lớn hơn một GBDT và phải công bố. Chọn cấu hình theo **ROC-AUC validation**, refit trên toàn train. Báo cáo test ROC-AUC, average precision (AP), F1, balanced accuracy, accuracy và log-loss; ngưỡng phân loại cố định 0.5. Mọi metric chính trình bày **mean ± sample std** qua ba seed.

**Ablation bắt buộc:** so sánh L1/L2/Elastic Net tại cùng `C=1` (Elastic Net dùng `l1_ratio=0.5`), giữ pipeline/split/solver nhất quán; tách khỏi bảng mô hình đã tuning. Đo hiệu năng và số hệ số có trị tuyệt đối lớn hơn `1e-6`.

## 4. Kế hoạch analysis và discussion

Phân tích chênh lệch hiệu năng theo cặp cùng seed, lỗi false positive/false negative, hệ số và odds ratio có điều kiện, SHAP toàn cục/cục bộ, chi phí tính toán và giới hạn của giải thích. Báo cáo thời gian tuning, refit, suy luận và SHAP riêng. Với Logistic ghi số hệ số + intercept; với GBDT ghi số cây, nút, lá và số giá trị lá, không coi các đại lượng này tương đương trực tiếp với số tham số Logistic.

SHAP dự kiến dùng 100 mẫu background từ train và tối đa 500 mẫu test cố định, lưu ID và cấu hình explainer. Kiểm tra tổng attribution khớp đầu ra được giải thích. Mở rộng tùy nguồn lực: bootstrap 30 lần trên train với siêu tham số cố định để kiểm tra độ ổn định; fairness theo nhóm chỉ khi báo cáo cả cỡ mẫu và định nghĩa metric. Nội dung chi tiết ở [Analysis and discussion](docs/analysis_and_discussion.md).

## 5. Tiến độ, phân công và tài nguyên

| Thời gian | Công việc và sản phẩm | Phụ trách dự kiến |
| --- | --- | --- |
| Tuần 1 | Chốt bài toán, paper, proposal 1–2 trang; nộp điểm kiểm tra | Cả nhóm; trưởng nhóm nộp |
| Tuần 2–3 | Đọc paper, công thức, kiểm tra dữ liệu; baseline và báo cáo tiến độ cuối tuần 3 | Thành viên A: dữ liệu/baseline; B: lý thuyết |
| Tuần 4–6 | Tuning ba seed, tái hiện, ablation, log và biểu đồ | A: thực nghiệm; B: SHAP; C: tái hiện/kiểm tra |
| Tuần 7–8 | Analysis, hạn chế, báo cáo, slide/video, kiểm tra chạy lại | C: tổng hợp; cả nhóm phản biện |

Tên A/B/C là vai trò dự kiến, cần thay bằng tên/MSSV và tỷ lệ đóng góp thực tế; nhóm hai người chia lại phần C. Tài nguyên dự kiến: CPU laptop/desktop, RAM mục tiêu 16 GB, không bắt buộc GPU; dùng Colab CPU nếu thiếu tài nguyên. Cấu hình máy thực tế và thời gian đo sẽ được lưu khi chạy.

## 6. Tái lập và sản phẩm

Python 3.10+, scikit-learn, SHAP; giữ dependencies trong `requirements.txt`, config từng thí nghiệm trong `configs/`, mã trong `src/`, kết quả trong `results/`, báo cáo trong `reports/`. Dùng Git, commit theo mốc thực tế; ghi rõ code thư viện, code tham khảo và phần nhóm tự viết/sửa.

### 6.1. Cách chạy lại

**Cài đặt cục bộ** (Python 3.10; dữ liệu Adult tự tải từ UCI ở lần chạy đầu vào `data/raw/`):

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Trên Windows PowerShell, dùng Python 3.10 để tạo môi trường cục bộ:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Nếu đã có môi trường Conda của dự án, chạy `conda activate ml_assignment`. Môi trường `.venv` chứa đường dẫn tuyệt đối đến Python trên máy tạo nó. Khi chuyển máy, tạo môi trường trong thư mục mới (ví dụ .venv-local nếu .venv đã tồn tại) và cài gói từ `requirements.txt`. Có thể kiểm tra Python đang dùng bằng `python -c "import sys; print(sys.executable)"`.

**Chạy một thí nghiệm:** `python -m experiments.<nhóm>.<task> --config configs/<nhóm>/<tên>.json`. Kết quả vẫn ghi vào `results/<tên config>/`, không thêm cấp thư mục nhóm (kèm `config.json` và `env.json` gồm phiên bản thư viện, CPU, RAM). Ví dụ `configs/modeling/baseline.json` vẫn tạo thư mục `baseline/`. Các lệnh dưới đây thay cho entry point phẳng trước khi reorganise.

| Bước | Lệnh | Nội dung |
| --- | --- | --- |
| Dữ liệu | `python -m experiments.data.check --config configs/data/data_check.json` | Checksum, số dòng, nhãn, missing |
| Baseline | `python -m experiments.modeling.baseline --config configs/modeling/baseline.json` | Logistic L2 `C=1`, 5-fold × 3 seed |
| Tuning | `python -m experiments.modeling.tuning --config configs/modeling/tuning_logistic_l1.json` (tương tự `_l2`, `_elasticnet`, `tuning_gbdt`, `tuning_gbdt_extended`, `tuning_ebm_gam`, `tuning_ebm_ga2m`) | GridSearchCV theo ROC-AUC, refit, test |
| Ablation | `python -m experiments.modeling.ablation --config configs/modeling/ablation.json` | L1/L2/EN tại `C=1` |
| SHAP | `python -m experiments.explainability.explain --config configs/explainability/explain.json` | TreeExplainer/LinearExplainer, kiểm tra tính cộng, hình |
| Tái hiện paper | `python -m experiments.reproduction.reproduce_shap --config configs/reproduction/reproduce_shap_v2.json` | Figure 3 và Figure 4 của Lundberg & Lee (2017) |
| Ổn định giải thích | `python -m experiments.explainability.stability --config configs/explainability/stability.json` | Bootstrap 30 lần trên train |
| Phân tích bổ sung | `python -m experiments.analysis.supplementary --config configs/analysis/analysis.json` | KTC bootstrap test, fairness, dependence plot |

**Chạy trên GitHub Actions** (cách nhóm đã dùng, runner Ubuntu 4 CPU): workflow `.github/workflows/run-experiment.yml` có trigger trên nhánh `Thuy-branch`, nhưng `configs/ci_runs.json` hiện có `runs: []`, nên không có job thí nghiệm khi push. Muốn chạy cần điền danh sách task/config hoặc dùng workflow_dispatch khi workflow khả dụng trên nhánh mặc định. Không tự sửa danh sách CI chỉ để chạy cục bộ. Bộ chạy cục bộ mới dùng `configs/suites.json`. Toàn bộ thí nghiệm có thể mất vài giờ tùy máy.

Workflow dùng task có dấu chấm, ví dụ `modeling.baseline`, và đường dẫn config đầy đủ. Nếu bỏ trống config, `src/common/config.py` cung cấp mặc định; riêng `modeling.tuning` phải chỉ rõ config mô hình. Workflow manual `active_workflows.yaml` checkout ref được chọn khi dispatch để chạy đúng cấu trúc của ref đó. Manifest CI giữ nguyên danh sách rỗng. Ví dụ một phần tử khi nhóm muốn bật CI: `{"task": "modeling.tuning", "config": "configs/modeling/tuning_logistic_l2.json"}`.

**Nguồn mã:** toàn bộ mã trong `src/` và `experiments/` do nhóm tự viết; chỉ gọi API của thư viện `scikit-learn`, `shap`, `lime`, `interpret-core`. Shapley chính xác và Shapley sampling trong `experiments/reproduction/reproduce_shap.py` do nhóm tự cài đặt theo công thức của paper.

Sản phẩm cuối theo mục 5 của đề: báo cáo PDF khoảng 8–12 trang chưa kể phụ lục, slide hoặc video, mã nguồn và README. **Khai báo AI:** OpenAI Codex hỗ trợ soạn proposal, khung phân tích, refactor phần dùng chung/diagnostics, runner tái hiện, kiểm tra và bản thảo report; Claude Code hỗ trợ viết mã thực nghiệm ban đầu. GitHub Copilot hỗ trợ workflow GitHub Actions và tổng hợp số liệu vào `docs/`. Nhóm phải kiểm tra tài liệu, xác thực kết quả, hiểu công thức/code và cập nhật khai báo theo sử dụng thực tế.

## Tài liệu

1. Lundberg & Lee (2017). [A Unified Approach to Interpreting Model Predictions](https://arxiv.org/abs/1705.07874).
2. UCI Machine Learning Repository. [Adult dataset](https://archive.ics.uci.edu/dataset/2/adult).
3. Tibshirani (1996). *Regression Shrinkage and Selection via the Lasso*; Zou & Hastie (2005). *Regularization and Variable Selection via the Elastic Net* — tài liệu nền tảng được đề bài gợi ý, cần đọc ở tuần 2.
4. [scikit-learn: Linear models](https://scikit-learn.org/stable/modules/linear_model.html); [SHAP TreeExplainer](https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html). Khi triển khai, đối chiếu API với phiên bản đã cài.
5. `00_ML-Assignment-Topics.pdf`: Topic 1 (trang 3–4), mục 4 (trang 12), sản phẩm nộp (trang 13).
