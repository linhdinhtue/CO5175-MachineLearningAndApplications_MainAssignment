# To-do list — Topic 1: Interpretable Regression & Explainability

> Bám theo quy trình 6 bước (mục 4.1), yêu cầu thực nghiệm/mã nguồn (4.2–4.4), sản phẩm nộp (5.1) và rubric (5.2) trong [00_ML-Assignment-Topics.md](../00_ML-Assignment-Topics.md).
> Đánh dấu `[x]` khi xong, ghi kèm artifact/tệp làm bằng chứng. Ô `[x]` có ghi **(Thủy)** là phần Thủy đã thực hiện.

## Tiến độ hiện tại (cập nhật 2026-11-19)

| Giai đoạn | Trạng thái | Bằng chứng (artifact GitHub Actions) |
| --- | --- | --- |
| 0. Thiết lập | Hạ tầng CI xong; còn phân quyền, chọn hình thức trình bày | `.github/workflows/run-experiment.yml`, `configs/ci_runs.json` |
| 1. Proposal | Chưa nộp; đã chốt hình paper cần tái hiện | — |
| 2. Dữ liệu & baseline | Xong phần code/kết quả; còn EDA notebook, lý thuyết, báo cáo tiến độ | `results/`(`data_check-1`, `baseline-1`) |
| 3. Thực nghiệm | **Xong toàn bộ**: tuning, ablation, SHAP, tái hiện paper (Figure 3, 4), 3 hướng bonus (EBM, ổn định giải thích, fairness), KTC bootstrap, dependence plot | `tuning_*-3`, `ablation-3`, `tuning_gbdt_extended-4`, `explain-4`, `reproduce_shap-5`, `reproduce_shap_v2-6`, `tuning_ebm_*-7`, `stability-7`, `analysis-7` |
| 4. Hoàn thiện | Đã điền số liệu và viết kết luận RQ1–RQ3 trong docs; README có hướng dẫn chạy; chưa viết báo cáo PDF/slide | `docs/analysis_and_discussion.md`, `README.md` |

## Phần Thủy đã thực hiện: hướng đi, ưu điểm, nhược điểm

Toàn bộ phần code thực nghiệm, kết quả và số liệu trong docs tới thời điểm này do Thủy thực hiện trên nhánh `Thuy-branch`.

| Hạng mục | Hướng đi đã chọn | Ưu điểm | Nhược điểm / rủi ro |
| --- | --- | --- | --- |
| Hạ tầng chạy | Chạy mọi thí nghiệm trên **GitHub Actions** (runner Ubuntu 4 CPU, 16 GB); mỗi config → một job → một artifact; danh sách chạy trong `configs/ci_runs.json` | 0 đ; không cần cài môi trường cục bộ; mỗi kết quả kèm `env.json` (phiên bản thư viện, commit SHA) nên truy vết và tái lập được | Mỗi lần thử phải push và chờ; khó debug; nếu repo private thì giới hạn 2 000 phút/tháng; sửa file qua web tạo nhiều commit nhỏ; workflow chỉ nằm trên `Thuy-branch` nên chưa có nút chạy tay |
| Dữ liệu | Dùng file train/test **chính thức** của UCI Adult; `?` → missing; giữ 24/5 bản ghi trùng; bỏ `fnlwgt` và `education` (giữ `education-num`) | Test chuẩn, so sánh được với tài liệu; có checksum SHA-256; không chọn biến bằng test | Giữ bản ghi trùng có thể làm rò rỉ nhẹ giữa fold CV; bỏ `fnlwgt` là lựa chọn thiết kế, chưa kiểm tra ảnh hưởng |
| Pipeline & CV | `Pipeline` scikit-learn (impute → one-hot → scale) fit **bên trong** từng fold; Stratified 5-fold; 3 seed | Tránh rò rỉ dữ liệu; mọi mô hình dùng cùng fold | Seed chỉ đổi cách chia fold → std trên test gần 0, không phản ánh bất định lấy mẫu |
| Tuning | `GridSearchCV` theo ROC-AUC, **12 cấu hình/mô hình**, refit toàn train | Ngân sách tuning bằng nhau, minh bạch; lưu đủ `cv_results_` | Lưới nhỏ; GBDT tốt nhất rơi vào **góc lưới** → đã chạy thêm lưới mở rộng (chỉ +0.0003 AUC) nhưng lưới này thêm sau khi xem test, phải ghi là phân tích bổ sung |
| Mô hình hộp đen | `GradientBoostingClassifier` của scikit-learn (không dùng XGBoost/LightGBM) | Không thêm thư viện; TreeExplainer hỗ trợ chính xác; đủ nhanh trên Adult | Chậm hơn các thư viện boosting chuyên dụng; không so với MLP (đề cho phép một mô hình hộp đen) |
| Giải thích | TreeExplainer (GBDT) và LinearExplainer (Elastic Net), `interventional`, giải thích **raw log-odds**, 100 background, 500 mẫu test cố định; gộp SHAP theo biến gốc | Tính cộng khớp tới 5e-8; rất nhanh (< 0.5 s); lưu ID mẫu để tái lập | Giả định đặc trưng độc lập; kết quả phụ thuộc background; gộp one-hot là phép cộng để báo cáo, không phải Shapley của biến gộp; mới chạy 1 seed |
| Tái hiện paper | Chọn **Figure 3** (Kernel SHAP vs Shapley sampling vs LIME) và **Figure 4** (ví dụ đồ chơi) của Lundberg & Lee (2017); bỏ Figure 5 (MNIST/DeepLIFT) | Chạy được trên CPU; Figure 4 so sánh chính xác được; Figure 3 kiểm tra được đúng luận điểm chính của paper | Paper không nêu dữ liệu/độ sâu cây/background cho Figure 3 → nhóm tự đặt giả định, chỉ so định tính; số Figure 4 đọc từ hình là xấp xỉ |
| Tái hiện lần 2 | Sửa cách tính ngân sách Shapley sampling theo paper (chia cho M đặc trưng), thêm biến thể Kernel SHAP top-10, ép cây sparse dùng đúng 3 đặc trưng; giữ cách cũ làm đối chứng | Giải thích được chênh lệch lần 1 bằng thực nghiệm; hình sparse gần trùng paper; có đối chứng cho thấy ưu thế của Kernel SHAP đến từ ước lượng đồng thời | Sửa thiết lập sau khi xem kết quả lần 1 (phải nói rõ trong báo cáo); Kernel SHAP vẫn lệch thấp ở cây sparse, nguyên nhân mới là giả thuyết |
| Bonus: EBM (GAM / GA2M) | Thêm Explainable Boosting Machine (`interpret-core`) làm mô hình **dễ diễn giải nhưng phi tuyến**; 2 biến thể: không tương tác (GAM thuần) và 10 tương tác cặp (GA2M); 3 seed, cùng fold | Trả lời trực tiếp câu hỏi của đề tài (có thu hẹp được khoảng cách với hộp đen mà vẫn đọc được mô hình không?); đề có nêu GAM/EBM và InterpretML; cặp GAM vs GA2M là một ablation tự nhiên về vai trò tương tác | Không tuning (chỉ 1 cấu hình mỗi biến thể, dùng mặc định) → ngân sách tuning **thấp hơn** các mô hình khác, phải ghi rõ; `interpret-core` đã ghim 0.7.8 theo `env.json` |
| Bonus: ổn định giải thích | Bootstrap 30 lần trên train, giữ cố định siêu tham số, background và 500 dòng được giải thích; đo Spearman xếp hạng, Jaccard top-5, tần suất chọn hệ số, độ nhất quán dấu | Nằm trong "Hướng mở rộng" của đề; biến "giải thích có đáng tin không" thành con số; đồng thời cho khoảng biến thiên hiệu năng thật (bù cho std qua seed ≈ 0) | Chỉ đo bất định do dữ liệu train, không gồm bất định do tuning lại; 30 lần là ít cho percentile đuôi |
| Bonus: fairness theo nhóm | TPR/FPR/PPV/tỷ lệ chọn theo `sex` và `race` cho 5 mô hình, ngưỡng 0.5; chỉ so nhóm có ≥ 100 mẫu | Nằm trong "Hướng mở rộng" của đề; rẻ (dùng lại dự đoán); cho thấy đánh đổi hiệu năng–diễn giải còn có chiều công bằng | Chỉ mô tả, không can thiệp giảm thiên lệch; nhóm `race` nhỏ nên khoảng cách kém tin cậy; không kết luận "công bằng" từ việc bỏ biến nhạy cảm |
| Bổ sung phân tích | Bootstrap 1 000 lần trên **test** cho chênh lệch theo cặp (GBDT/EBM − từng Logistic); dependence plot so SHAP GBDT, SHAP Logistic và hàm hình dạng EBM cho 4 biến số | Cho khoảng tin cậy 95% (rubric 30% cần "giải thích kết quả"); kiểm chứng giả thuyết phi tuyến của `age`/`capital-gain` bằng hình | Bootstrap test giữ mô hình cố định (seed 42), không gồm biến thiên do huấn luyện |

## Chiến lược tối đa điểm (theo rubric mục 5.2 của đề)

| Tiêu chí | Trọng số | Đã có | Hướng đã chọn để lấp chỗ trống và lý do |
| --- | --- | --- | --- |
| Thực nghiệm & phân tích | 30% | Baseline, 3 seed, tuning cùng ngân sách, ablation, SHAP, tái hiện Figure 3–4 có giải thích chênh lệch | Thêm **khoảng tin cậy bootstrap** và **dependence plot** — vì std qua seed ≈ 0 là điểm yếu dễ bị hỏi, và giả thuyết phi tuyến cần bằng chứng trực tiếp |
| Bonus cải tiến/ứng dụng | +1 | Chưa có | Chọn **cả 3 hướng rẻ**: EBM (trọng tâm), ổn định giải thích, fairness. Lý do: cả 3 đều được đề gợi ý trực tiếp cho Topic 1, chạy được trên CPU miễn phí, và EBM tạo được câu chuyện "cải tiến" rõ ràng: một mô hình vừa đọc được vừa (kỳ vọng) gần hiệu năng GBDT. Không chọn MLP vì không thêm gì cho câu chuyện diễn giải và tốn tuning |
| Mã nguồn & tái lập | 10% | Config từng thí nghiệm, CI, `env.json`, README hướng dẫn chạy (mục 6.1), `interpret-core` đã ghim | Còn: một người khác chạy lại theo README để xác nhận |
| Phương pháp & toán | 20% | Công thức cơ bản trong proposal | Viết trong báo cáo: loss Logistic + penalty, gradient boosting, Shapley (Eq. 8), Kernel SHAP (Theorem 2), Linear SHAP, EBM (cyclic boosting, GA2M) |
| Tổng quan | 15% | Chưa có | Viết trong báo cáo: so sánh mô hình nội tại (Lasso/EN/GAM/EBM) với hậu nghiệm (LIME/SHAP) có phản biện — dùng chính kết quả Figure 4 (LIME sai dấu) làm minh họa |
| Trình bày | 25% | Chưa có | Chọn A hoặc B sớm; mỗi người phải giải thích được code và công thức |

---

## Giai đoạn 0 — Thiết lập chung (đầu tuần 1)

- [ ] Thêm tất cả thành viên làm collaborator trên GitHub; thống nhất quy ước branch/commit (`feat:`, `fix:`, `docs:`, `exp:`).
- [x] Cập nhật `.gitignore`: bỏ qua `data/raw/`, `results/*`. **(Thủy)**
- [x] Chạy thí nghiệm trên GitHub Actions (nhánh `Thuy-branch`, tự chạy khi push; danh sách chạy trong `configs/ci_runs.json`, kết quả là artifact). **(Thủy)**
- [ ] Đưa workflow lên `main` khi nhóm thống nhất, để có nút **Run workflow** chạy tay.
- [ ] Rút gọn `requirements.txt` còn các gói thực sự dùng; bổ sung `interpret` nếu làm phần EBM.
- [ ] Chọn hình thức trình bày: A (thuyết trình 20' + Q&A) hoặc B (video 10').

## Giai đoạn 1 — Proposal (nộp cuối tuần 1) ◆

- [ ] Rút gọn proposal xuống **1–2 trang** (đề yêu cầu); chuyển chi tiết sang `docs/analysis_and_discussion.md`.
- [ ] Ghi phân công thực tế (tên + MSSV) trong bảng tiến độ của README.
- [x] Đọc Lundberg & Lee (2017), chốt **hình/thí nghiệm gốc**: Figure 3 (mục 5.1) và Figure 4 (mục 5.2) — ghi trong mục 6 của docs. **(Thủy)**
- [ ] Bổ sung tài liệu InterpretML (Nori et al., 2019) vào danh mục tham khảo.
- [x] Chốt hướng **bonus**: cả 3 hướng EBM (trọng tâm), ổn định giải thích, fairness — lý do xem mục "Chiến lược tối đa điểm". **(Thủy)**
- [ ] Xuất proposal ra PDF và nộp qua LMS (trưởng nhóm nộp).

## Giai đoạn 2 — Tài liệu & baseline (tuần 2–3)

### 2.1 Dữ liệu
- [x] `src/data.py`: tải Adult (`adult.data`, `adult.test`) từ UCI. — `data_check-1` **(Thủy)**
- [x] `data_meta.json`: URL nguồn, SHA-256, số dòng, tỷ lệ nhãn train/test. — `data_check-1` **(Thủy)**
- [x] Chuẩn hóa nhãn (`>50K.` → `>50K`), `?` → missing, giữ bản ghi trùng (24 train / 5 test). — `data_check-1` **(Thủy)**
- [ ] Notebook `notebooks/view_data.ipynb`: EDA cơ bản (phân bố nhãn, missing, biến số/phân loại, tương quan).

### 2.2 Pipeline & baseline
- [x] `src/preprocess.py`: `ColumnTransformer` (median/mode impute, one-hot, `StandardScaler` cho Logistic); bỏ `fnlwgt`, `education`. **(Thủy)**
- [x] Stratified 5-fold theo seed {42, 2026, 5175} (trong `experiments/baseline.py`, `experiments/tuning.py`). **(Thủy)**
- [x] `src/metrics.py`: ROC-AUC, AP, F1, balanced accuracy, accuracy, log-loss (ngưỡng 0.5). **(Thủy)**
- [x] `configs/baseline.json` + `experiments/baseline.py`: Logistic L2 `C=1`, 3 seed — test ROC-AUC 0.9033. — `baseline-1` **(Thủy)**
- [x] Ghi metadata máy (CPU, RAM, OS, phiên bản thư viện) vào `env.json` của mỗi artifact. **(Thủy)**

### 2.3 Lý thuyết
- [ ] Hệ thống hóa các hướng tiếp cận (mô hình nội tại vs hậu nghiệm: Lasso, Elastic Net, GAM/EBM, LIME, SHAP) kèm ưu/nhược điểm.
- [ ] Viết phần toán: loss Logistic + penalty, solver SAGA, GBDT (gradient boosting trên log-loss), Shapley values và 3 tính chất (local accuracy, missingness, consistency).
- [ ] Báo cáo tiến độ ngắn (kết quả baseline) — nộp **cuối tuần 3** ◆.

## Giai đoạn 3 — Tái hiện & thực nghiệm (tuần 4–6)

### 3.1 Tuning mô hình chính
- [x] `configs/tuning_*.json` cho L1, L2, Elastic Net, GBDT (12 cấu hình mỗi biến thể). — `tuning_*-3` **(Thủy)**
- [x] `experiments/tuning.py`: `GridSearchCV` theo ROC-AUC, refit toàn train, đánh giá test; lưu `cv_results_`, tham số tốt nhất, dự đoán test, thời gian tuning/refit/inference. — `tuning_*-3` **(Thủy)**
- [x] Ghi lại convergence warning và cấu hình lỗi (không loại âm thầm). **(Thủy)**
- [x] Đếm kích thước mô hình: hệ số + intercept (Logistic); số cây/nút/lá (GBDT); dung lượng model. **(Thủy)**
- [x] Chênh lệch theo cặp kèm KTC bootstrap 95% trên test: GBDT − Elastic Net ROC-AUC +0.0236 [0.0212, 0.0261]; GBDT − EBM chứa 0. — `analysis-7` **(Thủy)**
- [x] Phân tích độ nhạy GBDT với lưới mở rộng — chỉ +0.0003 ROC-AUC, khoảng cách GBDT–Logistic không bị ước lượng thấp. — `tuning_gbdt_extended-4` **(Thủy)**

### 3.2 Ablation (bắt buộc)
- [x] `configs/ablation.json` + `experiments/ablation.py`: L1/L2/EN tại `C=1`, `l1_ratio=0.5`; đo hiệu năng, số hệ số |β|>1e-6, refit time. — `ablation-3` **(Thủy)**

### 3.3 Giải thích
- [x] Bảng hệ số, odds ratio, nhóm tham chiếu (`features_seed*.csv`, `reference_categories.json`); tỷ lệ chọn và cùng dấu qua seed (mục 4.1 docs). — `tuning_*-3` **(Thủy)**
- [x] `experiments/explain.py`: TreeExplainer (GBDT) + LinearExplainer (Elastic Net), 100 background, 500 mẫu test cố định (lưu ID); beeswarm, bar, waterfall. — `explain-4` **(Thủy)**
- [x] Kiểm tra local accuracy: sai số max 5.1e-8 (GBDT), 7.1e-15 (Logistic). — `explain-4` **(Thủy)**
- [x] Gộp SHAP theo biến gốc, lưu cả bản chưa gộp. — `explain-4` **(Thủy)**
- [x] Chọn TP/TN/FP/FN (ID nhỏ nhất, seed 42) và 4 mẫu Logistic/GBDT bất đồng. — `explain-4` **(Thủy)**
- [x] Dependence plot 4 biến số: xác nhận `age` chữ U ngược, `capital-gain` dạng ngưỡng, `hours-per-week` bão hòa; `education-num` gần tuyến tính (mục 7.2 docs). — `analysis-7` **(Thủy)**

### 3.4 Tái hiện paper
- [x] `experiments/reproduce_shap.py`: Shapley chính xác bằng liệt kê tập con; khớp KernelExplainer (Figure 4, sai số ~1e-16) và TreeExplainer (sai số ≤ 5.5e-8). — `reproduce_shap-5` **(Thủy)**
- [x] Figure 4: Kernel SHAP = Shapley chính xác = số của paper (1, 1, 0) và (3, 2, 0); LIME khác SHAP như paper nhưng khác cả số LIME của paper. — `reproduce_shap-5` **(Thủy)**
- [x] Figure 3: Kernel SHAP hẹp hơn Shapley sampling ở mọi cỡ mẫu, LIME lệch; hình sparse gần trùng paper khi chia ngân sách sampling cho M đặc trưng. Còn 1 chênh lệch có giả thuyết: Kernel SHAP lệch thấp ~0.02 ở cây sparse. — `reproduce_shap-5`, `reproduce_shap_v2-6` **(Thủy)**
- [x] Điền bảng mục 6 trong docs (locator, số đọc từ hình, kết quả lần 1 và lần 2, chênh lệch). **(Thủy)**

### 3.5 Mở rộng / bonus (đã chọn cả 3)
- [x] EBM GAM 0.9267 và GA2M 0.9272 ROC-AUC — ngang GBDT 0.9270 mà vẫn diễn giải trực tiếp được (mục 7.1 docs). — `tuning_ebm_gam-7`, `tuning_ebm_ga2m-7` **(Thủy)**
- [x] Bootstrap 30 lần: Spearman xếp hạng ≥ 0.90, top-1 giữ nguyên 30/30; 34/81 hệ số ổn định hoàn toàn (mục 7.3 docs). — `stability-7` **(Thủy)**
- [x] Fairness: GBDT/EBM thu hẹp chênh TPR theo `sex` (0.10 → 0.06–0.07); chênh theo `race` lớn nhưng nhóm nhỏ (mục 7.4 docs). — `analysis-7` **(Thủy)**
- [x] Ghim `interpret-core==0.7.8` trong `requirements.txt`. **(Thủy)**

## Giai đoạn 4 — Phân tích & hoàn thiện (tuần 7–8)

### 4.1 Phân tích
- [x] Điền bảng kết quả chính, bảng ablation, bảng SHAP trong docs từ artifact. **(Thủy)**
- [x] Viết thảo luận và kết luận RQ1–RQ3, hạn chế (mục 7.5 docs) — dùng làm khung cho báo cáo PDF. **(Thủy)**
- [ ] Rà soát: mọi con số trong báo cáo truy được tới artifact.

### 4.2 Báo cáo PDF (8–12 trang, Overleaf/LaTeX)
- [ ] a) Introduction — bài toán, ứng dụng, các hướng tiếp cận, lý do chọn.
- [ ] b) Methods — pipeline, công thức cốt lõi.
- [ ] c) Experiments — cài đặt, dataset, kết quả, ablation, thảo luận.
- [ ] d) Improvement/application (bonus).
- [ ] e) Conclusion & limitations; f) References.
- [ ] g) Phụ lục: bảng phân công + **% đóng góp từng người**; khai báo AI (công cụ nào, phần nào).
- [ ] Đọc chéo, kiểm tra đạo văn/trích dẫn.

### 4.3 Mã nguồn & tái lập
- [x] Script chạy toàn bộ qua `configs/ci_runs.json` (CI) và bảng lệnh chạy cục bộ trong README mục 6.1. **(Thủy)**
- [x] README: lệnh cài đặt, chạy từng thí nghiệm, nguồn mã, khai báo AI. **(Thủy)**
- [ ] Một người **khác** người viết code chạy lại từ đầu theo README.

### 4.4 Trình bày
- [ ] Slide (phương án A) hoặc video 10' (phương án B, mỗi người trình bày một phần).
- [ ] Tập Q&A: mỗi người giải thích được code và công thức của người khác (mục 4.4 của đề).

### 4.5 Nộp ◆
- [ ] Nộp qua LMS **chậm nhất 2 ngày trước ngày thi cuối kỳ**: PDF + slide/video + link repo (trưởng nhóm nộp; trễ: −10%/ngày).

---

## Ước lượng kinh phí

Mục tiêu **0 đ**. Toàn bộ dự án chạy được bằng tài nguyên miễn phí.

| Hạng mục | Lựa chọn miễn phí | Phương án trả phí (không cần) | Chi phí dự kiến |
| --- | --- | --- | --- |
| Dữ liệu | UCI Adult (CC BY 4.0) | — | 0 đ |
| Thư viện | scikit-learn, SHAP, LIME, pandas, matplotlib, interpret (mã nguồn mở) | — | 0 đ |
| Tính toán | GitHub Actions (miễn phí không giới hạn với repo public; 2 000 phút/tháng với repo private); dự phòng Colab/Kaggle free | Colab Pro | 0 đ |
| Lưu trữ & Git | GitHub repo (free); Google Drive 15 GB | — | 0 đ |
| Viết báo cáo | Overleaf free (LaTeX) | Overleaf Premium | 0 đ |
| Slide | Google Slides / LaTeX Beamer / Canva free | — | 0 đ |
| Video (nếu chọn B) | OBS Studio + DaVinci Resolve free / Clipchamp | — | 0 đ |
| Họp nhóm | Google Meet / Zalo / MS Teams (tài khoản trường) | — | 0 đ |
| Công cụ AI | GitHub Copilot (miễn phí qua GitHub Student Developer Pack), ChatGPT/Claude bản free | Gói trả phí | 0 đ |
| **Tổng** | | | **0 đ** |

**Chi phí tính toán thực tế đã đo** (runner GitHub 4 CPU, thời gian tuning mỗi seed): Logistic 4–8 phút (12–23 phút cho 3 seed), GBDT khoảng 2.5 phút (khoảng 7.5 phút cho 3 seed), lưới GBDT mở rộng khoảng 7 phút (khoảng 21 phút cho 3 seed), SHAP dưới 1 giây. Không cần GPU.
