# To-do list — Topic 1: Interpretable Regression & Explainability

> Bám theo quy trình 6 bước (mục 4.1), yêu cầu thực nghiệm/mã nguồn (4.2–4.4), sản phẩm nộp (5.1) và rubric (5.2) trong [00_ML-Assignment-Topics.md](../00_ML-Assignment-Topics.md).
> Đánh dấu `[x]` khi xong, ghi kèm link commit/tệp làm bằng chứng.

## Tiến độ hiện tại (cập nhật 2026-11-19)

| Giai đoạn | Trạng thái | Bằng chứng (artifact GitHub Actions) |
| --- | --- | --- |
| 0. Thiết lập | Hạ tầng CI xong; còn phân quyền, chọn hình thức trình bày | `.github/workflows/run-experiment.yml`, `configs/ci_runs.json` |
| 1. Proposal | Chưa nộp; đã chốt hình paper cần tái hiện | — |
| 2. Dữ liệu & baseline | Xong phần code/kết quả; còn EDA notebook, lý thuyết, báo cáo tiến độ | `data_check-1`, `baseline-1` |
| 3. Thực nghiệm | Tuning, ablation, SHAP xong; tái hiện paper đang chạy; chưa làm bonus | `tuning_*-3`, `ablation-3`, `tuning_gbdt_extended-4`, `explain-4` |
| 4. Hoàn thiện | Đã điền số liệu vào docs; chưa viết báo cáo/slide | `docs/analysis_and_discussion.md` |

## Phân vai (đề xuất — nhóm xác nhận lại)

| Vai trò | Thành viên | Trách nhiệm chính |
| --- | --- | --- |
| **L** — Trưởng nhóm, tái hiện & tổng hợp (vai C) | Đinh Trương Tuệ Linh | Nộp bài, tái hiện paper SHAP, ghép báo cáo, quản lý repo |
| **T** — Lý thuyết & giải thích (vai B) | Đinh Thị Thu Thủy | Hệ thống hóa tài liệu, công thức, SHAP, hệ số/odds ratio |
| **R** — Dữ liệu & thực nghiệm (vai A) | Hứa Phước Trường | Dữ liệu, pipeline, baseline, tuning, ablation, đo thời gian |

Quy ước: mỗi người tự commit phần mình (giảng viên có thể xem lịch sử Git); mỗi người phải giải thích được toàn bộ code và công thức (mục 4.4).

---

## Giai đoạn 0 — Thiết lập chung (đầu tuần 1)

- [ ] **[L]** Thêm tất cả thành viên làm collaborator trên GitHub; thống nhất quy ước branch/commit (`feat:`, `fix:`, `docs:`, `exp:`).
- [x] **[L]** Cập nhật `.gitignore`: bỏ qua `data/raw/`, `results/*`.
- [x] **[L]** Chạy thí nghiệm trên GitHub Actions (nhánh `Thuy-branch`, tự chạy khi push; danh sách chạy trong `configs/ci_runs.json`, kết quả là artifact).
- [ ] **[Cả nhóm]** Tạo môi trường Python 3.10+ cục bộ (không bắt buộc vì đã chạy trên CI).
- [ ] **[R]** Rút gọn `requirements.txt` còn các gói thực sự dùng; bổ sung `interpret` nếu làm phần EBM.
- [ ] **[Cả nhóm]** Chọn hình thức trình bày: A (thuyết trình 20' + Q&A) hoặc B (video 10').

## Giai đoạn 1 — Proposal (nộp cuối tuần 1) ◆

- [ ] **[L]** Rút gọn proposal xuống **1–2 trang** (đề yêu cầu); chuyển chi tiết sang `docs/analysis_and_discussion.md`.
- [ ] **[L]** Thay A/B/C bằng tên + MSSV trong bảng tiến độ của README.
- [x] **[T]** Đọc Lundberg & Lee (2017), chốt **hình/thí nghiệm gốc**: Figure 3 (mục 5.1) và Figure 4 (mục 5.2) — ghi trong mục 6 của docs.
- [ ] **[T]** Bổ sung tài liệu InterpretML (Nori et al., 2019) vào danh mục tham khảo.
- [ ] **[Cả nhóm]** Chốt hướng **bonus** (bước 6): ví dụ EBM/GAM, bootstrap ổn định giải thích, hoặc fairness theo nhóm.
- [ ] **[L]** Xuất proposal ra PDF và nộp qua LMS.

## Giai đoạn 2 — Tài liệu & baseline (tuần 2–3)

### 2.1 Dữ liệu
- [x] **[R]** `src/data.py`: tải Adult (`adult.data`, `adult.test`) từ UCI, lưu vào `data/raw/`. — `data_check-1`
- [x] **[R]** `data_meta.json`: URL nguồn, SHA-256, số dòng, tỷ lệ nhãn train/test. — `data_check-1`
- [x] **[R]** Chuẩn hóa nhãn (`>50K.` → `>50K`), `?` → missing, giữ bản ghi trùng (24 train / 5 test). — `data_check-1`
- [ ] **[R]** Notebook `notebooks/view_data.ipynb`: EDA cơ bản (phân bố nhãn, missing, biến số/phân loại, tương quan).

### 2.2 Pipeline & baseline
- [x] **[R]** `src/preprocess.py`: `ColumnTransformer` (median/mode impute, one-hot, `StandardScaler` cho Logistic); bỏ `fnlwgt`, `education`.
- [x] **[R]** Stratified 5-fold theo seed {42, 2026, 5175} (trong `experiments/baseline.py`, `experiments/tuning.py`).
- [x] **[R]** `src/metrics.py`: ROC-AUC, AP, F1, balanced accuracy, accuracy, log-loss (ngưỡng 0.5).
- [x] **[R]** `configs/baseline.json` + `experiments/baseline.py`: Logistic L2 `C=1`, 3 seed — test ROC-AUC 0.9033. — `baseline-1`
- [x] **[R]** Ghi metadata máy (CPU, RAM, OS, phiên bản thư viện) vào `env.json` của mỗi artifact.

### 2.3 Lý thuyết
- [ ] **[T]** Hệ thống hóa các hướng tiếp cận (mô hình nội tại vs hậu nghiệm: Lasso, Elastic Net, GAM/EBM, LIME, SHAP) kèm ưu/nhược điểm.
- [ ] **[T]** Viết phần toán: loss Logistic + penalty, solver SAGA, GBDT (gradient boosting trên log-loss), Shapley values và 3 tính chất (local accuracy, missingness, consistency).
- [ ] **[L]** Báo cáo tiến độ ngắn (kết quả baseline) — nộp **cuối tuần 3** ◆.

## Giai đoạn 3 — Tái hiện & thực nghiệm (tuần 4–6)

### 3.1 Tuning mô hình chính
- [x] **[R]** `configs/tuning_*.json` cho L1, L2, Elastic Net, GBDT (12 cấu hình mỗi biến thể). — `tuning_*-3`
- [x] **[R]** `experiments/tuning.py`: `GridSearchCV` theo ROC-AUC, refit toàn train, đánh giá test; lưu `cv_results_`, tham số tốt nhất, dự đoán test, thời gian tuning/refit/inference. — `tuning_*-3`
- [x] **[R]** Ghi lại convergence warning và cấu hình lỗi (không loại âm thầm).
- [x] **[R]** Đếm kích thước mô hình: hệ số + intercept (Logistic); số cây/nút/lá (GBDT); dung lượng file model.
- [ ] **[R]** `src/aggregate.py`: tổng hợp mean ± sample std qua 3 seed (đã có trong `tuning.py`), còn thiếu chênh lệch theo cặp Δ_s.
- [x] **[R]** Phân tích độ nhạy GBDT với lưới mở rộng — chỉ +0.0003 ROC-AUC, khoảng cách GBDT–Logistic không bị ước lượng thấp. — `tuning_gbdt_extended-4`

### 3.2 Ablation (bắt buộc)
- [x] **[R]** `configs/ablation.json` + `experiments/ablation.py`: L1/L2/EN tại `C=1`, `l1_ratio=0.5`; đo hiệu năng, số hệ số |β|>1e-6, refit time. — `ablation-3`

### 3.3 Giải thích
- [x] **[T]** Bảng hệ số, odds ratio, nhóm tham chiếu (`features_seed*.csv`, `reference_categories.json`); tỷ lệ chọn và cùng dấu qua seed (mục 4.1 docs). — `tuning_*-3`
- [x] **[T]** `experiments/explain.py`: TreeExplainer (GBDT) + LinearExplainer (Elastic Net), 100 background, 500 mẫu test cố định (lưu ID); beeswarm, bar, waterfall. — `explain-4`
- [x] **[T]** Kiểm tra local accuracy: sai số max 5.1e-8 (GBDT), 7.1e-15 (Logistic). — `explain-4`
- [x] **[T]** Gộp SHAP theo biến gốc, lưu cả bản chưa gộp. — `explain-4`
- [x] **[T]** Chọn TP/TN/FP/FN (ID nhỏ nhất, seed 42) và 4 mẫu Logistic/GBDT bất đồng. — `explain-4`
- [ ] **[T]** Dependence plot cho `capital-gain` và `age` để xác nhận giả thuyết phi tuyến.

### 3.4 Tái hiện paper
- [ ] **[L]** `experiments/reproduce_shap.py`: Shapley chính xác bằng liệt kê tập con, so với KernelExplainer/TreeExplainer — **code xong, đang chờ artifact `reproduce_shap`**.
- [ ] **[L]** Tái hiện Figure 3 (Kernel SHAP vs Shapley sampling vs LIME) và Figure 4 (ví dụ đồ chơi) — **đang chờ artifact**.
- [ ] **[L]** Điền bảng mục 6 trong docs — đã ghi locator và số đọc từ hình; còn kết quả tái hiện.

### 3.5 Mở rộng / bonus (chọn ít nhất một)
- [ ] **[T]** EBM (`interpret`) như mô hình glass-box thứ hai; so với Logistic và GBDT.
- [ ] **[R]** Bootstrap 30 lần: độ ổn định hệ số/SHAP (Jaccard top-k, tương quan hạng).
- [ ] **[L]** Fairness theo `sex`/`race`: TPR/FPR theo nhóm, kèm cỡ mẫu.

## Giai đoạn 4 — Phân tích & hoàn thiện (tuần 7–8)

### 4.1 Phân tích
- [x] **[R]** Điền bảng kết quả chính, bảng ablation, bảng SHAP trong docs từ artifact.
- [ ] **[T]** Viết thảo luận RQ2–RQ3: regularization, hệ số vs SHAP, giới hạn giải thích.
- [ ] **[L]** Viết thảo luận RQ1, hạn chế, kết luận; kiểm tra mọi con số truy được tới file trong `results/`.

### 4.2 Báo cáo PDF (8–12 trang, Overleaf/LaTeX)
- [ ] **[T]** a) Introduction — bài toán, ứng dụng, các hướng tiếp cận, lý do chọn.
- [ ] **[T]** b) Methods — pipeline, công thức cốt lõi.
- [ ] **[R]** c) Experiments — cài đặt, dataset, kết quả, ablation, thảo luận.
- [ ] **[L]** d) Improvement/application (bonus).
- [ ] **[L]** e) Conclusion & limitations; f) References.
- [ ] **[L]** g) Phụ lục: bảng phân công + **% đóng góp từng người**; khai báo AI (công cụ nào, phần nào).
- [ ] **[Cả nhóm]** Đọc chéo, kiểm tra đạo văn/trích dẫn.

### 4.3 Mã nguồn & tái lập
- [ ] **[R]** Script chạy toàn bộ (`experiments/run_all.py` hoặc `Makefile`/`run_all.ps1`).
- [ ] **[L]** README: lệnh cài đặt, chạy, cấu trúc thư mục, nguồn code tham khảo và phần tự viết.
- [ ] **[Cả nhóm]** Một người **khác** người viết code clone repo sạch và chạy lại từ đầu theo README.

### 4.4 Trình bày
- [ ] **[Cả nhóm]** Slide (phương án A) hoặc video 10' (phương án B, mỗi người trình bày một phần).
- [ ] **[Cả nhóm]** Tập Q&A: mỗi người giải thích được code và công thức của người khác.

### 4.5 Nộp ◆
- [ ] **[L]** Nộp qua LMS **chậm nhất 2 ngày trước ngày thi cuối kỳ**: PDF + slide/video + link repo. (Trễ: −10%/ngày.)

---

## Ước lượng kinh phí

Mục tiêu **0 đ**. Toàn bộ dự án chạy được bằng tài nguyên miễn phí.

| Hạng mục | Lựa chọn miễn phí | Phương án trả phí (không cần) | Chi phí dự kiến |
| --- | --- | --- | --- |
| Dữ liệu | UCI Adult (CC BY 4.0) | — | 0 đ |
| Thư viện | scikit-learn, SHAP, pandas, matplotlib, interpret (mã nguồn mở) | — | 0 đ |
| Tính toán | Laptop CPU cá nhân; dự phòng Google Colab / Kaggle Notebooks bản free | Colab Pro | 0 đ |
| Lưu trữ & Git | GitHub repo (free); Google Drive 15 GB | — | 0 đ |
| Viết báo cáo | Overleaf free (LaTeX) | Overleaf Premium | 0 đ |
| Slide | Google Slides / LaTeX Beamer / Canva free | — | 0 đ |
| Video (nếu chọn B) | OBS Studio + DaVinci Resolve free / Clipchamp | — | 0 đ |
| Họp nhóm | Google Meet / Zalo / MS Teams (tài khoản trường) | — | 0 đ |
| Công cụ AI | GitHub Copilot (miễn phí qua GitHub Student Developer Pack), ChatGPT/Claude bản free | Gói trả phí | 0 đ |
| **Tổng** | | | **0 đ** |

**Vì sao CPU là đủ:** Adult có khoảng 48.8k dòng và khoảng 100 cột sau one-hot. Mỗi seed có 4 biến thể × 12 cấu hình × 5 fold = 240 lần fit, cộng 4 lần refit. Ba seed là khoảng 730 lần fit Logistic/GBDT cỡ nhỏ; TreeExplainer trên 500 mẫu nhanh vì có thuật toán đa thức cho cây. Không cần GPU.

**Rủi ro chi phí và cách tránh:**
- GBDT của scikit-learn chạy chậm → dùng `n_jobs=-1` trong `GridSearchCV`, hoặc đổi sang `HistGradientBoostingClassifier` (phải ghi rõ thay đổi protocol).
- KernelExplainer (nếu dùng cho tái hiện) chạy chậm → chỉ dùng trên bài toán nhỏ, ít biến.
- Colab free bị ngắt phiên → lưu kết quả từng seed ra file, chạy tiếp được từ seed dở.
