# To-do list — Topic 1: Interpretable Regression & Explainability

> Bám theo quy trình 6 bước (mục 4.1), yêu cầu thực nghiệm/mã nguồn (4.2–4.4), sản phẩm nộp (5.1) và rubric (5.2) trong [00_ML-Assignment-Topics.md](../00_ML-Assignment-Topics.md).
> Đánh dấu `[x]` khi xong, ghi kèm link commit/tệp làm bằng chứng.

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
- [ ] **[L]** Cập nhật `.gitignore`: bỏ qua `data/raw/`, `results/**/*.pkl`, `.venv/`, checkpoint notebook.
- [ ] **[Cả nhóm]** Tạo môi trường Python 3.10+ (`python -m venv .venv`), `pip install -r requirements.txt`, xác nhận import được `sklearn`, `shap`, `pandas`.
- [ ] **[R]** Rút gọn `requirements.txt` còn các gói thực sự dùng; bổ sung `interpret` nếu làm phần EBM.
- [ ] **[Cả nhóm]** Chọn hình thức trình bày: A (thuyết trình 20' + Q&A) hoặc B (video 10').

## Giai đoạn 1 — Proposal (nộp cuối tuần 1) ◆

- [ ] **[L]** Rút gọn proposal xuống **1–2 trang** (đề yêu cầu); chuyển chi tiết sang `docs/analysis_and_discussion.md`.
- [ ] **[L]** Thay A/B/C bằng tên + MSSV trong bảng tiến độ của README.
- [ ] **[T]** Đọc nhanh Lundberg & Lee (2017), chốt **hình/bảng/thí nghiệm gốc** sẽ đối chiếu khi tái hiện (ghi locator: Figure/Section).
- [ ] **[T]** Bổ sung tài liệu InterpretML (Nori et al., 2019) vào danh mục tham khảo.
- [ ] **[Cả nhóm]** Chốt hướng **bonus** (bước 6): ví dụ EBM/GAM, bootstrap ổn định giải thích, hoặc fairness theo nhóm.
- [ ] **[L]** Xuất proposal ra PDF và nộp qua LMS.

## Giai đoạn 2 — Tài liệu & baseline (tuần 2–3)

### 2.1 Dữ liệu
- [ ] **[R]** Viết `src/data.py`: tải Adult (`adult.data`, `adult.test`) từ UCI, lưu vào `data/raw/`.
- [ ] **[R]** Ghi `data/README` hoặc `results/data_meta.json`: URL nguồn, SHA-256, số dòng, tỷ lệ nhãn train/test.
- [ ] **[R]** Chuẩn hóa nhãn (`>50K.` → `>50K`), `?` → missing, quy tắc xử lý bản ghi trùng.
- [ ] **[R]** Notebook `notebooks/view_data.ipynb`: EDA cơ bản (phân bố nhãn, missing, biến số/phân loại, tương quan).

### 2.2 Pipeline & baseline
- [ ] **[R]** `src/preprocess.py`: `ColumnTransformer` (median/mode impute, one-hot, `StandardScaler` cho Logistic); bỏ `fnlwgt`, `education`.
- [ ] **[R]** `src/cv.py`: tạo Stratified 5-fold cố định theo seed {42, 2026, 5175}, lưu chỉ số fold.
- [ ] **[R]** `src/metrics.py`: ROC-AUC, AP, F1, balanced accuracy, accuracy, log-loss (ngưỡng 0.5).
- [ ] **[R]** `configs/baseline.yaml` + `experiments/run_baseline.py`: Logistic L2 `C=1`, 3 seed.
- [ ] **[R]** Ghi metadata máy (CPU, RAM, OS, phiên bản thư viện) vào `results/env.json`.

### 2.3 Lý thuyết
- [ ] **[T]** Hệ thống hóa các hướng tiếp cận (mô hình nội tại vs hậu nghiệm: Lasso, Elastic Net, GAM/EBM, LIME, SHAP) kèm ưu/nhược điểm.
- [ ] **[T]** Viết phần toán: loss Logistic + penalty, solver SAGA, GBDT (gradient boosting trên log-loss), Shapley values và 3 tính chất (local accuracy, missingness, consistency).
- [ ] **[L]** Báo cáo tiến độ ngắn (kết quả baseline) — nộp **cuối tuần 3** ◆.

## Giai đoạn 3 — Tái hiện & thực nghiệm (tuần 4–6)

### 3.1 Tuning mô hình chính
- [ ] **[R]** `configs/tuning_*.yaml` cho L1, L2, Elastic Net, GBDT (12 cấu hình mỗi biến thể, như README).
- [ ] **[R]** `experiments/run_tuning.py`: `GridSearchCV` theo ROC-AUC, refit toàn train, đánh giá test; lưu `cv_results_`, tham số tốt nhất, dự đoán test, thời gian tuning/refit/inference.
- [ ] **[R]** Ghi lại convergence warning và cấu hình lỗi (không loại âm thầm).
- [ ] **[R]** Đếm kích thước mô hình: hệ số + intercept (Logistic); số cây/nút/lá (GBDT); dung lượng file model.
- [ ] **[R]** `src/aggregate.py`: tổng hợp mean ± sample std qua 3 seed, chênh lệch theo cặp Δ_s.

### 3.2 Ablation (bắt buộc)
- [ ] **[R]** `configs/ablation.yaml` + `experiments/run_ablation.py`: L1/L2/EN tại `C=1`, `l1_ratio=0.5`; đo hiệu năng, số hệ số |β|>1e-6, refit time.

### 3.3 Giải thích
- [ ] **[T]** `src/explain_linear.py`: bảng hệ số, odds ratio, nhóm tham chiếu, tỷ lệ chọn và tỷ lệ cùng dấu qua seed.
- [ ] **[T]** `src/explain_shap.py`: TreeExplainer, 100 background train, ≤500 mẫu test cố định (lưu ID); beeswarm, bar, waterfall.
- [ ] **[T]** Kiểm tra local accuracy: `|f(x) − (φ0 + Σφj)|` trên toàn bộ mẫu giải thích; ghi sai số max.
- [ ] **[T]** Gộp SHAP theo biến gốc (cộng các cột one-hot), lưu cả bản chưa gộp.
- [ ] **[T]** Chọn TP/TN/FP/FN theo quy tắc cố định (ID nhỏ nhất, seed 42) và các mẫu Logistic/GBDT bất đồng.

### 3.4 Tái hiện paper
- [ ] **[L]** Thí nghiệm nhỏ: tính Shapley chính xác bằng liệt kê tập con, so với KernelExplainer/TreeExplainer.
- [ ] **[L]** Tái hiện hình/bảng đã chốt ở Giai đoạn 1; ghi số liệu paper vs nhóm và giải thích chênh lệch.
- [ ] **[L]** Điền bảng mục 6 trong `docs/analysis_and_discussion.md`.

### 3.5 Mở rộng / bonus (chọn ít nhất một)
- [ ] **[T]** EBM (`interpret`) như mô hình glass-box thứ hai; so với Logistic và GBDT.
- [ ] **[R]** Bootstrap 30 lần: độ ổn định hệ số/SHAP (Jaccard top-k, tương quan hạng).
- [ ] **[L]** Fairness theo `sex`/`race`: TPR/FPR theo nhóm, kèm cỡ mẫu.

## Giai đoạn 4 — Phân tích & hoàn thiện (tuần 7–8)

### 4.1 Phân tích
- [ ] **[R]** Điền bảng kết quả chính và bảng ablation trong docs từ `results/`.
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
