# Reorganise folder theo trách nhiệm

Đợt thay đổi này chỉ tổ chức lại code, config, tests và tài liệu sau refactor. Các thuật toán, siêu tham số, seed, dữ liệu và artifact lịch sử được giữ nguyên. Không cập nhật artifact cũ để thay đường dẫn module/config trong metadata của chúng.

## Cấu trúc hiện tại

```text
src/
├── data/
│   ├── loader.py
│   └── preprocessing.py
├── models/factory.py
├── evaluation/
│   ├── metrics.py
│   └── validation.py
├── explainability/shap_utils.py
└── common/
    ├── config.py
    └── artifacts.py

experiments/
├── data/check.py
├── modeling/
│   ├── baseline.py
│   ├── tuning.py
│   └── ablation.py
├── explainability/
│   ├── explain.py
│   └── stability.py
├── reproduction/reproduce_shap.py
├── analysis/supplementary.py
└── run_suite.py

configs/
├── data/data_check.json
├── modeling/                  # baseline, ablation, tuning_*.json
├── explainability/            # explain.json, stability.json
├── reproduction/              # reproduce_shap.json, reproduce_shap_v2.json
├── analysis/analysis.json
├── suites.json
└── ci_runs.json

tests/
├── data/test_preprocessing.py
├── evaluation/test_validation.py
├── reproduction/test_shap.py
└── test_layout.py
```

Mỗi thư mục Python có `__init__.py`. Config là JSON dữ liệu; các subfolder config không cần là Python package. `docs/` và `reports/` giữ vị trí cũ và được cập nhật tham chiếu.

## Quy tắc import và chạy

- Import logic từ `src.<nhóm>.<module>`, không import file thí nghiệm để dùng làm thư viện mô hình.
- Thí nghiệm chạy bằng `python -m experiments.<nhóm>.<module>` từ gốc repository; không dùng `python path/to/file.py` hoặc sửa `sys.path` trong source.
- `src/common/config.py` giữ hàm `parse_config`; `src/common/artifacts.py` giữ ROOT, output, JSON và metadata. ROOT được điều chỉnh theo độ sâu package mới; source hashes duyệt đệ quy để bao gồm subfolder.
- Chỉ tên config cuối đường dẫn quyết định tên output. `configs/modeling/baseline.json` vẫn tạo `baseline/`; log suite vẫn là `baseline.log`. Tên kết quả `tuning_logistic_l1`, `reproduce_shap_v2`, v.v. không đổi.
- Không có shim cho entry point phẳng cũ. README và workflow dùng tên mới; muốn chạy source của một commit cũ thì checkout commit đó cùng config tương ứng.

Kích hoạt môi trường Python theo [hướng dẫn cài đặt](../README.md#61-cách-chạy-lại) trước khi chạy các lệnh sau từ thư mục gốc.

```powershell
python -m experiments.data.check --config configs/data/data_check.json
python -m experiments.modeling.baseline --config configs/modeling/baseline.json
python -m experiments.modeling.tuning --config configs/modeling/tuning_logistic_l2.json
python -m experiments.explainability.explain --config configs/explainability/explain.json
python -m experiments.reproduction.reproduce_shap --config configs/reproduction/reproduce_shap_v2.json
python -m experiments.analysis.supplementary --config configs/analysis/analysis.json
```

Các lệnh đơn lẻ dùng output mặc định; để giữ mỗi lần chạy riêng, ưu tiên `experiments.run_suite --suite baseline|core|full --run-id <ID mới> --execute`. Bỏ `--execute` để chỉ in kế hoạch. Chi tiết tại [learning guide](learning_guide.md).

## Workflow và manifest

`configs/suites.json` giữ thứ tự và cùng thí nghiệm, đổi task sang dotted module và config sang đường dẫn tương đối theo nhóm. Ví dụ `["modeling.tuning", "modeling/tuning_logistic_l2"]`.

`configs/ci_runs.json` giữ nguyên `runs: []`, không tự bật compute cloud. Khi bật thủ công, mỗi entry dùng `task: modeling.tuning`, `config: configs/modeling/tuning_logistic_l2.json`.

Hai workflow dùng task mới và helper `python -m src.common.config <task>` để tìm config mặc định. Tuning yêu cầu config cụ thể. Workflow manual trước đây cố định checkout `Thuy-branch`; nay checkout ref được chọn khi dispatch, tránh chạy code của nhánh cũ với lệnh package mới. Tên artifact upload lấy basename config. Các workflow chỉ được sửa cục bộ, chưa dispatch lên GitHub.

## Kiểm chứng

- 8 tests pass, gồm bốn test cũ được chia theo package và bốn test về cấu trúc: imports/config trong suite, default workflow, tên output và ROOT/source hash đệ quy.
- So AST của 16 module trước/sau khi bỏ import: giống nhau. Các hàm `parse_config`, `output_dir`, `save_json` cũng giữ nguyên AST. Ngoại lệ hạ tầng có chủ đích: xác định ROOT, duyệt hash đệ quy, đường dẫn log và định tuyến CLI/CI.
- 16 config ngoài suite giữ nguyên nội dung JSON; suite giữ nguyên thứ tự và lựa chọn thí nghiệm.
- SHA-256 xác nhận 265 file dữ liệu/kết quả có trước đợt reorganise không thay đổi. Snapshot và script so sánh nằm trong `results/reproductions/reorganise-check-01/` (local, không commit).
- `reports.build_evidence` tính lại 6 metric từ 21 tệp prediction cũ thành công. Nội dung bảng sinh lại giống `reports/evidence.md`; khác kiểu newline Windows không phải thay đổi dữ liệu.
- Baseline chạy đủ 5-fold × 3 seed qua entry point mới, lưu ở `results/reproductions/reorganise-baseline-01/`. SHA-256 dữ liệu trùng nguồn cũ. So với run local trước reorganise (`refactor-baseline-02`), cả 6 metric trên CV và test đều chênh **0**.
- So với artifact GitHub `baseline-1`, test chỉ lệch log-loss tối đa `1.11e-16`; CV lệch ROC-AUC tối đa `6.45e-8`, AP `1.53e-8`. Các sai khác này đã có trong run local trước reorganise, không phát sinh từ việc di chuyển folder. Bảng đối chiếu chi tiết ở `results/reproductions/reorganise-check-01/baseline_comparison.json`.

Không chạy lại toàn bộ suite full để kiểm tra một thay đổi đường dẫn. Môi trường local chưa có `interpret-core`; import module EBM vẫn được trì hoãn như trước. Việc xác nhận full training hoặc GitHub Actions chạy thành công cần một run tương ứng, không suy ra từ kiểm tra imports.
