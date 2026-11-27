# Nhật ký refactor và kiểm chứng — 2026-11-25

Đường dẫn source/config trong tài liệu đã được cập nhật theo subfolder hiện tại; artifact lịch sử giữ nguyên. Đợt reorganise tiếp theo có kiểm chứng riêng tại [reorganisation.md](reorganisation.md).

## Phạm vi và nguồn công việc

Giữ cấu trúc `src/`, `experiments/`, `configs/` của code Thủy đã làm và các artifact lịch sử. Không thay dataset/split/metric/ngưỡng/không gian tuning chính. Không tạo lịch sử commit thay cho đóng góp thực tế. Thay đổi do Codex hỗ trợ gồm:

- Gom `transform` và factory SHAP explainer vào `src/explainability/shap_utils.py`, dùng lại cho explanation/stability/analysis.
- Thêm `src/evaluation/validation.py`: bắt cảnh báo hội tụ trong worker của từng fold; GridSearchCV vẫn chọn và refit theo ROC-AUC. Giữ các cột CSV cũ và bổ sung diagnostics; kích thước pickle của pipeline có thể tăng nhẹ vì metadata audit.
- Đặt rõ `sampling_budget` cho reproduction v1 (single_feature) và v2 (all_features); đọc được khóa `kernel_l1_reg` của config lịch sử.
- Thêm runner có kế hoạch trước khi chạy, ID riêng, log/status từng bước; từ chối dùng lại run ID. Giữ CI manifest rỗng, không tự kích hoạt job cloud.
- Thêm bảng evidence từ artifact cũ, hướng dẫn học, bản thảo report và sửa diễn giải EBM quá mạnh.

## Các kiểm tra đã thực hiện

| Kiểm tra | Kết quả | Giới hạn |
| --- | --- | --- |
| 4 regression tests (`python -m unittest discover -s tests -v`) | Pass: preprocessing không học từ validation; cảnh báo ở 4 fit CV song song được ghi; lựa chọn/prediction giống pipeline thường; Shapley trò chơi max khớp; config v1/v2 khác ngân sách | Dữ liệu nhỏ, không thay full training |
| Tính lại 6 metric từ artifact cũ | 21 tệp prediction khớp bảng test, tolerance 1e-10 | Kiểm tra nhất quán số liệu, không huấn luyện lại |
| Tải Adult và kiểm tra checksum | SHA-256 và số byte của ba tệp trùng artifact `data_check-1` | Mạng bị sandbox chặn lần đầu; lần chạy lại được cho phép đã thành công |
| Baseline đầy đủ: 5-fold × 3 seed + refit/test | Hoàn thành tại `results/reproductions/refactor-baseline-02/` | Chỉ baseline, chưa phải toàn bộ suite core/full |
| Đối chiếu baseline mới với `baseline-1` | ROC-AUC/AP/F1/balanced accuracy/accuracy chênh 0; max chênh log-loss 1.11e-16 | Không so thời gian Windows local với runner Ubuntu như cải tiến hiệu năng |
| Tuning smoke trên Adult, 2 C × 2 fold, seed 42, n_jobs=2 | Hoàn thành; diagnostics đủ 4 dòng; chọn C=0.1, test AUC ≈ 0.903613 | Config rút gọn để kiểm tra code, không đưa vào bảng kết quả chính 3 seed |
| Reproduction Figure 3 v1/v2 | Chạy dense/sparse tại n_evals=200, rep 0–1; 12 dòng v1 và 20 dòng v2 khớp artifact cũ tới max 4.44e-15 | Chưa chạy lại 200 repetition × 9 budgets |

## Nơi lưu bằng chứng mới

- `results/reproductions/refactor-baseline-01/`: run lỗi vì sandbox chặn mạng; giữ log, không dùng làm kết quả.
- `results/reproductions/refactor-baseline-02/`: run thành công, `run.json`, log, `data_check/`, `baseline/`.
- `results/reproductions/refactor-tuning-smoke-01/tuning_smoke/`: CV results/diagnostics, refit warnings, test metrics; config đầu vào nằm ở thư mục cha.
- `results/reproductions/refactor-figure3-01/`: hai config rút gọn, estimates/summary/plots dùng để so artifact lịch sử.
- `reports/evidence.md`: bảng chính và ablation trích từ artifact lịch sử, kèm hash nguồn. File này không trộn số liệu smoke hoặc baseline mới vào kết quả gốc.

Source hashes trong run metadata phản ánh thời điểm chạy; sau đó có chuẩn hóa line endings và bổ sung preflight cho runner. Những thay đổi này không đổi phép tính mô hình. Các run mới hoàn chỉnh của người học sẽ lưu hash code cuối cùng khi chạy.

## Việc còn lại trước khi coi là tái hiện toàn bộ và nộp bài

1. Cài đủ dependencies nếu chạy EBM: môi trường local hiện chưa có `interpret-core`; runner `full` kiểm tra và báo thiếu trước khi bắt đầu.
2. Tự chạy suite core/full bằng run ID mới; so checksum/config/metric, ghi độ lệch và nguyên nhân. Các thí nghiệm EBM, 30 bootstrap train, 1.000 bootstrap test và toàn bộ Figure 3 chưa được chạy lại trong lần refactor này.
3. Nhóm đọc chéo report, kiểm tra nhận định với biểu đồ và paper; xác nhận proposal đã nộp và cập nhật % đóng góp thực tế.
4. Dàn trang nguồn `reports/report.md` thành PDF 8–12 trang và kiểm tra hình/công thức; làm slide hoặc video. Bản thảo Markdown chưa phải bản nộp cuối.

Hướng dẫn từng bước: [learning_guide.md](learning_guide.md). Mọi thành viên cần tự giải thích được các thay đổi, code và công thức theo mục 4.4.
