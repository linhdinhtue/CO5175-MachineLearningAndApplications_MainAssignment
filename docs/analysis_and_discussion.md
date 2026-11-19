# Analysis and discussion — Topic 1

> Trạng thái: phân tích phương pháp và thiết kế thảo luận trước thực nghiệm. Chưa có dataset tải về, mô hình đã fit hay kết quả đo trong dự án tại thời điểm soạn. Các ô “Chưa đo” là dữ liệu còn thiếu, không phải kết quả bằng 0. Tài liệu này chưa phải phần kết luận thực nghiệm để nộp.

Tài liệu triển khai phần phân tích của [proposal](../README.md), bám mục 4.1–4.4 trong `00_ML-Assignment-Topics.pdf`. Mục tiêu là trả lời ba câu hỏi về hiệu năng, ảnh hưởng của regularization và giá trị/giới hạn của giải thích.

## 1. Phân tích phương pháp

### 1.1. Vì sao dùng Logistic Regression làm baseline?

Với bài toán phân loại nhị phân, mô hình biểu diễn log-odds bằng hàm tuyến tính:

$$
\log\frac{p(y=1\mid x)}{1-p(y=1\mid x)}=b+\sum_j\beta_jx_j.
$$

Khi giữ các biến còn lại cố định, tăng một đơn vị ở biến đầu vào làm odds nhân với $e^{\beta_j}$. Với biến số được chuẩn hóa, một đơn vị là một độ lệch chuẩn của dữ liệu train; với one-hot, hệ số phải đọc tương đối với nhóm tham chiếu đã bỏ. Đây là diễn giải của mô hình, không chứng minh tác động nhân quả. Một hệ số dương cũng không có nghĩa xác suất tăng một lượng cố định trên mọi quan sát.

Tính cộng tính giúp kiểm tra mô hình dễ hơn, nhưng cũng giới hạn biểu diễn quan hệ phi tuyến và tương tác khi chưa thiết kế thêm đặc trưng. Vì vậy, nếu GBDT tốt hơn thì khả năng học phi tuyến là một giả thuyết giải thích, chưa phải nguyên nhân được chứng minh chỉ bằng bảng điểm.

### 1.2. Regularization thay đổi điều gì?

L1 có thể tạo nghiệm thưa; L2 co hệ số; Elastic Net kết hợp hai dạng phạt. Cần xem đồng thời hiệu năng, số hệ số khác 0 và sự thay đổi hệ số giữa seed. Không mặc định ít hệ số đồng nghĩa giải thích tốt hơn: các biến tương quan có thể thay thế nhau, làm đặc trưng được chọn thay đổi dù hiệu năng tương tự. Công thức và phạm vi tuning nằm trong proposal; tài liệu triển khai tham khảo [scikit-learn](https://scikit-learn.org/stable/modules/linear_model.html).

### 1.3. Giải thích nội tại và giải thích hậu nghiệm

Logistic Regression cho phép đọc trực tiếp cấu trúc dự đoán. GBDT kết hợp nhiều cây; attribution hậu nghiệm giúp khảo sát một dự đoán nhưng không biến toàn bộ mô hình thành một quy tắc đơn giản. Cần phân biệt khả năng đọc mô hình, độ trung thực của attribution đối với đầu ra và mức hữu ích đối với người đọc. Chưa có khảo sát người dùng thì không kết luận phương pháp nào “dễ hiểu hơn” bằng bằng chứng thực nghiệm.

## 2. Điều kiện để so sánh có ý nghĩa

Trước khi điền bảng kết quả, xác nhận:

- Tệp train/test, checksum, số mẫu, tỷ lệ lớp và xử lý trùng lặp đã được lưu; test không tham gia lựa chọn mô hình.
- Imputation, encoding và scaling nằm trong pipeline từng CV fold. Lưu danh sách đặc trưng sau biến đổi và nhóm tham chiếu.
- Mỗi mô hình dùng cùng folds trong từng seed 42, 2026, 5175; 12 cấu hình × 5 folds cho mỗi biến thể. Ghi cả lần fit lỗi và convergence warning, không âm thầm loại cấu hình xấu.
- Chọn hyperparameter bằng ROC-AUC CV; refit trên toàn train; dùng ngưỡng 0.5 cho metric cần nhãn. Mọi thay đổi protocol phải được ghi trước khi xem test.
- Log phần cứng, phiên bản thư viện, seed, tham số tối ưu, thời gian và kích thước mô hình. Tách tuning, refit và giải thích.

CV giữ vai trò validation. Ba seed với test cố định phản ánh biến thiên do CV/tối ưu, không phải ba dataset độc lập và không đo đầy đủ bất định khi lấy mẫu quần thể.

## 3. Hiệu năng dự đoán và phân tích sai số

### 3.0. Dữ liệu và baseline (đã đo)

Nguồn: artifact `data_check-1`, `baseline-1` (GitHub Actions run 44458977361, commit `981529a`; Ubuntu, 4 CPU, 15.6 GB RAM; Python 3.10.22, scikit-learn 1.7.2).

**Dữ liệu.** Train 32 561 dòng (24 720 `<=50K` / 7 841 `>50K`, 24.1% dương), test 16 281 dòng (12 435 / 3 846, 23.6% dương). Missing chỉ ở `workclass`, `occupation`, `native-country` (train: 1 836 / 1 843 / 583). Trùng lặp: 24 dòng train, 5 dòng test — giữ nguyên vì Adult không có ID. SHA-256 lưu trong `data_meta.json`.

**Baseline Logistic L2, `C=1`, SAGA** (bỏ `fnlwgt`, `education`; 81 hệ số + 1 intercept; hội tụ sau khoảng 240 vòng lặp):

| Đánh giá | ROC-AUC | AP | F1 | Balanced acc. | Accuracy | Log-loss |
| --- | --- | --- | --- | --- | --- | --- |
| CV 5-fold × 3 seed (15 fold, mean ± std) | 0.9053 ± 0.0029 | 0.7638 ± 0.0070 | 0.6597 ± 0.0089 | 0.7653 ± 0.0060 | 0.8509 ± 0.0035 | 0.3214 ± 0.0043 |
| Test (3 seed, mean ± std) | 0.9033 ± 0.0000 | 0.7580 ± 0.0000 | 0.6561 ± 0.0000 | 0.7639 ± 0.0000 | 0.8523 ± 0.0000 | 0.3205 ± 0.0000 |

Refit khoảng 7.3 s, inference trên 16 281 mẫu khoảng 0.03 s. Confusion matrix test: TN 11 584, FP 851, FN 1 553, TP 2 293; recall lớp dương 0.596 thấp hơn đáng kể so với lớp âm.

**Quan sát quan trọng:** std trên test bằng 0 (khoảng 1e-8) vì Logistic là bài toán lồi; seed chỉ đổi thứ tự duyệt của SAGA nên mọi seed hội tụ về cùng nghiệm. Biến thiên có ý nghĩa nằm ở giữa các fold CV (std ROC-AUC 0.003). Hệ quả: với mô hình tuned, std qua seed chủ yếu đến từ việc chọn siêu tham số khác nhau; cần nói rõ điều này khi diễn giải mean ± std.

### 3.1. Bảng kết quả chính

Mỗi ô metric sẽ là mean ± sample std trên ba seed. Baseline `C=1` được báo cáo riêng trong báo cáo tiến độ; bảng dưới dành cho các mô hình đã tuning.

| Mô hình | ROC-AUC ↑ | AP ↑ | F1 ↑ | Balanced accuracy ↑ | Accuracy ↑ | Log-loss ↓ |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic L1 | 0.9033 ± 0.0002 | 0.7580 ± 0.0002 | 0.6553 ± 0.0008 | 0.7634 ± 0.0005 | 0.8521 ± 0.0003 | 0.3206 ± 0.0002 |
| Logistic L2 | 0.9035 ± 0.0001 | 0.7581 ± 0.0000 | 0.6556 ± 0.0004 | 0.7634 ± 0.0002 | 0.8524 ± 0.0002 | 0.3204 ± 0.0000 |
| Logistic Elastic Net | 0.9033 ± 0.0000 | 0.7581 ± 0.0000 | 0.6559 ± 0.0000 | 0.7637 ± 0.0000 | 0.8523 ± 0.0000 | 0.3206 ± 0.0000 |
| GBDT | 0.9270 ± 0.0000 | 0.8248 ± 0.0000 | 0.7068 ± 0.0000 | 0.7952 ± 0.0000 | 0.8730 ± 0.0000 | 0.2782 ± 0.0000 |

Nguồn: artifact `tuning_logistic_l1-3`, `tuning_logistic_l2-3`, `tuning_logistic_elasticnet-3`, `tuning_gbdt-3` (`test_summary.csv`). “0.0000” nghĩa là std < 5e-5.

| Mô hình | Siêu tham số chọn (seed 42 / 2026 / 5175) | CV ROC-AUC (mean 3 seed) | Kích thước mô hình | Tuning (s, 4 CPU) | Refit (s) |
| --- | --- | --- | --- | --- | --- |
| L1 | `C` = 1.52 / 1.52 / 0.53 | 0.9053 | 65 / 65 / 56 hệ số khác 0 trên 81 + 1 intercept; ~5 KB | 304 | 8.0 |
| L2 | `C` = 0.19 / 0.53 / 0.53 | 0.9053 | 81 hệ số + 1 intercept; ~5 KB | 449 | 5.2 |
| Elastic Net | `C`=1, `l1_ratio`=0.8 (cả 3 seed) | 0.9053 | 64 hệ số khác 0 + 1 intercept; ~5 KB | 239 | 9.3 |
| GBDT | `n_estimators`=200, `learning_rate`=0.1, `max_depth`=4 (cả 3 seed) | 0.9285 | 200 cây, 5 938 nút, 3 069 lá (3 069 giá trị lá); ~502 KB | 146 | 10.0 |

Không có lần fit lỗi, không có convergence warning khi refit, không mô hình nào chạm `max_iter`. Thời gian tuning là wall-clock với `n_jobs=-1` trên runner GitHub 4 CPU.

**Nhận xét sơ bộ (RQ1, RQ2):**

- GBDT hơn mọi biến thể Logistic một cách nhất quán trên cả 3 seed và 6 metric: ΔROC-AUC ≈ +0.024, ΔAP ≈ +0.067, ΔF1 ≈ +0.051, Δlog-loss ≈ −0.042. Confusion matrix test: GBDT giảm FN từ ~1 555 xuống 1 355 và FP từ ~849 xuống 712. Khoảng cách này lớn hơn nhiều so với std giữa các fold CV (≈ 0.002–0.003).
- Ba biến thể Logistic gần như trùng nhau về hiệu năng (chênh ROC-AUC < 0.0003). CV ROC-AUC của L1 phẳng trong khoảng `C` ∈ [0.19, 100] (0.9048–0.9051), nên regularization ở mức vừa phải chủ yếu ảnh hưởng đến độ thưa, không đến hiệu năng.
- L1 và Elastic Net loại 16–25 hệ số mà không mất hiệu năng. Số hệ số của L1 thay đổi giữa seed (65 vs 56) do `C` được chọn khác nhau — cần kiểm tra biến nào bị loại.
- Độ biến thiên giữa seed trên test gần 0 vì các mô hình gần như tất định (Logistic lồi; GBDT không dùng `subsample`/`max_features`), seed chỉ đổi cách chia fold. Mean ± std qua seed vì vậy không phản ánh bất định lấy mẫu; dùng std giữa fold CV hoặc bootstrap test nếu cần khoảng tin cậy.
- **Hạn chế cần báo cáo:** cấu hình GBDT tốt nhất nằm ở **góc lưới** (giá trị lớn nhất của cả 3 tham số), CV AUC còn tăng đơn điệu theo độ phức tạp. Do đó khoảng cách GBDT–Logistic có thể đang bị ước lượng thấp. Phân tích độ nhạy với lưới mở rộng (`tuning_gbdt_extended.json`) được thêm **sau khi đã xem kết quả test**, nên phải trình bày là phân tích bổ sung, không thay bảng chính.
- Logistic “tốt nhất theo validation” (CV ROC-AUC trung bình 3 seed): Elastic Net 0.90534 > L1 0.90533 > L2 0.90530 — chênh lệch không đáng kể; Elastic Net (`C`=1, `l1_ratio`=0.8) được dùng làm đại diện Logistic cho phần SHAP.

Với metric \(m_s\) ở seed \(s\):

\[
\bar m=\frac{1}{3}\sum_{s=1}^{3}m_s,\qquad
s_m=\sqrt{\frac{\sum_{s=1}^{3}(m_s-\bar m)^2}{2}}.
\]

Báo cáo thêm \(\Delta_s=m_{GBDT,s}-m_{Logistic,s}\) cho từng biến thể Logistic. Nếu muốn gọi một biến thể là “Logistic tốt nhất”, phải chọn bằng validation, không chọn từ test. Diễn giải đồng thời hướng chênh lệch, độ lớn và tính nhất quán qua seed. Mean ± std với ba seed không tự tạo ra kiểm định ý nghĩa thống kê; hai dải std chồng lấn hoặc không chồng lấn không đủ để kết luận có/không có khác biệt.

Accuracy mô tả tỷ lệ dự đoán đúng nhưng có thể che khuất sai số lớp thiểu số. ROC-AUC đánh giá thứ hạng; AP bổ sung góc nhìn precision–recall; F1 phụ thuộc ngưỡng; log-loss đánh giá chất lượng xác suất dự đoán. Nếu các metric không thống nhất, thảo luận theo mục đích sử dụng thay vì chỉ chọn metric có lợi.

### 3.2. Sai số cục bộ

Lưu confusion matrix, precision/recall từng lớp và ID quan sát. Với mỗi mô hình, chọn tối đa một TP, TN, FP, FN bằng quy tắc cố định: ID nhỏ nhất trong mỗi nhóm ở seed 42; nếu nhóm rỗng thì ghi rõ. Phân tích các trường hợp này mang tính minh họa, không đại diện cho toàn bộ test. Bổ sung các quan sát mà Logistic và GBDT dự đoán khác nhau để khảo sát giả thuyết về tương tác/phi tuyến.

## 4. Diễn giải hệ số và SHAP

### 4.1. Bảng hệ số cần xuất

Xuất tên biến sau encoding, đơn vị/nhóm tham chiếu, hệ số từng seed, odds ratio, tỷ lệ được chọn và tỷ lệ cùng dấu. Dùng ngưỡng \(|\beta|>10^{-6}\) để đếm hệ số hoạt động. Tách biến số và biến dummy khi so sánh độ lớn; không xem hệ số lớn nhất là đặc trưng quan trọng nhất một cách tự động. Hệ số có regularization không đi kèm suy luận p-value cổ điển nếu chưa có phương pháp phù hợp.

### 4.2. SHAP toàn cục và cục bộ

Theo [Lundberg & Lee (2017)](https://arxiv.org/abs/1705.07874), SHAP gán đóng góp cho đặc trưng trong một mô hình giải thích cộng tính. Kiểm tra tính cộng:

$$
f(x)\approx\phi_0+\sum_j\phi_j(x).
$$

Lưu rõ \(f\) là raw score hay xác suất, lớp đang giải thích, background và sai số số học. Không cộng attribution trong log-odds rồi diễn giải trực tiếp thành phần trăm xác suất. Thiết lập dự kiến là TreeExplainer với 100 mẫu background từ train, `feature_perturbation="interventional"`, giải thích raw output; kiểm tra API/đầu ra trên phiên bản đã cài. Cách xử lý biến phụ thuộc và lựa chọn đầu ra là thành phần của định nghĩa lời giải thích, xem [TreeExplainer](https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html).

Trên tối đa 500 mẫu test cố định, vẽ beeswarm và bar plot mean absolute SHAP; waterfall cho các trường hợp ở mục 3.2. Mẫu dùng giải thích phải lưu ID và phân bố nhãn. Mean absolute SHAP cho biết độ lớn đóng góp trung bình, không biểu thị dấu ảnh hưởng.

Để báo cáo theo biến gốc, cộng SHAP của các cột one-hot thuộc cùng biến cho từng quan sát rồi mới lấy trung bình trị tuyệt đối. Lưu cả attribution chưa gộp; đây là phép tổng hợp để báo cáo, không mặc định tương đương với việc tính Shapley lại trên một trò chơi có biến được gộp. Không so sánh trực tiếp trị tuyệt đối hệ số Logistic với trị tuyệt đối SHAP vì chúng khác định nghĩa và đơn vị.

Khi hệ số và SHAP ưu tiên các biến khác nhau, kiểm tra scaling, tương quan, nhóm tham chiếu, phân bố dữ liệu và khả năng phi tuyến. Sự khác nhau không tự chứng minh một phương pháp giải thích sai.

## 5. Ablation và đánh đổi hiệu năng–diễn giải

Ablation chính giữ `C=1`, cùng split/pipeline/solver, thay penalty L1/L2/Elastic Net; Elastic Net dùng `l1_ratio=0.5`. Đây là so sánh tại một mức thiết lập cố định, không hàm ý ba penalty có độ mạnh hiệu dụng bằng nhau. Bảng tuned ở mục 3 trả lời câu hỏi hiệu năng sau lựa chọn siêu tham số; bảng ablation trả lời tác động của thay penalty trong protocol đã cố định.

| Penalty | ROC-AUC (mean ± std) | Số hệ số hoạt động (mean ± std) | Refit time (mean ± std) |
| --- | --- | --- | --- |
| L1 | 0.9033 ± 0.0000 | 63 ± 0 | 11.23 ± 0.03 s |
| L2 | 0.9033 ± 0.0000 | 81 ± 0 | 5.48 ± 0.02 s |
| Elastic Net | 0.9033 ± 0.0000 | 66 ± 0 | 7.59 ± 0.01 s |

Nguồn: artifact `ablation-3` (`ablation_summary.csv`). Tại `C=1`, đổi penalty không làm thay đổi hiệu năng (chênh ROC-AUC ~2e-5) nhưng L1 loại 18/81 hệ số, Elastic Net loại 15/81; đổi lại, SAGA cần nhiều vòng lặp hơn (L1 ~440, EN ~288, L2 ~240) nên refit chậm hơn gấp ~2 lần.

Thảo luận theo các khả năng, chưa khẳng định chúng đã xảy ra:

- Nếu L1 giảm mạnh số hệ số mà hiệu năng gần tương đương: cân nhắc lợi ích của mô hình gọn, đồng thời kiểm tra biến được chọn có ổn định không.
- Nếu GBDT cải thiện hiệu năng nhất quán: đối chiếu mức tăng với chi phí fit/SHAP và nhu cầu giải trình.
- Nếu hiệu năng gần nhau nhưng GBDT đắt hơn: Logistic có thể phù hợp khi ưu tiên tính đơn giản; cần nêu tiêu chí ứng dụng thay vì kết luận chung cho mọi bài toán.
- Nếu kết quả nhạy với seed: báo cáo độ không ổn định, kiểm tra hội tụ và phân bố folds trước khi diễn giải thứ hạng mô hình.

Ghi số hệ số + intercept của Logistic; số cây/nút/lá và giá trị lá của GBDT, cùng kích thước model lưu trên đĩa. Không đặt các loại số đếm khác bản chất vào một cột “parameters” mà thiếu định nghĩa. Thời gian inference đo trên cùng số mẫu, cùng máy và chế độ luồng.

## 6. Tái hiện paper và đối chiếu

Đề yêu cầu tái hiện kết quả chính của paper, ghi chênh lệch và giải thích. Kiểm tra local accuracy đơn lẻ chỉ xác nhận một thuộc tính, **chưa đáp ứng toàn bộ yêu cầu tái hiện kết quả chính**. Trước thực nghiệm phải hoàn thiện bảng dưới từ bản paper đã đọc; nếu phạm vi nguyên bản vượt tài nguyên, nêu rõ với giảng viên trong proposal.

| Nội dung đối chiếu | Trạng thái / bằng chứng cần có |
| --- | --- |
| Hình/bảng/thí nghiệm chính trong paper | Chưa chốt; cần ghi locator, metric và thiết lập gốc |
| Dữ liệu, mô hình, cách tính attribution | Cần ghi phiên bản và các giả định so với paper |
| Số liệu công bố | Chưa trích; không dùng số ước lượng từ trí nhớ |
| Kết quả tái hiện, chênh lệch | Chưa chạy; báo cáo cùng đơn vị và protocol |
| Thực nghiệm GBDT trên Adult | Ứng dụng phương pháp; không gọi là tái hiện benchmark gốc |

Với bài toán nhỏ, có thể tính tổng Shapley bằng liệt kê tập con dưới cùng định nghĩa giá trị tập con và background, rồi đo sai số so với explainer. Đây là kiểm tra triển khai bổ sung. Chênh lệch số liệu paper chỉ có ý nghĩa khi metric và thiết lập tương thích; nếu khác dataset thì phân tích định tính và nói rõ không so trực tiếp.

## 7. Mở rộng, hạn chế và kết luận hiện tại

**Bootstrap (tùy chọn):** resample train 30 lần, cố định hyperparameter đã chọn và tập quan sát để giải thích; refit toàn pipeline. Đo tần suất chọn biến, mức nhất quán dấu, Jaccard top-k và tương quan thứ hạng. Phân biệt bất định do resampling với thay background SHAP; giữ background cố định nếu chỉ muốn nghiên cứu ảnh hưởng refit. Đây là ổn định có điều kiện trên hyperparameter đã chọn, không gồm bất định do tuning lại.

**Fairness (tùy chọn):** báo cáo cỡ mẫu, tỷ lệ lớp, TPR/FPR theo nhóm tại cùng ngưỡng; mẫu số bằng 0 ghi không xác định. Không kết luận công bằng chỉ vì bỏ biến nhạy cảm: biến khác có thể chứa thông tin liên quan. Các phân tích này chỉ nhằm đánh giá mô hình học thuật trên dataset, không chứng minh phù hợp triển khai thực tế.

**Hạn chế dự kiến:** một dataset không đại diện mọi bài toán; test cố định và ba seed chưa bao phủ toàn bộ bất định; ngân sách tuning bằng số lần fit không bằng chi phí; lời giải thích nhạy với biến tương quan và background; độ thưa không phải thước đo đầy đủ của khả năng hiểu; chưa có thực nghiệm người dùng hoặc bằng chứng nhân quả.

**Kết luận hiện tại:** đã xác định protocol và các bằng chứng cần thu thập. Chưa có cơ sở chọn mô hình tốt nhất, xác định biến quan trọng nhất hay kết luận GBDT đánh đổi hiệu năng lấy khả năng diễn giải ở mức nào. Sau khi chạy, thay các ô chưa đo bằng số liệu có nguồn, thêm hình và đường dẫn config/log, rồi viết kết luận theo RQ1–RQ3.

## 8. Bằng chứng cần bàn giao và khai báo AI

Các đầu ra dự kiến trong `results/` gồm metric theo seed, dự đoán test, CV results, config, metadata máy, bảng hệ số, SHAP values/ID/background, ablation và figures. Đây là danh sách cần tạo, chưa phải tệp đã tồn tại. Mọi số liệu trong báo cáo phải truy được tới đầu ra tương ứng.

OpenAI Codex hỗ trợ soạn tài liệu phân tích và thiết kế thảo luận này. Nhóm chịu trách nhiệm đọc paper, kiểm chứng phép đo, hiểu công thức/mã nguồn và ghi đúng đóng góp thực tế, theo mục 4.4 của đề.
