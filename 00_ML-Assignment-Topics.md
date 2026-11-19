<!-- Trang PDF 1 -->

  M A C H I N E L E A R N I N G A N D A P P L I C A T I O N S
  Danh sách chủ đề Bài tập lớn (Assignment)

   Chương trình Thạc sĩ Khoa học Máy tính · Mô tả chủ đề, hướng dẫn thực hiện và tiêu chí đánh giá



   6–8 tuần        2–3           18         ≥ 2 ngày
      Thời gian thực hiện          Thành viên mỗi nhóm         Chủ đề trong 6 nhóm      Nộp báo cáo trước ngày thi
                                     (khuyến khích)                                                     cuối kỳ


Mục lục

1. Giới thiệu..............................................................................................................................................2

2. Tổng quan danh sách chủ đề.................................................................................................................2

3. Mô tả chi tiết từng chủ đề.....................................................................................................................3
   3.1 Nhóm I – Data Analytics.................................................................................................................................3
   3.2 Nhóm II – Forecasting.....................................................................................................................................5
   3.3 Nhóm III – Computer Vision...........................................................................................................................6
   3.4 Nhóm IV – OOD / NCD / GCD......................................................................................................................... 8
   3.5 Nhóm V – Continual Learning.........................................................................................................................9
   3.6 Nhóm VI – Mathematical Fundamentals......................................................................................................10

4. Hướng dẫn thực hiện.......................................................................................................................... 12
   4.1 Quy trình 6 bước..........................................................................................................................................12
   4.2 Yêu cầu về thực nghiệm...............................................................................................................................12
   4.3 Yêu cầu về mã nguồn....................................................................................................................................12
   4.4 Sử dụng công cụ AI....................................................................................................................................... 12

5. Sản phẩm nộp & tiêu chí đánh giá.......................................................................................................13
   5.1 Sản phẩm nộp...............................................................................................................................................13
   5.2 Rubric đánh giá (thang 10)...........................................................................................................................14

6. Lịch trình & quy định........................................................................................................................... 14
   6.1 Lộ trình 6–8 tuần..........................................................................................................................................14
   6.2 Hình thức trình bày...................................................................................................................................... 15
   6.3 Quy định....................................................................................................................................................... 15

<!-- Trang PDF 2 -->

                                               Machine Learning and Applications | Danh sách chủ đề Bài tập lớn (Assignment)

 1. Giới thiệu

Danh sách gồm 18 chủ đề trong 6 nhóm, bám sát nội dung giảng dạy (regression, SVM, Neural Network,
Deep Neural Network) và ba hướng ứng dụng chính: Data Analysis, Forecasting, Computer Vision.

 • Đối tượng: học viên cao học ngành Khoa học Máy tính, làm việc theo nhóm 2–3 người (khuyến khích)
    trong 6–8 tuần.
 • Mục tiêu: đọc hiểu và tái hiện (reproduce) một phương pháp từ tài liệu nghiên cứu, thực nghiệm trên
   benchmark chuẩn, phân tích kết quả có phản biện và đề xuất cải tiến hoặc ứng dụng.
 • Mức độ: mỗi chủ đề có yêu cầu tối thiểu (đủ điểm đạt) và hướng mở rộng (điểm cao/bonus), để các
   nhóm có nền tảng khác nhau đều chọn được đề phù hợp. Yêu cầu tối thiểu đã được cân chỉnh cho
   nhóm 2–3 người trong 6–8 tuần.

 2. Tổng quan danh sách chủ đề

Bảng dưới tóm tắt 18 chủ đề theo nhóm, mô hình liên quan trong môn học và độ khó ước lượng.

Độ khó: Trung bình Khá Khó – ước lượng tương đối để cân nhắc khi đăng ký.

  #   Chủ đề                           Mô hình trong môn học             Độ khó

 Nhóm I – Data Analytics

  1    Interpretable regression & explainability              Linear/Logistic regression, GAM         Trung bình

  2   SVM & kernel methods at scale                  SVM, kernel, Logistic regression         Trung bình

  3   Deep learning framework for tabular data          MLP, DNN, GBDT                       Khá

 Nhóm II – Forecasting

  4   Time series forecasting using machine learning       Linear models, GBDT, RNN/LSTM        Trung bình

        Transformers & foundation models for time series
  5                                                             Linear, Transformer                    Khó
        forecasting

         Probabilistic forecasting & uncertainty
  6                                                       Quantile regression, DNN                Khá
        quantification

 Nhóm III – Computer Vision

  7   CNN vs Vision Transformer for image classification   CNN, ViT                             Trung bình

                                               DNN, Logistic regression (linear
  8    Self-supervised & transfer learning with few labels                                        Khá
                                                        probe)

  9    Visual anomaly detection for industrial inspection   CNN features, kNN, One-class SVM        Khá

  10    Adversarial robustness of deep networks          CNN, DNN                           Khó



                                                    Trang 2 / 15

<!-- Trang PDF 3 -->

                                               Machine Learning and Applications | Danh sách chủ đề Bài tập lớn (Assignment)


  #   Chủ đề                           Mô hình trong môn học             Độ khó

 Nhóm IV – OOD / NCD / GCD

  11    Out-of-Distribution Detection (OOD)           DNN                                 Khá

  12   Novel Class Discovery (NCD)                    DNN, clustering                       Khó

  13   Generalized Category Discovery (GCD)            DNN, ViT, clustering                    Khó

 Nhóm V – Continual Learning

        Continual learning using representation &
  14                                     DNN                                Khó
        contrastive learning

  15   Continual learning with game theory           DNN                                Khó

 Nhóm VI – Mathematical Fundamentals

       Sharpness-Aware Minimization (SAM), flat minima
  16                                            DNN, optimization                     Khó
       và generalization

  17    Diffusion maps                                       Kernel, spectral methods                Khó

  18    Calibration of neural networks                          Logistic regression, DNN                Trung bình


 3. Mô tả chi tiết từng chủ đề

Mỗi chủ đề được trình bày thành một khối gồm: mô tả, yêu cầu tối thiểu, hướng mở rộng, dữ liệu gợi ý, tài
liệu tham khảo và mã nguồn.

  Lưu ý chung
   • Dataset: nhóm có thể đề xuất dataset khác với gợi ý, miễn là phù hợp với bài toán đang giải.
   • Tài liệu: khuyến khích tham khảo thêm các tài liệu liên quan ngoài danh sách đã gợi ý, và trích dẫn
      đầy đủ trong báo cáo.


3.1 Nhóm I – Data Analytics

 Chủ đề 1. Interpretable regression & explainability

 Mô hình: Linear/Logistic regression, GAM    Độ khó: Trung bình

 Mô tả             So sánh mô hình dễ diễn giải (Linear/Logistic regression với L1/L2/Elastic Net, GAM/EBM) với
              mô hình hộp đen (GBDT, MLP) được giải thích bằng SHAP; phân tích đánh đổi accuracy –
                         interpretability.

 Yêu cầu tối thiểu    • Triển khai regularized regression, chọn siêu tham số bằng cross-validation.
                 • So sánh với ít nhất một mô hình hộp đen; trình bày hệ số mô hình và SHAP values.


                                                    Trang 3 / 15

<!-- Trang PDF 4 -->

                                              Machine Learning and Applications | Danh sách chủ đề Bài tập lớn (Assignment)


Hướng mở rộng    • Kiểm tra độ ổn định của giải thích (bootstrap); fairness giữa các nhóm dân số.

Dữ liệu gợi ý       UCI Adult, German Credit, MIMIC (nếu có quyền truy cập).

Tài liệu tham         [1]  Regression Shrinkage and Selection via the Lasso (Tibshirani, 1996)
khảo                   [2]  Regularization and Variable Selection via the Elastic Net (Zou & Hastie, 2005)
                         [3]  A Unified Approach to Interpreting Model Predictions (Lundberg & Lee, 2017) –
                           https://arxiv.org/abs/1705.07874
                         [4]  InterpretML: A Unified Framework for Machine Learning Interpretability (Nori et al., 2019) –
                           https://arxiv.org/abs/1909.09223

Mã nguồn         • https://github.com/shap/shap
                • https://github.com/interpretml/interpret



Chủ đề 2. SVM & kernel methods at scale

Mô hình: SVM, kernel, Logistic regression    Độ khó: Trung bình

Mô tả             Nghiên cứu SVM tuyến tính và phi tuyến (kernel trick) cùng các kỹ thuật mở rộng cho dữ liệu
                      lớn: Random Fourier Features, Nyström approximation, GPU SVM.

Yêu cầu tối thiểu    • So sánh SVM (linear, RBF), Logistic regression và MLP về accuracy, thời gian huấn luyện và
                   bộ nhớ khi số mẫu tăng dần.

Hướng mở rộng    • Phân tích sai số xấp xỉ kernel theo số features; SVM trên đặc trưng trích từ mạng
                        pretrained.

Dữ liệu gợi ý       20 Newsgroups, Covertype, MNIST/Fashion-MNIST.

Tài liệu tham         [1]  Support-Vector Networks (Cortes & Vapnik, 1995)
khảo                   [2]  Random Features for Large-Scale Kernel Machines (Rahimi & Recht, NeurIPS 2007)
                         [3]  LIBSVM: A Library for Support Vector Machines (Chang & Lin) –
                           https://www.csie.ntu.edu.tw/~cjlin/libsvm/

Mã nguồn         • https://github.com/Xtra-Computing/thundersvm
                • scikit-learn: module kernel_approximation



Chủ đề 3. Deep learning framework for tabular data

Mô hình: MLP, DNN, GBDT    Độ khó: Khá

Mô tả            Đánh giá các kiến trúc deep learning cho dữ liệu bảng (MLP, ResNet, FT-Transformer, TabPFN)
                   so với GBDT (XGBoost, LightGBM, CatBoost), có kiểm định thống kê khi so sánh.

Yêu cầu tối thiểu    •  Ít nhất 2 dataset, 2 mô hình DL + 1 GBDT, cùng ngân sách tuning.
                • Báo cáo mean ± std qua nhiều seed.

Hướng mở rộng    • Phân tích khi nào DL thắng GBDT (kích thước dữ liệu, feature heterogeneity); thử tabular
                      foundation model trên dữ liệu nhỏ.


                                                   Trang 4 / 15

<!-- Trang PDF 5 -->

                                               Machine Learning and Applications | Danh sách chủ đề Bài tập lớn (Assignment)


  Tài liệu tham         [1]  PyTorch Tabular: A Framework for Deep Learning with Tabular Data (Joseph, 2021) –
 khảo                      https://arxiv.org/abs/2104.13638
                          [2]  A Closer Look at Deep Learning Methods on Tabular Datasets (Ye et al., 2024) –
                            https://arxiv.org/abs/2407.00956
                          [3]  Accurate predictions on small data with a tabular foundation model (Hollmann et al., Nature
                          2025) – https://www.nature.com/articles/s41586-024-08328-6

 Mã nguồn         • https://github.com/LAMDA-Tabular/Tabular-Survey


3.2 Nhóm II – Forecasting

 Chủ đề 4. Time series forecasting using machine learning

 Mô hình: Linear models, GBDT, RNN/LSTM    Độ khó: Trung bình

 Mô tả             Xây dựng pipeline dự báo doanh số từ baseline thống kê đến ML: seasonal naive, ARIMA,
                       Linear regression với lag features, GBDT, LSTM.

 Yêu cầu tối thiểu    • Feature engineering (lag, rolling, calendar).
                 • Đánh giá bằng time-series cross-validation (rolling origin); báo cáo MAE/RMSE/MASE.

 Hướng mở rộng    • Dự báo phân cấp (hierarchical reconciliation); ảnh hưởng của ngày lễ/khuyến mãi.

 Dữ liệu gợi ý       Walmart Sales; M5 – https://www.kaggle.com/competitions/m5-forecasting-accuracy

  Tài liệu tham         [1]  Walmart Sales Time Series Forecasting Using Machine Learning (GitHub project) –
 khảo                     https://github.com/abhinav-bhardwaj/Walmart-Sales-Time-Series-Forecasting-Using-
                          Machine-Learning



 Chủ đề 5. Transformers & foundation models for time series forecasting

 Mô hình: Linear, Transformer    Độ khó: Khó

 Mô tả            Kiểm chứng tranh luận “Transformer có thực sự hiệu quả cho dự báo?”: so sánh mô hình
                    tuyến tính (DLinear), Transformer (PatchTST) và foundation model zero-shot (Chronos,
                    TimesFM).

 Yêu cầu tối thiểu    • Tái hiện DLinear và PatchTST trên ít nhất 2 benchmark (ETT, Weather, Electricity), nhiều
                         horizon.

 Hướng mở rộng    • Zero-shot vs fine-tune foundation model trên dữ liệu thực tế của Việt Nam (điện năng, giá
                     nông sản, chất lượng không khí).

  Tài liệu tham         [1]  Are Transformers Effective for Time Series Forecasting? (Zeng et al., 2022) –
 khảo                      https://arxiv.org/abs/2205.13504
                          [2]  A Time Series is Worth 64 Words: Long-term Forecasting with Transformers (Nie et al., 2022) –
                            https://arxiv.org/abs/2211.14730
                          [3]  Chronos: Learning the Language of Time Series (Ansari et al., 2024) –
                            https://arxiv.org/abs/2403.07815
                          [4]  A decoder-only foundation model for time-series forecasting (Das et al., 2023) –

                                                    Trang 5 / 15

<!-- Trang PDF 6 -->

                                               Machine Learning and Applications | Danh sách chủ đề Bài tập lớn (Assignment)


                            https://arxiv.org/abs/2310.10688

 Mã nguồn         • https://github.com/thuml/Time-Series-Library
                 • https://github.com/amazon-science/chronos-forecasting



 Chủ đề 6. Probabilistic forecasting & uncertainty quantification

 Mô hình: Quantile regression, DNN    Độ khó: Khá

 Mô tả             Thay vì dự báo một điểm, ước lượng khoảng dự báo bằng quantile regression, DeepAR và
                    conformal prediction.

 Yêu cầu tối thiểu    • So sánh ít nhất 2 phương pháp theo coverage, interval width và CRPS/pinball loss.

 Hướng mở rộng    • Conformal prediction cho chuỗi có distribution shift; ứng dụng vào bài toán tồn kho hoặc
                     năng lượng.

  Tài liệu tham         [1]  DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks (Salinas et al.,
 khảo                    2017) – https://arxiv.org/abs/1704.04110
                          [2]  A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty
                             Quantification (Angelopoulos & Bates, 2021) – https://arxiv.org/abs/2107.07511

 Mã nguồn         • https://github.com/awslabs/gluonts
                 • https://github.com/scikit-learn-contrib/MAPIE


3.3 Nhóm III – Computer Vision

 Chủ đề 7. CNN vs Vision Transformer for image classification

 Mô hình: CNN, ViT    Độ khó: Trung bình

 Mô tả             So sánh inductive bias của CNN (ResNet, ConvNeXt) và ViT theo kích thước dữ liệu, data
                    augmentation và chi phí tính toán.

 Yêu cầu tối thiểu    • Fine-tune CNN và ViT pretrained trên 1 dataset (kèm huấn luyện từ đầu với một mô hình
                    nhỏ để so sánh).
                 • Vẽ learning curve theo % dữ liệu; trực quan hóa bằng Grad-CAM/attention map.

 Hướng mở rộng    • Đánh giá robustness với corruption (CIFAR-10-C) hoặc shape/texture bias.

 Dữ liệu gợi ý        CIFAR-10/100, Oxford-IIIT Pets, một bộ ảnh y tế/nông nghiệp.

  Tài liệu tham         [1]  Deep Residual Learning for Image Recognition (He et al., 2015) –
 khảo                      https://arxiv.org/abs/1512.03385
                          [2]  An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (Dosovitskiy et
                                        al., 2020) – https://arxiv.org/abs/2010.11929
                          [3]  A ConvNet for the 2020s (Liu et al., 2022) – https://arxiv.org/abs/2201.03545

 Mã nguồn         • https://github.com/huggingface/pytorch-image-models


                                                    Trang 6 / 15

<!-- Trang PDF 7 -->

                                              Machine Learning and Applications | Danh sách chủ đề Bài tập lớn (Assignment)



Chủ đề 8. Self-supervised & transfer learning with few labels

Mô hình: DNN, Logistic regression (linear probe)    Độ khó: Khá

Mô tả              Khai thác biểu diễn từ mô hình tự giám sát (SimCLR, DINOv2) và vision-language (CLIP) khi chỉ
                  có ít nhãn.

Yêu cầu tối thiểu    • Linear probe (Logistic regression/SVM trên embedding), kNN và fine-tune với 1%, 10%,
                  100% nhãn.
                • So sánh với supervised baseline.

Hướng mở rộng    • Zero-shot CLIP, prompt engineering; ứng dụng vào domain có ít nhãn (ảnh y tế, bệnh cây
                         trồng).

Tài liệu tham         [1]  A Simple Framework for Contrastive Learning of Visual Representations (Chen et al., 2020) –
khảo                      https://arxiv.org/abs/2002.05709
                         [2]  DINOv2: Learning Robust Visual Features without Supervision (Oquab et al., 2023) –
                           https://arxiv.org/abs/2304.07193
                         [3]  Learning Transferable Visual Models From Natural Language Supervision (Radford et al.,
                        2021) – https://arxiv.org/abs/2103.00020

Mã nguồn         • https://github.com/facebookresearch/dinov2
                • https://github.com/openai/CLIP



Chủ đề 9. Visual anomaly detection for industrial inspection

Mô hình: CNN features, kNN, One-class SVM    Độ khó: Khá

Mô tả              Phát hiện lỗi sản phẩm khi chỉ có ảnh mẫu “bình thường”, dựa trên đặc trưng CNN pretrained
                  + kNN/memory bank (PatchCore) hoặc One-class SVM.

Yêu cầu tối thiểu    • Tái hiện PatchCore và một baseline (One-class SVM hoặc autoencoder) trên MVTec AD.
                • Báo cáo image-level và pixel-level AUROC.

Hướng mở rộng    • Phân tích tốc độ suy luận để triển khai trên dây chuyền; liên hệ với bài toán OOD (chủ đề
                         11).

Tài liệu tham         [1]  Towards Total Recall in Industrial Anomaly Detection (Roth et al., 2021) –
khảo                      https://arxiv.org/abs/2106.08265
                         [2]  MVTec AD dataset – https://www.mvtec.com/company/research/datasets/mvtec-ad

Mã nguồn         • https://github.com/openvinotoolkit/anomalib



Chủ đề 10. Adversarial robustness of deep networks

Mô hình: CNN, DNN    Độ khó: Khó

Mô tả             Nghiên cứu tấn công đối kháng (FGSM, PGD) và phòng thủ bằng adversarial training; liên hệ


                                                   Trang 7 / 15

<!-- Trang PDF 8 -->

                                               Machine Learning and Applications | Danh sách chủ đề Bài tập lớn (Assignment)


                      với margin của SVM.

 Yêu cầu tối thiểu    • Tấn công một CNN chuẩn trên CIFAR-10, huấn luyện lại bằng PGD adversarial training.
                 • Báo cáo clean vs robust accuracy theo epsilon.

 Hướng mở rộng    • Đánh giá bằng AutoAttack/RobustBench; phân tích đánh đổi robustness – accuracy.

  Tài liệu tham         [1]  Explaining and Harnessing Adversarial Examples (Goodfellow et al., 2014) –
 khảo                      https://arxiv.org/abs/1412.6572
                          [2]  Towards Deep Learning Models Resistant to Adversarial Attacks (Madry et al., 2017) –
                            https://arxiv.org/abs/1706.06083
                          [3]  RobustBench: a standardized adversarial robustness benchmark (Croce et al., 2020) –
                            https://arxiv.org/abs/2010.09670

 Mã nguồn         • https://github.com/RobustBench/robustbench


3.4 Nhóm IV – OOD / NCD / GCD

 Chủ đề 11. Out-of-Distribution Detection (OOD)

 Mô hình: DNN    Độ khó: Khá

 Mô tả              Phát hiện các mẫu đầu vào nằm ngoài phân phối huấn luyện, để mô hình không đưa ra dự
                  đoán sai với độ tự tin cao.

 Yêu cầu tối thiểu    • So sánh ít nhất 3 phương pháp post-hoc (MSP, Energy, Mahalanobis/kNN) trên một cặp
                    ID/OOD chuẩn.
                 • Báo cáo AUROC và FPR@95.

 Hướng mở rộng    • Near-OOD vs far-OOD; OOD với backbone pretrained lớn (CLIP).

  Tài liệu tham         [1]  Generalized Out-of-Distribution Detection: A Survey (Yang et al., 2021) –
 khảo                      https://arxiv.org/abs/2110.11334

 Mã nguồn         • https://github.com/Jingkang50/OODSurvey (benchmark: OpenOOD)



 Chủ đề 12. Novel Class Discovery (NCD)

 Mô hình: DNN, clustering    Độ khó: Khó

 Mô tả              Phát hiện và phân cụm các lớp mới trong dữ liệu chưa gán nhãn, dựa trên tri thức học được
                    từ các lớp đã biết.

 Yêu cầu tối thiểu    • Tái hiện một phương pháp NCD trên CIFAR-10/100.
                 • Đánh giá bằng clustering accuracy (Hungarian matching), so với baseline k-means trên đặc
                         trưng.

  Tài liệu tham         [1]  Novel Class Discovery: an Introduction and Key Concepts (Troisemaine et al., 2023) –
 khảo                      https://arxiv.org/abs/2302.12028


                                                    Trang 8 / 15

<!-- Trang PDF 9 -->

                                               Machine Learning and Applications | Danh sách chủ đề Bài tập lớn (Assignment)


 Mã nguồn         • https://github.com/JosephKJ/Awesome-Novel-Class-Discovery



 Chủ đề 13. Generalized Category Discovery (GCD)

 Mô hình: DNN, ViT, clustering    Độ khó: Khó

 Mô tả         Mở rộng của NCD: dữ liệu chưa gán nhãn chứa cả lớp đã biết lẫn lớp mới; mô hình phải nhận
                     diện lớp cũ và khám phá lớp mới cùng lúc.

 Yêu cầu tối thiểu    • Tái hiện GCD với backbone ViT (DINO) trên một benchmark generic và một benchmark
                           fine-grained.
                 • Báo cáo accuracy cho All/Old/New classes.

 Hướng mở rộng    • Ước lượng số lớp mới khi chưa biết trước.

  Tài liệu tham         [1]  Generalized Category Discovery (Vaze et al., 2022) – https://arxiv.org/abs/2201.02609
 khảo

 Mã nguồn         • https://github.com/JosephKJ/Awesome-Novel-Class-Discovery


3.5 Nhóm V – Continual Learning

 Chủ đề 14. Continual learning using representation & contrastive learning

 Mô hình: DNN    Độ khó: Khó

 Mô tả            Học tuần tự nhiều tác vụ mà không quên kiến thức cũ (catastrophic forgetting), dựa trên học
                     biểu diễn và contrastive learning.

 Yêu cầu tối thiểu    • So sánh fine-tuning tuần tự, một phương pháp replay (ER) và Co2L/CCLIS trên Split CIFAR-
                         10.
                 • Báo cáo average accuracy và forgetting.

  Tài liệu tham         [1]  A Comprehensive Survey of Continual Learning: Theory, Method and Application (Wang et al.,
 khảo                    2023) – https://arxiv.org/abs/2302.00487
                          [2]  Contrastive Continual Learning with Importance Sampling and Prototype-Instance Relation
                                 Distillation (Li et al., 2024) – https://arxiv.org/abs/2403.04599
                          [3]  Co²L: Contrastive Continual Learning (Cha et al., 2021) – https://arxiv.org/abs/2106.14413

 Mã nguồn         • https://github.com/lijy373/CCLIS
                 • https://github.com/chaht01/Co2L



 Chủ đề 15. Continual learning with game theory

 Mô hình: DNN    Độ khó: Khó

 Mô tả         Mô hình hóa continual learning dưới góc nhìn lý thuyết trò chơi, cân bằng giữa tính ổn định


                                                    Trang 9 / 15

<!-- Trang PDF 10 -->

                                               Machine Learning and Applications | Danh sách chủ đề Bài tập lớn (Assignment)


                           (stability) và khả năng học mới (plasticity).

 Yêu cầu tối thiểu    • Mô hình hóa đánh đổi stability–plasticity như một trò chơi.
                 • Triển khai và so sánh với một baseline continual learning.

  Tài liệu tham         [1]  A Comprehensive Survey of Continual Learning: Theory, Method and Application (Wang et al.,
 khảo                    2023) – https://arxiv.org/abs/2302.00487
                          [2]  Applications of game theory in deep learning: a survey (Hazra & Anjaria, Multimedia Tools
                       and Applications 2022) –
                           https://gametheory.fit.hanu.vn/wp-content/uploads/2024/03/s11042-022-12153-2-1.pdf
                          [3]  End-to-End Learning and Intervention in Games (Li et al., 2020) –
                            https://arxiv.org/abs/2010.13834
                          [4]  What Game Are We Playing? End-to-end Learning in Normal and Extensive Form Games (Ling
                             et al., IJCAI 2018) – https://www.ijcai.org/proceedings/2018/0055.pdf
                          [5]  Bài báo trên OpenReview – https://openreview.net/forum?id=u1XV9BPAB9


3.6 Nhóm VI – Mathematical Fundamentals

 Chủ đề 16. Sharpness-Aware Minimization (SAM), flat minima và generalization

 Mô hình: DNN, optimization    Độ khó: Khó

 Mô tả          SAM được đề xuất với kỳ vọng nghiệm “phẳng” sẽ tổng quát hóa tốt hơn. Chủ đề bổ sung góc
                     nhìn phản biện từ Schliserman, Vansover-Hager & Koren (OPT 2025): trong smooth stochastic
                    convex optimization, cực tiểu phẳng không đảm bảo tổng quát hóa tốt. SA-ERM có thể trả về
                   cực tiểu phẳng với population risk Ω(1) trong khi một cực tiểu nhọn lại tổng quát hóa tối ưu;
                  SA-GD có thể đạt cực tiểu phẳng nhưng vẫn tổng quát hóa kém; SAM có thể hội tụ về cực tiểu
                    nhọn. Nhóm cần đối chiếu các kết luận lý thuyết này với thực nghiệm trên mạng sâu.

 Cập nhật SAM



                                  Bán kính r, learning rate η, empirical loss F_S trên tập huấn luyện S.

 Yêu cầu tối thiểu    • Lý thuyết: trình bày SA-ERM, SA-GD, SAM và định nghĩa ρ-flatness; tóm tắt và giải thích
                         trực quan các định lý chính của paper OPT 2025 (không yêu cầu chứng minh lại).
                 • Thực nghiệm: so sánh SGD với SAM và một biến thể (ASAM hoặc F-SAM) trên CIFAR-10 với
                   một ResNet nhỏ; đo sharpness (top Hessian eigenvalue hoặc m-sharpness), test accuracy
                       và generalization gap qua ít nhất 3 seed.
                 • Phân tích: sharpness có tương quan nhất quán với generalization gap không, và kết quả
                      thực nghiệm phù hợp hay mâu thuẫn với kết luận lý thuyết ở điểm nào.

 Hướng mở rộng    • Mô phỏng số một bài toán convex nhỏ (ví dụ hồi quy tuyến tính/logistic nhiều chiều, ít
                    mẫu) để minh họa trường hợp cực tiểu phẳng nhưng tổng quát hóa kém.
                 • Khảo sát ảnh hưởng của bán kính r và learning rate η; thảo luận vì sao kết luận trong
                      convex có thể khác với deep learning.

  Tài liệu tham         [1]   Flat Minima and Generalization: Insights from Stochastic Convex Optimization (Schliserman,
 khảo                    Vansover-Hager & Koren, OPT 2025) – https://opt-ml.org/papers/2025/paper53.pdf (arXiv:
                            https://arxiv.org/abs/2511.03548)


                                                   Trang 10 / 15

<!-- Trang PDF 11 -->

                                              Machine Learning and Applications | Danh sách chủ đề Bài tập lớn (Assignment)


                         [2]  Sharpness-Aware Minimization for Efficiently Improving Generalization (Foret et al., 2020) –
                           https://arxiv.org/abs/2010.01412
                         [3]  Friendly Sharpness-Aware Minimization (Li et al., 2024) – https://arxiv.org/abs/2403.12350
                         [4]  When Do Flat Minima Optimizers Work? (Kaddour et al., 2022) –
                           https://arxiv.org/abs/2202.00661

Mã nguồn         • https://github.com/google-research/sam
                • https://github.com/nblt/F-SAM



Chủ đề 17. Diffusion maps

Mô hình: Kernel, spectral methods    Độ khó: Khó

Mô tả            Phương pháp giảm chiều phi tuyến dựa trên quá trình khuếch tán (random walk) trên đồ thị
                dữ liệu, bảo toàn cấu trúc hình học của manifold.

Yêu cầu tối thiểu    • Triển khai diffusion maps, so sánh với PCA, kernel PCA, t-SNE/UMAP trên dữ liệu tổng hợp
                        (Swiss roll) và dữ liệu thật.
                • Phân tích ảnh hưởng của kernel bandwidth.

Tài liệu tham         [1]  Diffusion maps (Coifman & Lafon, Applied and Computational Harmonic Analysis 2006) –
khảo                     https://www.sciencedirect.com/science/article/pii/S1063520306000546
                         [2]  A Survey of Geometric Optimization for Deep Learning: From Euclidean Space to Riemannian
                         Manifold (Fei et al., ACM Computing Surveys 2025) –
                          https://dl.acm.org/doi/epdf/10.1145/3708498
                         [3]  Deep Diffusion Maps (García-Heredia et al., 2025) – https://arxiv.org/abs/2505.06087



Chủ đề 18. Calibration of neural networks

Mô hình: Logistic regression, DNN    Độ khó: Trung bình

Mô tả           Mạng sâu hiện đại thường quá tự tin. Chủ đề nối Logistic regression (Platt scaling) với DNN
                   (temperature scaling), đồng thời giới thiệu hiện tượng double descent.

Yêu cầu tối thiểu    • Đo ECE và vẽ reliability diagram cho Logistic regression, MLP, ResNet.
                • Áp dụng temperature scaling và ít nhất một phương pháp khác (label smoothing, MC
                    Dropout hoặc deep ensembles).

Hướng mở rộng    • Calibration dưới distribution shift; quan hệ giữa độ rộng mô hình, double descent và
                          calibration.

Tài liệu tham         [1]  On Calibration of Modern Neural Networks (Guo et al., 2017) –
khảo                      https://arxiv.org/abs/1706.04599
                         [2]  Deep Double Descent: Where Bigger Models and More Data Hurt (Nakkiran et al., 2019) –
                           https://arxiv.org/abs/1912.02292





                                                  Trang 11 / 15

<!-- Trang PDF 12 -->

                                               Machine Learning and Applications | Danh sách chủ đề Bài tập lớn (Assignment)

 4. Hướng dẫn thực hiện

4.1 Quy trình 6 bước

Mỗi nhóm thực hiện theo 6 bước dưới đây; bước 2 (proposal) là điểm kiểm tra bắt buộc trước khi làm thực
nghiệm.

 1. Chọn đề và lập nhóm. Khuyến khích nhóm 2–3 học viên, cử một trưởng nhóm. Đăng ký 2 nguyện vọng;
    mỗi chủ đề tối đa 2 nhóm, ưu tiên theo thứ tự đăng ký.

 2. Viết proposal (1–2 trang, nộp cuối tuần 1). Nêu bài toán, paper chính sẽ tái hiện, dataset, baseline,
    metric, phân công công việc và tài nguyên tính toán dự kiến (GPU, Colab/Kaggle).

 3. Đọc tài liệu và hệ thống hóa. Tóm tắt các hướng tiếp cận chính, ưu/nhược điểm, và lý do chọn phương
    pháp của nhóm. Trình bày phần toán học cốt lõi (hàm mục tiêu, thuật toán tối ưu).

 4. Xây dựng baseline trước. Chạy một mô hình đơn giản (Linear/Logistic regression, SVM hoặc mô hình
    chuẩn của bài toán) để có mốc so sánh và kiểm tra pipeline dữ liệu. Nộp báo cáo tiến độ ngắn (kết quả
    baseline) vào cuối tuần 3.

 5. Tái hiện và thực nghiệm. Tái hiện kết quả chính của paper, ghi lại chênh lệch so với số liệu công bố và
     giải thích. Thực hiện ablation study cho ít nhất một thành phần quan trọng.

 6. Cải tiến hoặc ứng dụng (khuyến khích, có điểm thưởng). Đề xuất một thay đổi về phương pháp, hoặc
    áp dụng vào một bài toán/dataset thực tế mới, kèm thực nghiệm chứng minh.

4.2 Yêu cầu về thực nghiệm

 • Chia dữ liệu train/validation/test rõ ràng; với chuỗi thời gian phải chia theo thời gian, không xáo trộn.
 • Chạy ít nhất 3 random seed cho kết quả chính, báo cáo mean ± std.
 • Các mô hình được so sánh phải có ngân sách tuning tương đương; nêu rõ không gian siêu tham số đã
    thử.
 • Ghi lại cấu hình phần cứng, thời gian huấn luyện và số tham số mô hình.

4.3 Yêu cầu về mã nguồn

 • Python 3.10+, PyTorch hoặc scikit-learn; dùng Git và commit thường xuyên (giảng viên có thể xem lịch
   sử đóng góp).
 • Có README.md hướng dẫn cài đặt và chạy lại toàn bộ kết quả, requirements.txt hoặc
   environment.yml, file config cho từng thí nghiệm.
 • Được dùng code từ repo công khai nhưng phải ghi nguồn và nêu rõ phần nào nhóm tự viết hoặc sửa
    đổi.

4.4 Sử dụng công cụ AI

   • Được dùng công cụ AI để hỗ trợ viết code và tra cứu, nhưng phải khai báo trong báo cáo (công cụ
      nào, dùng cho phần nào).
   • Mọi thành viên phải giải thích được mã nguồn và công thức trong báo cáo.



                                                   Trang 12 / 15

<!-- Trang PDF 13 -->

                                               Machine Learning and Applications | Danh sách chủ đề Bài tập lớn (Assignment)

 5. Sản phẩm nộp & tiêu chí đánh giá

5.1 Sản phẩm nộp

Mỗi nhóm được đánh giá qua báo cáo đầy đủ và một trong hai hình thức trình bày (A: thuyết trình + Q&A
trên lớp; B: video thuyết trình). Trưởng nhóm nộp một thư mục qua LMS gồm:

 1. Báo cáo đầy đủ (PDF, khuyến khích LaTeX, khoảng 8–12 trang chưa kể phụ lục):
      a) Introduction – problem statement, ứng dụng, các cách tiếp cận hiện có kèm ưu/nhược điểm, lý do
       chọn phương pháp của nhóm.
     b) Methods/Approaches – mô tả tổng quan, kiến trúc/pipeline, các bước chính, công thức cốt lõi.
      c) Experiments – mô tả cài đặt, benchmark dataset, kết quả, ablation, đánh giá và thảo luận.
     d) Improvement or application proposal (khuyến khích, có điểm thưởng).
      e) Conclusion & limitations.
       f)  References.
      g) Phụ lục: bảng phân công và mức độ đóng góp (%) của từng thành viên; khai báo sử dụng công cụ
          AI.
 2. Slide thuyết trình (phương án A) hoặc video thuyết trình 10 phút (phương án B).
 3. Mã nguồn (hoặc link Git repo) kèm README.

  Hạn nộp
  Chậm nhất 2 ngày trước ngày thi cuối kỳ, nộp qua LMS (trưởng nhóm nộp).





                                                   Trang 13 / 15

<!-- Trang PDF 14 -->

                                               Machine Learning and Applications | Danh sách chủ đề Bài tập lớn (Assignment)

5.2 Rubric đánh giá (thang 10)

 Tiêu chí                          Trọng số    Biểu hiện mức tốt

 Tổng quan & hiểu bài toán          15%     Hệ thống hóa được các hướng tiếp cận, so sánh có phản biện

 Phương pháp & nền tảng toán                     Trình bày đúng, đủ công thức, giải thích được vì sao phương
                              20%
 học                                       pháp hoạt động

                                         Có baseline, nhiều seed, ablation, giải thích kết quả và chênh
 Thực nghiệm & phân tích kết quả     30%
                                                    lệch so với paper

                                            Chạy lại được theo README, cấu trúc rõ ràng, ghi nguồn đầy
 Mã nguồn & khả năng tái lập         10%
                                         đủ

 Tiêu chí trình bày (25%) – chỉ chọn MỘT trong hai phương án cho mỗi nhóm

                                         Đúng thời gian (20 phút + 10 phút Q&A), rõ ràng, mọi thành
 A. Thuyết trình & Q&A             25%
                                                   viên trả lời được câu hỏi

                                               Video 10 phút, mỗi thành viên trình bày một phần; nội dung rõ
  B. Video thuyết trình              25%
                                                     ràng, có demo hoặc kết quả thực nghiệm minh họa

  Cải tiến / ứng dụng mới            Bonus +1    Ý tưởng có cơ sở, có thực nghiệm chứng minh

 Tổng                          100%     Cộng điểm thưởng tối đa +1 (thang 10)


  Lưu ý
  Với mỗi nhóm, giảng viên chỉ chọn một trong hai phương án A hoặc B để chấm tiêu chí trình bày (25%),
  nên tổng điểm luôn là 100%.
  Điểm cá nhân có thể điều chỉnh theo bảng đóng góp và phần trả lời Q&A (phương án A) hoặc phần
   trình bày trong video (phương án B) của từng thành viên.


 6. Lịch trình & quy định

6.1 Lộ trình 6–8 tuần

Bài tập lớn kéo dài 6–8 tuần kể từ khi công bố đề; báo cáo phải nộp chậm nhất 2 ngày trước ngày thi cuối
kỳ. Các mốc trong tuần là đề xuất, giảng viên điều chỉnh theo lịch học kỳ. Lộ trình dưới đây lấy mốc 8 tuần;
nếu chỉ có 6 tuần, giai đoạn tái hiện rút còn tuần 4–5 và tuần 6 dành cho hoàn thiện, trình bày.

        Tuần 1               Tuần 2–3              Tuần 4–6              Tuần 7–8

 Chọn đề, proposal        Tài liệu, baseline          Tái hiện kết quả        Hoàn thiện
  • Lập nhóm 2–3 người     • Hệ thống hóa paper     • Ablation, nhiều seed     • Báo cáo + slide/video
  • Proposal 1–2 trang      • Chạy baseline đơn giản   • Cải tiến / ứng dụng      • Trình bày (A hoặc B)

 ◆ Duyệt proposal – cuối  ◆ Báo cáo tiến độ – cuối   —           ◆ Nộp báo cáo – chậm
  tuần 1                      tuần 3                                                nhất 2 ngày trước thi cuối
                                                                                 kỳ



                                                   Trang 14 / 15

<!-- Trang PDF 15 -->

                                               Machine Learning and Applications | Danh sách chủ đề Bài tập lớn (Assignment)

6.2 Hình thức trình bày

 • Phương án A – Thuyết trình: tuần cuối của bài tập lớn (tuần 6–8 tùy lịch), 20 phút trình bày + 10 phút
   Q&A mỗi nhóm; thứ tự các nhóm công bố trước 1 tuần.
 • Phương án B – Video thuyết trình: video 10 phút nộp cùng báo cáo qua LMS, mỗi thành viên trình bày
   một phần.

6.3 Quy định

 • Nộp trễ: trừ 10% điểm báo cáo mỗi ngày, tối đa 3 ngày.
 • Liêm chính học thuật: sao chép báo cáo/mã nguồn của nhóm khác hoặc không ghi nguồn sẽ bị xử lý
   theo quy định của trường; báo cáo có thể được kiểm tra đạo văn.





                                                   Trang 15 / 15
