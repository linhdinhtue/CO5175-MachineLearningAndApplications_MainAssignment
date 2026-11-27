# Bảng bằng chứng tự sinh từ artifact của nhóm

Đã tính lại 6 metric từ 21 tệp dự đoán; sai số cho phép 1e-10. Đây là kiểm tra artifact cũ, không phải huấn luyện lại. Std dùng ddof=1.

| Model | roc_auc | average_precision | f1 | balanced_accuracy | accuracy | log_loss |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic L1 | 0.9033 ± 0.0002 | 0.7580 ± 0.0002 | 0.6553 ± 0.0008 | 0.7634 ± 0.0005 | 0.8521 ± 0.0003 | 0.3206 ± 0.0002 |
| Logistic L2 | 0.9035 ± 0.0001 | 0.7581 ± 0.0000 | 0.6556 ± 0.0004 | 0.7634 ± 0.0002 | 0.8524 ± 0.0002 | 0.3204 ± 0.0000 |
| Logistic Elastic Net | 0.9033 ± 0.0000 | 0.7581 ± 0.0000 | 0.6559 ± 0.0000 | 0.7637 ± 0.0000 | 0.8523 ± 0.0000 | 0.3206 ± 0.0000 |
| GBDT | 0.9270 ± 0.0000 | 0.8248 ± 0.0000 | 0.7068 ± 0.0000 | 0.7952 ± 0.0000 | 0.8730 ± 0.0000 | 0.2782 ± 0.0000 |
| GBDT extended (exploratory) | 0.9273 ± 0.0000 | 0.8255 ± 0.0000 | 0.7070 ± 0.0002 | 0.7962 ± 0.0002 | 0.8724 ± 0.0001 | 0.2768 ± 0.0000 |
| EBM GAM (1 config) | 0.9267 ± 0.0002 | 0.8254 ± 0.0003 | 0.7069 ± 0.0012 | 0.7958 ± 0.0008 | 0.8727 ± 0.0005 | 0.2768 ± 0.0003 |
| EBM GA2M (1 config) | 0.9272 ± 0.0002 | 0.8266 ± 0.0004 | 0.7088 ± 0.0009 | 0.7973 ± 0.0007 | 0.8732 ± 0.0004 | 0.2758 ± 0.0004 |

## Ablation tại C=1

| Penalty | ROC-AUC | Active coefficients | Refit seconds |
| --- | --- | --- | --- |
| elasticnet | 0.9033 ± 0.0000 | 66.0000 ± 0.0000 | 7.5851 ± 0.0091 |
| l1 | 0.9033 ± 0.0000 | 63.0000 ± 0.0000 | 11.2287 ± 0.0328 |
| l2 | 0.9033 ± 0.0000 | 81.0000 ± 0.0000 | 5.4806 ± 0.0174 |

## Nguồn và SHA-256

Các đường dẫn dưới đây tính từ thư mục artifact đầu vào.

| Artifact file | SHA-256 |
| --- | --- |
| tuning_logistic_l1-3/test_metrics.csv | 8d39ce65680cb16aa424c11757c6e495b5d091f3e823a04457ac77e5c2aaa6a6 |
| tuning_logistic_l1-3/config.json | 45580f541fa2768037a8e7eb2198e8458663f1a584f95f95e5a64171ca0b765a |
| tuning_logistic_l1-3/env.json | 37495140f60ed18b5076949dccb3a1c1feb44332f9140fa1dd93e06d42f89bbe |
| tuning_logistic_l1-3/predictions_seed42.csv | 6e1757460de5ddb31270940e5c151744f388d2d72cf2e4985c6fd262a47c375a |
| tuning_logistic_l1-3/predictions_seed2026.csv | 2c066634c3250a58dae1614d3025399ec0c1cc8d2594498dc19abb64e8e3c969 |
| tuning_logistic_l1-3/predictions_seed5175.csv | 746d1d1529f977e4e67ed3aba07e4e33d39ea24388dfc603e4ac5b97e79a0661 |
| tuning_logistic_l2-3/test_metrics.csv | 2df9d881e8560f7a503c08cff51d2e3dd9ba175a897f4328ed0eed607e0e74ba |
| tuning_logistic_l2-3/config.json | 25f192a1eb9e15a91ef9385294954830df94cd72f7386a8e2779dde0f89eed7e |
| tuning_logistic_l2-3/env.json | 21d2a69022860686a0df45d5ba507b1f042fef9c27b842de2496e667d418f58a |
| tuning_logistic_l2-3/predictions_seed42.csv | 3323ca48f94a6ac83b28858ec1817534d02f4feb489b97d6b42f54923995b1b0 |
| tuning_logistic_l2-3/predictions_seed2026.csv | 0dfdeffa7d42897cb750496cd0a008d52f0c72efbd4ac6fc33ad4629f162e7f6 |
| tuning_logistic_l2-3/predictions_seed5175.csv | 3ff1c6611b619d86f9b264be833b1b1534dfb86a7417b9e0a48d8b00c0a819d4 |
| tuning_logistic_elasticnet-3/test_metrics.csv | 68f00593af49d5b7ec305ee3ebaf31f56abd41df372a155786e5fc09c03a4e89 |
| tuning_logistic_elasticnet-3/config.json | a7ae9ea27fa4ab7e0eb4691b8b0f2e05fd35c3cd9be4390aab7492ebf7fee0d1 |
| tuning_logistic_elasticnet-3/env.json | 37495140f60ed18b5076949dccb3a1c1feb44332f9140fa1dd93e06d42f89bbe |
| tuning_logistic_elasticnet-3/predictions_seed42.csv | 0983a0d0bb504197c7c3416f7933ce7a44c1cddef9bcd7733c8b74a3e7a89f51 |
| tuning_logistic_elasticnet-3/predictions_seed2026.csv | 6a9f7bd38bd977bf357da117beeee6a5c9c6799b852c1f291e0c44ba15ee43a3 |
| tuning_logistic_elasticnet-3/predictions_seed5175.csv | 82789ad7d2e8627bf5814add30520dab0331d6410c26f4d662c53c59a61ec42c |
| tuning_gbdt-3/test_metrics.csv | 5d7cbf26445c6942ce4de863f158969f6ee3ccdd458b2db0714021ca30fa79d4 |
| tuning_gbdt-3/config.json | dd2ae6e003d397f619d3c2814b6f4b5d5a3636e0b289712f4e68fa7997837761 |
| tuning_gbdt-3/env.json | 37495140f60ed18b5076949dccb3a1c1feb44332f9140fa1dd93e06d42f89bbe |
| tuning_gbdt-3/predictions_seed42.csv | 5633f48e4a1796867290023b9a1dc0d7aa96c5cf7039faca64ceaa6da8871137 |
| tuning_gbdt-3/predictions_seed2026.csv | 2f734fb4a2a3cbca60597a1ab5dcf7ac7075214a94f9a3e08ae2a523ae547dc8 |
| tuning_gbdt-3/predictions_seed5175.csv | c647960f2fa7bccf714f4509cc56ed76f44e05ffda4c06c916eaf318a936e523 |
| tuning_gbdt_extended-4/test_metrics.csv | 0af36fe4d15378169844ef0f2b9e61c546e75f298a566d1b81cba13f2dd1a8ef |
| tuning_gbdt_extended-4/config.json | 70f3729d16386faaf777cc39fd372601b34cd19b9b93355f85fed54e029788c1 |
| tuning_gbdt_extended-4/env.json | c66a3455b5570419b95c08e40b1ab656cf50571a4b92deea384c64f6267fe081 |
| tuning_gbdt_extended-4/predictions_seed42.csv | cfa43311cc10ff3115ff3b51b288fa3697c557ccc01f742256358c35e69d2595 |
| tuning_gbdt_extended-4/predictions_seed2026.csv | f281459d0cde1e4b0ecb68bfab4e7c11bdd2e165af00dc1685631192b653b334 |
| tuning_gbdt_extended-4/predictions_seed5175.csv | 23a21617c35509cb52e60bca5a0b14b66b3774d3b139b1c02d8e03cb7a251654 |
| tuning_ebm_gam-7/test_metrics.csv | 91e658bf3478c82709f77f50dc1d000d8595d3293d7f4e25cd0e2555dce5b0ee |
| tuning_ebm_gam-7/config.json | 38a275dea02a1dfec0256e10b90096d51cbefae6ca77f25834f170af08b4c83e |
| tuning_ebm_gam-7/env.json | f4c8aa61c44e0b3f7d819075f06076b50fd8069066e2fd759af25286229d762b |
| tuning_ebm_gam-7/predictions_seed42.csv | c5e44894e96aabdd0db9b6e0ff7b7541f7dc129ce7947240415afe829379aa64 |
| tuning_ebm_gam-7/predictions_seed2026.csv | 3446e4d8f80631f78b929fe307711ff952eaee4f485a026bd4445eccd32b4029 |
| tuning_ebm_gam-7/predictions_seed5175.csv | 839dc15afc37cdad75ce7428991628da797de7de2f02da319a472b6150373364 |
| tuning_ebm_ga2m-7/test_metrics.csv | 5e545d6f4801357e3692c8c64f9b0a851ff0610ff093d2bd1b014bb3ff2e030e |
| tuning_ebm_ga2m-7/config.json | 87c806a4d836378ed6233199db4813e06fef4a050528ec5253a678835be81df7 |
| tuning_ebm_ga2m-7/env.json | f4c8aa61c44e0b3f7d819075f06076b50fd8069066e2fd759af25286229d762b |
| tuning_ebm_ga2m-7/predictions_seed42.csv | 2c2ae03c455c92f3c9f8a2702a2dce0c93861f10d8a41c2b71c28fe5320bd02e |
| tuning_ebm_ga2m-7/predictions_seed2026.csv | 4a0979781f18a7271936939c52622fca6bd0ca12b21ab0ccb0870a83228dafa2 |
| tuning_ebm_ga2m-7/predictions_seed5175.csv | d89fbd149ee19f3e823061837445fffaf4b1c7d8b8a2e1737422df857c5e1a5f |
| ablation-3/ablation_metrics.csv | 6c6d82170fb9c223941bff25fc276efc8b2615466f8737d7ca23837f6b6d4531 |
