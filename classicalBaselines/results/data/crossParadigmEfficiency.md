# Cross-Paradigm Efficiency Results:

All cycle measurements cover one complete raw-window inference. MicroBiConvLSTM includes its full neural forward pass. Classical pipelines include handcrafted feature extraction, scaling, PCA where applicable, and final prediction. Measurements are invariant-TSC reference cycles on one pinned WSL CPU thread.

Model-state size is a compact FP32-equivalent representation, not framework container size. XGBoost uses compact split and leaf arrays.

- CPU affinity: `0`.
- Samples per estimator: `24`.
- Repeats per sample: `5`.
- Measurements per estimator: `120`.

## Cross-Dataset Summary:

| Model | Mean Macro-F1 (%) | Mean State (KiB) | Mean Mcycles/window | F1 points/KiB | F1 points/Mcycle |
|---|---:|---:|---:|---:|---:|
| microbi_convlstm | 82.96 | 42.83 | 0.491 | 1.937 | 169.081 |
| logistic_regression | 77.04 | 12.91 | 1.065 | 5.966 | 72.338 |
| rbf_svm | 82.90 | 1160.99 | 1.631 | 0.071 | 50.828 |
| xgboost | 79.74 | 864.63 | 2.404 | 0.092 | 33.171 |

## Per-Dataset Results:

| Model | Dataset | Macro-F1 (%) | State (KiB) | Mcycles/window | F1 points/KiB | F1 points/Mcycle |
|---|---|---:|---:|---:|---:|---:|
| microbi_convlstm | pamap2 | 65.02 | 45.38 | 0.444 | 1.433 | 146.337 |
| microbi_convlstm | wisdm | 80.87 | 39.23 | 0.471 | 2.062 | 171.704 |
| microbi_convlstm | unimib | 79.23 | 39.80 | 0.460 | 1.991 | 172.194 |
| microbi_convlstm | ucihar | 93.01 | 41.10 | 0.504 | 2.263 | 184.703 |
| microbi_convlstm | skoda | 96.67 | 48.62 | 0.574 | 1.988 | 168.333 |
| logistic_regression | pamap2 | 59.20 | 21.87 | 1.141 | 2.707 | 51.875 |
| logistic_regression | wisdm | 75.12 | 1.99 | 0.889 | 37.709 | 84.550 |
| logistic_regression | unimib | 62.16 | 2.74 | 0.903 | 22.667 | 68.848 |
| logistic_regression | ucihar | 92.70 | 5.93 | 1.057 | 15.633 | 87.715 |
| logistic_regression | skoda | 96.01 | 32.04 | 1.335 | 2.997 | 71.893 |
| rbf_svm | pamap2 | 77.23 | 1965.10 | 1.930 | 0.039 | 40.025 |
| rbf_svm | wisdm | 80.11 | 480.56 | 1.290 | 0.167 | 62.079 |
| rbf_svm | unimib | 67.55 | 711.19 | 1.410 | 0.095 | 47.921 |
| rbf_svm | ucihar | 92.71 | 493.66 | 1.451 | 0.188 | 63.883 |
| rbf_svm | skoda | 96.92 | 2154.43 | 2.074 | 0.045 | 46.721 |
| xgboost | pamap2 | 56.49 | 897.25 | 2.798 | 0.063 | 20.188 |
| xgboost | wisdm | 84.61 | 623.82 | 1.948 | 0.136 | 43.443 |
| xgboost | unimib | 65.27 | 1113.53 | 2.479 | 0.059 | 26.329 |
| xgboost | ucihar | 92.81 | 408.63 | 1.841 | 0.227 | 50.415 |
| xgboost | skoda | 99.53 | 1279.90 | 2.954 | 0.078 | 33.695 |

## Interpretation:

- State size includes every learned or fitted numeric array required for inference.
- Cycle counts are host implementation measurements from one pinned CPU thread.
- Cross-device claims require rerunning every complete pipeline on the same target hardware and compiler configuration.
