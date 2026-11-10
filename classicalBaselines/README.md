# Classical HAR Baselines and Cross Paradigm Efficiency:

The study evaluates multinomial logistic regression, a PCA RBF SVM plus XGBoost against MicroBiConvLSTM. Each classical model receives the same raw sensor windows used by the neural models. The pipeline extracts 21 deterministic time, frequency plus distribution features per sensor channel. Scaling, PCA plus estimator fitting use the training split only.

Master seed 17 deterministically generates five random experiment seeds. No held out labels are used to fit preprocessing transforms or estimator parameters.

## Programs:

- `trainClassicalBaselines.py` extracts features, fits three estimators, evaluates macro F1 plus writes resumable run records.
- `profileClassicalComplexity.py` derives compact learned state sizes plus analytical inference complexity.
- `benchmarkCrossParadigmEfficiency.py` measures one complete raw window on one pinned CPU thread.
- `readTsc.c` provides the serialized x86 timestamp counter reader used by the host benchmark.
- `tests/testClassicalBaselines.py` checks feature shape, deterministic seeds plus stratified sample capping.

## Classical Training:

```bash
python classicalBaselines/trainClassicalBaselines.py \
  --models all \
  --datasets all \
  --seeds 5 \
  --master-seed 17 \
  --run-name classicalMaster17FiveSeeds
```

Append `--dry-run` to inspect the plan without fitting estimators. Model containers plus feature caches remain generated artifacts.

## Cross Paradigm Measurement:

Compile the timestamp counter helper under WSL before running the benchmark.

```bash
gcc -O3 -shared -fPIC classicalBaselines/readTsc.c -o results/crossParadigm/libReadTsc.so

python classicalBaselines/benchmarkCrossParadigmEfficiency.py \
  --classical-run results/classicalBaselines/classicalMaster17FiveSeeds \
  --micro-root results/knowledge_distillation/checkpoints/microbi_convlstm \
  --tsc-library results/crossParadigm/libReadTsc.so \
  --output-dir results/crossParadigm \
  --datasets pamap2 wisdm unimib ucihar skoda
```

The classical paths include feature extraction, FFT operations, scaling, PCA when required plus prediction. MicroBiConvLSTM includes its complete forward pass. State size uses a compact FP32 equivalent inference representation.

## Reported Efficiency Results:

| Model | Macro F1 higher | State KiB lower | Mcycles per window lower | F1 per KiB higher | F1 per Mcycle higher |
|---|---:|---:|---:|---:|---:|
| MicroBiConvLSTM | 82.96% | 42.83 | 0.491 | 1.937 | 169.081 |
| Logistic Regression | 77.04% | 12.91 | 1.065 | 5.966 | 72.338 |
| PCA RBF SVM | 82.90% | 1,160.99 | 1.631 | 0.071 | 50.828 |
| XGBoost | 79.74% | 864.63 | 2.404 | 0.092 | 33.171 |

All values average five seeds across PAMAP2, WISDM, UniMiB, UCI HAR plus SKODA.

## Saved Results:

- `results/data/classicalBaselineSummary.json` contains aggregate accuracy, macro F1, balanced accuracy plus fit time.
- `results/data/classicalBaselinePerRunResults.json` plus its CSV form contain all classical runs.
- `results/data/classicalComplexity.json` plus its CSV and Markdown forms preserve estimator state plus complexity results.
- `results/data/crossParadigmEfficiency.json` plus its CSV and Markdown forms preserve raw window measurements.
- `results/data/crossParadigmEfficiencySummary.csv` contains the report summary table.

Classical estimators do not use neural epoch training. Their evaluation therefore does not create neural training and validation loss curves.
