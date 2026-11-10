# Code Base Index:

This index maps the files used for the experiments and report.

## Core Models:

- `models/microBiConvLstm.py` defines MicroBiConvLSTM.
- `models/microBiConvLstmVariants.py` defines the controlled ablation variants.
- `baselines/deepConvLstm.py` defines DeepConvLSTM.
- `baselines/tinyHar.py` defines TinyHAR.
- `baselines/tinierHar.py` defines TinierHAR.

## Data Loaders:

- `data/` provides the loader interface imported by the experiment programs.
- `nanoharmamba/data/` provides the complete loaders for the eight reported datasets.
- Dataset roots are relative to `datasets/` at the repository root.

## Neural Experiments:

- `configs/hpo/` contains the 32 selected model and dataset hyperparameter records.
- `scripts/trainMicroBiConvLstm.py` trains MicroBiConvLSTM with one or more seeds.
- `scripts/trainBaselines.py` trains the three neural baselines.
- `scripts/hpoMicroBiConvLstm.py` runs MicroBiConvLSTM hyperparameter search.
- `scripts/hpoBaselines.py` runs baseline hyperparameter search.
- `scripts/runPaperRetraining.py` plus its shell launcher coordinate a selected report retraining run.
- `scripts/ablationStudiesMicroBiConvLstm.py` runs the architectural ablations.
- `scripts/trainingCurveArtifacts.py` saves loss histories plus curve images.
- `scripts/create_paper_figures.py` creates the report comparison figures.

## Classical Experiments:

- `classicalBaselines/trainClassicalBaselines.py` extracts features, fits models plus records five seed metrics.
- `classicalBaselines/profileClassicalComplexity.py` records compact estimator state plus analytical complexity.
- `classicalBaselines/benchmarkCrossParadigmEfficiency.py` measures complete raw window inference on one pinned CPU thread.
- `classicalBaselines/readTsc.c` provides the timestamp counter reader used by the host benchmark.
- `classicalBaselines/results/data/` contains the report result records.

## Knowledge Distillation:

- `KnowledgeDistillation/config.json` defines datasets, model routes, five seeds plus training settings.
- `KnowledgeDistillation/recover.py` recovers source checkpoints with recorded training overrides.
- `KnowledgeDistillation/train.py` implements supervised, direct KD plus assistant route stages.
- `KnowledgeDistillation/run_study.py` plans and runs the complete study.
- `KnowledgeDistillation/reported_results.json` contains the completed report summary.
- `KnowledgeDistillation/tests/` contains focused KD tests.

## Report:

- `paper/IMLReport.tex` is the manuscript source.
- `paper/IMLReport.pdf` is the compiled six page report.
- `paper/img/` contains every included figure.
- `paper/scripts/build_figure4.py` combines four source curves into Figure 4.

## Excluded Material:

- Raw datasets, generated checkpoints plus run caches are excluded.
- Device deployment sources plus device result files are excluded.
- Memory footprint programs plus result files are excluded.
