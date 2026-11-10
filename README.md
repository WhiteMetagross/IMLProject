# IML Project:

This repository contains the code, result records, figures plus LaTeX source used for the Group 14 IML report on MicroBiConvLSTM. It covers neural baselines, classical baselines, ablation studies, knowledge distillation plus teacher assistant distillation.

## Team:

Group Number: 14.

### Team Members:

1. IIB2024017, Mridankan Mandal.
2. IIB2024001, Aditya Pachauri.
3. IIB2024033, Dhannu Ram Meena.
4. IIB2024012, Ankit Ekka.

## Report:

- `paper/IMLReport.pdf` is the compiled report.
- `paper/IMLReport.tex` is the LaTeX source.
- `paper/img/` contains the report figures.
- `paper/scripts/build_figure4.py` rebuilds the unified four panel training curve figure.

## Experiment Code:

- `models/` contains MicroBiConvLSTM plus its ablation variants.
- `baselines/` contains DeepConvLSTM, TinyHAR plus TinierHAR.
- `data/` exposes the dataset loader interface used by the training programs.
- `nanoharmamba/data/` contains the underlying dataset loaders required by that interface.
- `scripts/` contains neural training, HPO, retraining, ablation plus figure programs.
- `configs/hpo/` contains the selected hyperparameter records used by the reported neural runs.
- `classicalBaselines/` contains logistic regression, a PCA RBF SVM plus XGBoost experiments.
- `KnowledgeDistillation/` contains direct distillation plus teacher assistant student distillation.

## Recorded Results:

- Classical per run, aggregate plus cross paradigm result records are stored in `classicalBaselines/results/data/`.
- The KD configuration plus the completed five seed summary are stored in `KnowledgeDistillation/config.json` plus `KnowledgeDistillation/reported_results.json`.
- Master seed 17 deterministically generates the five experiment seeds used by the classical and KD studies.

## Scope:

- Raw datasets are not committed.
- Generated checkpoints are not committed.
- Device deployment code, device measurements plus memory footprint experiments are outside this coursework repository.
- See `InstallationAndSetup.md` for setup plus validation commands.
- See `CodeBaseIndex.md` for the file map.
