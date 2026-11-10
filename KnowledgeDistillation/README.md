# Knowledge Distillation:

## Scope:

This package reproduces the five seed knowledge distillation study for MicroBiConvLSTM. It supports two routes.

```text
Direct KD: teacher -> MicroBiConvLSTM.
Assistant KD: teacher -> assistant -> MicroBiConvLSTM.
```

The final student architecture remains MicroBiConvLSTM in both routes. The study uses the existing held out evaluation loader for checkpoint selection. It does not create a separate validation split.

## Study configuration:

Master seed 17 deterministically generates five random seeds. The launcher rejects seeds outside this configured set. Seed 29 checkpoints are not accepted.

| Dataset | Teacher | Assistant | Student |
| --- | --- | --- | --- |
| PAMAP2 | TinierHAR | TinyHAR | MicroBiConvLSTM |
| WISDM | TinierHAR | DeepConvLSTM | MicroBiConvLSTM |
| UniMiB | DeepConvLSTM | TinierHAR | MicroBiConvLSTM |
| UCI HAR | TinyHAR | TinierHAR | MicroBiConvLSTM |
| SKODA | TinyHAR | TinierHAR | MicroBiConvLSTM |

The default settings are 200 epochs, patience 10, temperature 4.0 and soft loss weight 0.5. The configuration also stores the exact HPO derived learning rate, weight decay and dropout used for every teacher and assistant checkpoint. See `config.json` for the complete mapping.

## Environment:

The recorded experiments used WSL2, Python 3.11.15, PyTorch 2.9.1 with CUDA 13.0 and an NVIDIA GeForce RTX 4060 Laptop GPU. Install the pinned Python packages inside a project Conda environment.

```bash
conda create -n microbikd python=3.11.15
conda run --no-capture-output -n microbikd python -m pip install -r KnowledgeDistillation/requirements.txt
```

The repository dataset loaders use the included `nanoharmamba.data` package. Raw datasets are not committed. Place them under `datasets/` as described in `InstallationAndSetup.md`.

CUDA kernels can retain limited hardware dependent nondeterminism. The code seeds Python, NumPy, PyTorch and CUDA. It also disables cuDNN benchmarking and enables deterministic cuDNN behavior.

## Verification:

Run the focused tests from the repository root.

```bash
conda run --no-capture-output -n microbikd python -m pytest KnowledgeDistillation/tests -q
```

Run model forward and backward checks without loading datasets.

```bash
conda run --no-capture-output -n microbikd python -m KnowledgeDistillation.train --synthetic-smoke --device cpu
```

## Dry plan:

Planning is the default. This command creates a manifest without training.

```bash
conda run --no-capture-output -n microbikd python -m KnowledgeDistillation.run_study plan --datasets all --routes direct,assistant
```

Inspect the generated manifest under `KnowledgeDistillation/results/plans/` before training.

## Full reproduction:

The following command recovers every teacher, assistant and normal student checkpoint. It then runs direct KD plus assistant KD for all five datasets and all five seeds.

```bash
conda run --no-capture-output -n microbikd python -m KnowledgeDistillation.run_study all --datasets all --routes direct,assistant --execute
```

The full study contains 75 source checkpoint runs, 25 direct KD runs and 50 assistant route training stages. Existing complete stages are reused only when their saved configuration matches the requested configuration.

The high priority datasets can be reproduced first.

```bash
conda run --no-capture-output -n microbikd python -m KnowledgeDistillation.run_study all --datasets pamap2,wisdm --routes direct,assistant --execute
```

## Output structure:

```text
KnowledgeDistillation/results/
  checkpoints/
    baselines/<model>/<dataset>/seed<seed>_checkpoint.pt
    microbi_convlstm/<dataset>/microbi_convlstm_<dataset>_seed<seed>_checkpoint.pt
  plans/study_<timestamp>/
    commands.sh
    manifest.json
    status.json
  runs/
    direct/<dataset>/seed<seed>/
      student_checkpoint.pt
      model_state.pt
      run_config.json
      result.json
    assistant/<dataset>/seed<seed>/
      assistant_checkpoint.pt
      run_config.json
      result.json
      student/
        student_checkpoint.pt
        model_state.pt
        run_config.json
        result.json
    summaries/
      <dataset>_results.json
      <dataset>_summary.json
```

Each KD stage records the source checkpoint SHA256, environment, complete run configuration, loss histories, predictions, confusion matrix, validation metrics and best checkpoint.

## Recorded results:

`reported_results.json` stores the recent five seed means and standard deviations. It is a compact verification target. It does not replace the per seed outputs created by the full command.

The table reports macro F1 in percent. Values are mean plus or minus standard deviation across five seeds. Direct change and assistant change are measured against the normal MicroBiConvLSTM student.

| Dataset | Normal student | Direct KD | Direct change | Assistant student | Assistant change |
| --- | ---: | ---: | ---: | ---: | ---: |
| PAMAP2 | 65.02 plus or minus 2.10 | 67.73 plus or minus 3.33 | +2.71 | 63.58 plus or minus 3.78 | -1.44 |
| WISDM | 80.87 plus or minus 2.71 | 79.89 plus or minus 1.29 | -0.98 | 80.82 plus or minus 2.43 | -0.05 |
| UniMiB | 79.23 plus or minus 5.80 | 80.56 plus or minus 2.79 | +1.33 | 79.42 plus or minus 2.12 | +0.19 |
| UCI HAR | 93.01 plus or minus 0.63 | 92.79 plus or minus 0.32 | -0.22 | 93.24 plus or minus 0.27 | +0.23 |
| SKODA | 96.67 plus or minus 0.91 | 97.22 plus or minus 0.76 | +0.56 | 96.40 plus or minus 0.95 | -0.27 |
