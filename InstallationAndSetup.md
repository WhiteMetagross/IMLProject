# Installation and Setup:

Run Python research commands inside WSL with a project Conda environment. Do not install packages into the Conda base environment.

## Clone:

```bash
git clone https://github.com/WhiteMetagross/IMLProject.git
cd IMLProject
```

## Environment:

The validated development environment used Python 3.11. Create a project environment then install the recorded requirements.

```bash
conda create -n iml-research python=3.11 -y
conda run --no-capture-output -n iml-research python -m pip install -r requirements.txt
```

Record the resolved package versions before a full run.

```bash
conda run --no-capture-output -n iml-research python -m pip freeze > environment.lock.txt
```

## Dataset Layout:

Place legally obtained datasets below `datasets/` with these names.

```text
datasets/
  UCI HAR Dataset/
  motion-sense-master/
  WISDM_ar_v1.1/
  PAMAP2_Dataset/
  Opportunity/
  UniMiB-SHAR/
  Skoda/
  Daphnet/
```

Raw data is not stored in Git. The loaders preserve the preprocessing plus split logic used by the report experiments.

## Focused Validation:

Run tests from the repository root.

```bash
conda run --no-capture-output -n iml-research python -m pytest KnowledgeDistillation/tests classicalBaselines/tests -q
conda run --no-capture-output -n iml-research python -m compileall -q models baselines data nanoharmamba scripts KnowledgeDistillation classicalBaselines
```

Inspect the complete KD plan without training.

```bash
conda run --no-capture-output -n iml-research python KnowledgeDistillation/run_study.py plan --datasets all --routes direct,assistant
```

## Experiment Entry Points:

Train the neural candidate or baselines with explicit dataset plus seed settings.

```bash
conda run --no-capture-output -n iml-research python scripts/trainMicroBiConvLstm.py --dataset ucihar --seeds 5
conda run --no-capture-output -n iml-research python scripts/trainBaselines.py --dataset ucihar --model all --seeds 5
```

Run the classical study with five seeds generated from master seed 17.

```bash
conda run --no-capture-output -n iml-research python classicalBaselines/trainClassicalBaselines.py --models all --datasets all --seeds 5 --master-seed 17 --run-name classicalMaster17FiveSeeds
```

Run the full configured KD study only after source checkpoints and datasets are available.

```bash
conda run --no-capture-output -n iml-research python KnowledgeDistillation/run_study.py all --datasets all --routes direct,assistant --execute
```

All long runs write generated artifacts under ignored result directories.

## Report Build:

Use Windows MiKTeX from the `paper` directory.

```powershell
$paperRoot = Join-Path (Get-Location) 'paper'
$miktexBin = 'C:\Program Files\MiKTeX\miktex\bin\x64'
$env:Path = "$miktexBin;$env:Path"
Push-Location $paperRoot
& "$miktexBin\latexmk.exe" -pdf -interaction=nonstopmode -halt-on-error -file-line-error -outdir=build IMLReport.tex
Pop-Location
```

The expected output is `paper/build/IMLReport.pdf`.
