"""Train direct and sequential knowledge distillation models for HAR.

Direct route:
    teacher -> MicroBiConvLSTM

Sequential route:
    teacher -> assistant -> MicroBiConvLSTM

The script expects source checkpoints produced by the paper-aligned trainers.
It never changes the deployment architecture of MicroBiConvLSTM.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader


THIS_FILE = Path(__file__).resolve()
PACKAGE_ROOT = THIS_FILE.parent
PROJECT_ROOT = PACKAGE_ROOT.parent

sys.path.insert(0, str(PROJECT_ROOT))

from baselines import DeepConvLSTM, TinierHAR, TinyHAR
from models.microBiConvLstm import MicroBiConvLSTM
from scripts import trainBaselines as baselineTraining
from scripts import trainMicroBiConvLstm as microTraining


DEFAULT_CONFIG = PACKAGE_ROOT / "config.json"
ACTIVE_STUDY_CONFIG = DEFAULT_CONFIG
MODEL_NAMES = {"deepconvlstm", "tinyhar", "tinierhar", "microbi_convlstm"}


def loadStudyConfig(path: Path = DEFAULT_CONFIG) -> Dict:
    """Load and validate study configuration."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    requiredTopLevel = {
        "masterSeed",
        "seedCount",
        "temperature",
        "alpha",
        "epochs",
        "patience",
        "baselineTrainingOverrides",
        "datasets",
    }
    missing = sorted(requiredTopLevel.difference(payload))
    if missing:
        raise ValueError(f"Study config missing fields: {', '.join(missing)}")

    if not 0.0 <= float(payload["alpha"]) <= 1.0:
        raise ValueError("alpha must be between 0 and 1.")
    if float(payload["temperature"]) <= 0.0:
        raise ValueError("temperature must be positive.")

    for datasetName, mapping in payload["datasets"].items():
        for role in ("teacher", "assistant", "student"):
            if mapping.get(role) not in MODEL_NAMES:
                raise ValueError(
                    f"Unsupported {role} for {datasetName}: {mapping.get(role)}"
                )
        if mapping["student"] != "microbi_convlstm":
            raise ValueError(f"Student must be microbi_convlstm for {datasetName}.")
        for role in ("teacher", "assistant"):
            modelName = mapping[role]
            if datasetName not in payload["baselineTrainingOverrides"].get(modelName, {}):
                raise ValueError(
                    f"Missing baseline training override for {modelName} on {datasetName}."
                )
    return payload


def generateStudySeeds(masterSeed: int, count: int) -> List[int]:
    """Generate the canonical study seeds from the master seed."""
    rng = np.random.default_rng(masterSeed)
    return [int(rng.integers(0, 2**31 - 1)) for _ in range(count)]


def parseCsv(value: str, allowed: Iterable[str], label: str) -> List[str]:
    """Parse a comma separated CLI value."""
    allowedValues = list(allowed)
    if value == "all":
        return allowedValues
    parsed = [token.strip().lower() for token in value.split(",") if token.strip()]
    if not parsed:
        raise ValueError(f"{label} must contain at least one value.")
    invalid = [token for token in parsed if token not in allowedValues]
    if invalid:
        raise ValueError(f"Unsupported {label}: {', '.join(invalid)}")
    return parsed


def parseSeedList(value: Optional[str], config: Dict) -> List[int]:
    """Use explicit seeds or generate the configured five seeds."""
    studySeeds = generateStudySeeds(int(config["masterSeed"]), int(config["seedCount"]))
    if value:
        seeds = [int(token.strip()) for token in value.split(",") if token.strip()]
        if not seeds:
            raise ValueError("Seed list contains no valid integers.")
        invalid = [seed for seed in seeds if seed not in studySeeds]
        if invalid:
            raise ValueError(
                "KD seed list contains seeds outside the configured master seed 17 study set."
            )
        return seeds
    return studySeeds


def setSeed(seed: int) -> None:
    """Seed Python, NumPy, PyTorch and CUDA."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def getDevice(requested: str = "auto") -> torch.device:
    """Resolve requested execution device."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def datasetConfig(datasetName: str) -> Dict:
    """Return the paper-aligned dataset configuration."""
    return deepcopy(microTraining.DATASET_CONFIGS[datasetName])


def targetTrainingConfig(modelName: str, datasetName: str) -> Dict:
    """Return target optimizer and dropout settings used by existing trainers."""
    if modelName == "microbi_convlstm":
        config = datasetConfig(datasetName)
        config["hpoSource"] = None
        return config

    studyConfig = loadStudyConfig(ACTIVE_STUDY_CONFIG)
    config = deepcopy(baselineTraining.DATASET_CONFIGS[datasetName])
    overrides = deepcopy(
        studyConfig["baselineTrainingOverrides"][modelName][datasetName]
    )
    config.update(overrides)
    config["hpoSource"] = f"{ACTIVE_STUDY_CONFIG}#baselineTrainingOverrides"
    return config


def createModel(modelName: str, datasetName: str) -> nn.Module:
    """Create one study model using existing architecture definitions."""
    config = targetTrainingConfig(modelName, datasetName)
    common = {
        "numClasses": config["numClasses"],
        "inChannels": config["inputChannels"],
        "seqLen": config["seqLen"],
    }
    if modelName == "microbi_convlstm":
        return MicroBiConvLSTM(
            **common,
            dropout=config["dropout"],
            aggregation="last",
        )
    if modelName == "tinyhar":
        return TinyHAR(**common, dropout=config.get("dropout", 0.5))
    if modelName == "tinierhar":
        return TinierHAR(**common, dropout=config.get("dropout", 0.2))
    if modelName == "deepconvlstm":
        return DeepConvLSTM(**common, dropout=config.get("dropout", 0.5))
    raise ValueError(f"Unsupported model: {modelName}")


def parameterCount(model: nn.Module) -> int:
    """Count all model parameters."""
    return sum(parameter.numel() for parameter in model.parameters())


def freezeSource(model: nn.Module) -> None:
    """Freeze a source model and switch it to evaluation mode."""
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def knowledgeDistillationLoss(
    targetLogits: torch.Tensor,
    sourceLogits: torch.Tensor,
    labels: torch.Tensor,
    hardCriterion: nn.Module,
    temperature: float,
    alpha: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return total, hard label and softened output losses."""
    if targetLogits.shape != sourceLogits.shape:
        raise ValueError(
            f"Logit shape mismatch: target {tuple(targetLogits.shape)} versus "
            f"source {tuple(sourceLogits.shape)}."
        )
    hardLoss = hardCriterion(targetLogits, labels)
    targetLogProbabilities = F.log_softmax(targetLogits / temperature, dim=1)
    sourceProbabilities = F.softmax(sourceLogits / temperature, dim=1)
    softLoss = F.kl_div(
        targetLogProbabilities,
        sourceProbabilities,
        reduction="batchmean",
    ) * (temperature**2)
    totalLoss = (1.0 - alpha) * hardLoss + alpha * softLoss
    return totalLoss, hardLoss, softLoss


def sha256File(path: Path) -> str:
    """Compute checkpoint SHA256 without loading it into memory at once."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extractStateDict(payload) -> Dict[str, torch.Tensor]:
    """Extract a state dictionary from supported checkpoint formats."""
    if isinstance(payload, dict) and "model_state_dict" in payload:
        return payload["model_state_dict"]
    if isinstance(payload, dict) and payload and all(
        isinstance(value, torch.Tensor) for value in payload.values()
    ):
        return payload
    raise ValueError("Checkpoint does not contain a supported model state dictionary.")


def loadSourceModel(
    modelName: str,
    datasetName: str,
    checkpointPath: Path,
    device: torch.device,
) -> nn.Module:
    """Create, load and freeze a source model."""
    if not checkpointPath.exists():
        raise FileNotFoundError(f"Missing source checkpoint: {checkpointPath}")
    model = createModel(modelName, datasetName).to(device)
    payload = torch.load(checkpointPath, map_location=device, weights_only=False)
    model.load_state_dict(extractStateDict(payload), strict=True)
    freezeSource(model)
    return model


def baselineCheckpointPath(
    checkpointRoot: Path,
    modelName: str,
    datasetName: str,
    seed: int,
) -> Path:
    """Resolve a recovered paper-aligned checkpoint."""
    if modelName == "microbi_convlstm":
        return (
            checkpointRoot
            / "microbi_convlstm"
            / datasetName
            / f"microbi_convlstm_{datasetName}_seed{seed}_checkpoint.pt"
        )
    return checkpointRoot / "baselines" / modelName / datasetName / f"seed{seed}_checkpoint.pt"


def assistantCheckpointPath(runRoot: Path, datasetName: str, seed: int) -> Path:
    """Resolve the distilled assistant checkpoint used by stage two."""
    return runRoot / "assistant" / datasetName / f"seed{seed}" / "assistant_checkpoint.pt"


def loadDataset(datasetName: str, batchSize: int):
    """Load current paper-aligned train and held-out validation loaders."""
    previous = Path.cwd()
    try:
        os.chdir(PROJECT_ROOT)
        return baselineTraining.loadDataset(datasetName, batchSize)
    finally:
        os.chdir(previous)


def createCriterion(classWeights: Optional[torch.Tensor], device: torch.device) -> nn.Module:
    """Create the existing hard label criterion."""
    if classWeights is None:
        return nn.CrossEntropyLoss()
    return nn.CrossEntropyLoss(weight=classWeights.to(device))


def limitedBatches(loader: DataLoader, maximum: Optional[int]):
    """Iterate over a loader with an optional smoke-test batch limit."""
    for index, batch in enumerate(loader):
        if maximum is not None and index >= maximum:
            break
        yield batch


def evaluatedExampleCount(loader: DataLoader, maximum: Optional[int]) -> int:
    """Return evaluated sample count for full or limited loader iteration."""
    if maximum is None:
        return len(loader.dataset)
    total = 0
    for inputs, _ in limitedBatches(loader, maximum):
        total += int(inputs.size(0))
    return total


def trainDistillationEpoch(
    source: nn.Module,
    target: nn.Module,
    loader: DataLoader,
    hardCriterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    scaler: GradScaler,
    temperature: float,
    alpha: float,
    maxBatches: Optional[int] = None,
) -> Dict[str, float]:
    """Train target for one distillation epoch."""
    source.eval()
    target.train()
    useAmp = device.type == "cuda"
    totals = {"total": 0.0, "hard": 0.0, "soft": 0.0, "count": 0}
    predictions: List[int] = []
    labelsAll: List[int] = []

    for inputs, labels in limitedBatches(loader, maxBatches):
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            sourceLogits = source(inputs)

        if useAmp:
            with autocast(device_type="cuda"):
                targetLogits = target(inputs)
                totalLoss, hardLoss, softLoss = knowledgeDistillationLoss(
                    targetLogits,
                    sourceLogits,
                    labels,
                    hardCriterion,
                    temperature,
                    alpha,
                )
            scaler.scale(totalLoss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(target.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            targetLogits = target(inputs)
            totalLoss, hardLoss, softLoss = knowledgeDistillationLoss(
                targetLogits,
                sourceLogits,
                labels,
                hardCriterion,
                temperature,
                alpha,
            )
            totalLoss.backward()
            torch.nn.utils.clip_grad_norm_(target.parameters(), max_norm=1.0)
            optimizer.step()

        batchSize = int(inputs.size(0))
        totals["total"] += float(totalLoss.detach()) * batchSize
        totals["hard"] += float(hardLoss.detach()) * batchSize
        totals["soft"] += float(softLoss.detach()) * batchSize
        totals["count"] += batchSize
        predictions.extend(targetLogits.argmax(dim=1).detach().cpu().tolist())
        labelsAll.extend(labels.detach().cpu().tolist())

    if totals["count"] == 0:
        raise RuntimeError("Training loader produced no samples.")
    count = totals["count"]
    return {
        "totalLoss": totals["total"] / count,
        "hardLoss": totals["hard"] / count,
        "softLoss": totals["soft"] / count,
        "accuracy": accuracy_score(labelsAll, predictions),
        "macroF1": f1_score(labelsAll, predictions, average="macro", zero_division=0),
        "samples": count,
    }


@torch.no_grad()
def evaluateTarget(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    numClasses: int,
    maxBatches: Optional[int] = None,
) -> Dict:
    """Evaluate target with hard labels."""
    model.eval()
    totalLoss = 0.0
    sampleCount = 0
    predictions: List[int] = []
    labelsAll: List[int] = []

    for inputs, labels in limitedBatches(loader, maxBatches):
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(inputs)
        loss = criterion(logits, labels)
        batchSize = int(inputs.size(0))
        totalLoss += float(loss) * batchSize
        sampleCount += batchSize
        predictions.extend(logits.argmax(dim=1).cpu().tolist())
        labelsAll.extend(labels.cpu().tolist())

    if sampleCount == 0:
        raise RuntimeError("Validation loader produced no samples.")
    labelRange = list(range(numClasses))
    return {
        "loss": totalLoss / sampleCount,
        "accuracy": accuracy_score(labelsAll, predictions),
        "macroF1": f1_score(labelsAll, predictions, average="macro", zero_division=0),
        "perClassF1": f1_score(
            labelsAll,
            predictions,
            labels=labelRange,
            average=None,
            zero_division=0,
        ).tolist(),
        "confusionMatrix": confusion_matrix(
            labelsAll,
            predictions,
            labels=labelRange,
        ).tolist(),
        "predictions": predictions,
        "labels": labelsAll,
        "samples": sampleCount,
    }


def gitCommit() -> Optional[str]:
    """Return current Git commit when available."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def environmentMetadata(device: torch.device) -> Dict:
    """Capture runtime versions used by a run."""
    gpuName = None
    if torch.cuda.is_available():
        gpuName = torch.cuda.get_device_name(0)
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cudaRuntime": torch.version.cuda,
        "cudaAvailable": torch.cuda.is_available(),
        "gpu": gpuName,
        "numpy": np.__version__,
        "device": str(device),
        "gitCommit": gitCommit(),
    }


def saveJson(path: Path, payload: Dict) -> None:
    """Write JSON after creating its parent directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def reusableStageResult(
    sourceName: str,
    targetName: str,
    datasetName: str,
    seed: int,
    sourceCheckpoint: Path,
    outputDirectory: Path,
    outputCheckpointName: str,
    epochs: int,
    patience: int,
    temperature: float,
    alpha: float,
    maxTrainBatches: Optional[int],
    maxValidationBatches: Optional[int],
) -> Optional[Dict]:
    """Return an existing stage only when its full training identity matches."""
    checkpointPath = outputDirectory / outputCheckpointName
    configPath = outputDirectory / "run_config.json"
    resultPath = outputDirectory / "result.json"
    if not checkpointPath.exists() or not configPath.exists() or not resultPath.exists():
        return None

    existingConfig = json.loads(configPath.read_text(encoding="utf-8"))
    expected = {
        "dataset": datasetName,
        "seed": seed,
        "sourceModel": sourceName,
        "targetModel": targetName,
        "sourceCheckpointSha256": sha256File(sourceCheckpoint),
        "temperature": temperature,
        "alpha": alpha,
        "epochs": epochs,
        "patience": patience,
        "maxTrainBatches": maxTrainBatches,
        "maxValidationBatches": maxValidationBatches,
    }
    mismatches = {
        key: {"expected": value, "found": existingConfig.get(key)}
        for key, value in expected.items()
        if existingConfig.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            f"Existing stage at {outputDirectory} has a different configuration: "
            f"{json.dumps(mismatches, sort_keys=True)}. Use --overwrite to replace it."
        )
    print(f"Reusing complete stage: {outputDirectory}")
    return {
        "checkpoint": str(checkpointPath),
        "runConfig": existingConfig,
        "result": json.loads(resultPath.read_text(encoding="utf-8")),
        "reused": True,
    }


def trainStage(
    sourceName: str,
    targetName: str,
    datasetName: str,
    seed: int,
    sourceCheckpoint: Path,
    outputDirectory: Path,
    outputCheckpointName: str,
    epochs: int,
    patience: int,
    temperature: float,
    alpha: float,
    device: torch.device,
    maxTrainBatches: Optional[int] = None,
    maxValidationBatches: Optional[int] = None,
    overwrite: bool = False,
) -> Dict:
    """Train one frozen-source distillation stage."""
    if not sourceCheckpoint.exists():
        raise FileNotFoundError(f"Missing source checkpoint: {sourceCheckpoint}")
    if not overwrite:
        reusable = reusableStageResult(
            sourceName=sourceName,
            targetName=targetName,
            datasetName=datasetName,
            seed=seed,
            sourceCheckpoint=sourceCheckpoint,
            outputDirectory=outputDirectory,
            outputCheckpointName=outputCheckpointName,
            epochs=epochs,
            patience=patience,
            temperature=temperature,
            alpha=alpha,
            maxTrainBatches=maxTrainBatches,
            maxValidationBatches=maxValidationBatches,
        )
        if reusable is not None:
            return reusable

    setSeed(seed)
    targetConfig = targetTrainingConfig(targetName, datasetName)
    trainLoader, validationLoader, classWeights = loadDataset(
        datasetName,
        int(targetConfig["batchSize"]),
    )
    criterion = createCriterion(classWeights, device)
    source = loadSourceModel(sourceName, datasetName, sourceCheckpoint, device)
    # Source construction consumes RNG. Reset so target initialization matches
    # the paper-aligned target trainer for the same seed.
    setSeed(seed)
    target = createModel(targetName, datasetName).to(device)

    optimizer = optim.AdamW(
        target.parameters(),
        lr=float(targetConfig["lr"]),
        weight_decay=float(targetConfig["weightDecay"]),
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=1e-6,
    )
    scaler = GradScaler(device.type, enabled=device.type == "cuda")

    history = {
        "trainTotalLoss": [],
        "trainHardLoss": [],
        "trainSoftLoss": [],
        "trainMacroF1": [],
        "validationLoss": [],
        "validationMacroF1": [],
    }
    bestF1 = float("-inf")
    bestEpoch = 0
    bestState = None
    epochsWithoutImprovement = 0
    startTime = time.time()

    for epoch in range(1, epochs + 1):
        trainMetrics = trainDistillationEpoch(
            source=source,
            target=target,
            loader=trainLoader,
            hardCriterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            temperature=temperature,
            alpha=alpha,
            maxBatches=maxTrainBatches,
        )
        validationMetrics = evaluateTarget(
            model=target,
            loader=validationLoader,
            criterion=criterion,
            device=device,
            numClasses=int(targetConfig["numClasses"]),
            maxBatches=maxValidationBatches,
        )
        scheduler.step()

        history["trainTotalLoss"].append(trainMetrics["totalLoss"])
        history["trainHardLoss"].append(trainMetrics["hardLoss"])
        history["trainSoftLoss"].append(trainMetrics["softLoss"])
        history["trainMacroF1"].append(trainMetrics["macroF1"])
        history["validationLoss"].append(validationMetrics["loss"])
        history["validationMacroF1"].append(validationMetrics["macroF1"])

        if validationMetrics["macroF1"] > bestF1:
            bestF1 = float(validationMetrics["macroF1"])
            bestEpoch = epoch
            bestState = {
                key: value.detach().cpu().clone()
                for key, value in target.state_dict().items()
            }
            epochsWithoutImprovement = 0
        else:
            epochsWithoutImprovement += 1

        print(
            f"{datasetName} seed={seed} {sourceName}->{targetName} "
            f"epoch={epoch}/{epochs} trainF1={trainMetrics['macroF1']:.4f} "
            f"validationF1={validationMetrics['macroF1']:.4f} "
            f"best={bestF1:.4f}"
        )
        if epochsWithoutImprovement >= patience:
            break

    if bestState is None:
        raise RuntimeError("No best target state was recorded.")

    target.load_state_dict(bestState)
    finalMetrics = evaluateTarget(
        model=target,
        loader=validationLoader,
        criterion=criterion,
        device=device,
        numClasses=int(targetConfig["numClasses"]),
        maxBatches=maxValidationBatches,
    )
    outputDirectory.mkdir(parents=True, exist_ok=True)
    checkpointPath = outputDirectory / outputCheckpointName
    sourceHash = sha256File(sourceCheckpoint)
    elapsed = time.time() - startTime
    runConfig = {
        "dataset": datasetName,
        "seed": seed,
        "sourceModel": sourceName,
        "targetModel": targetName,
        "sourceCheckpoint": str(sourceCheckpoint.resolve()),
        "sourceCheckpointSha256": sourceHash,
        "temperature": temperature,
        "alpha": alpha,
        "epochs": epochs,
        "patience": patience,
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR",
        "selectionMetric": "macro_f1",
        "validationSource": "existing held-out evaluation loader used by the paper-aligned trainers",
        "targetTrainingConfig": targetConfig,
        "maxTrainBatches": maxTrainBatches,
        "maxValidationBatches": maxValidationBatches,
        "sourceParameters": parameterCount(source),
        "targetParameters": parameterCount(target),
        "environment": environmentMetadata(device),
    }
    result = {
        "bestEpoch": bestEpoch,
        "validationMacroF1": finalMetrics["macroF1"],
        "validationAccuracy": finalMetrics["accuracy"],
        "validationLoss": finalMetrics["loss"],
        "perClassF1": finalMetrics["perClassF1"],
        "confusionMatrix": finalMetrics["confusionMatrix"],
        "predictions": finalMetrics["predictions"],
        "labels": finalMetrics["labels"],
        "trainingSeconds": elapsed,
        "history": history,
    }
    torch.save(
        {
            "model_state_dict": bestState,
            "run_config": runConfig,
            "result": result,
        },
        checkpointPath,
    )
    torch.save(bestState, outputDirectory / "model_state.pt")
    saveJson(outputDirectory / "run_config.json", runConfig)
    saveJson(outputDirectory / "result.json", result)
    return {
        "checkpoint": str(checkpointPath),
        "runConfig": runConfig,
        "result": result,
    }


def runDirectRoute(
    mapping: Dict,
    datasetName: str,
    seed: int,
    checkpointRoot: Path,
    runRoot: Path,
    **trainingArguments,
) -> Dict:
    """Run teacher to MicroBiConvLSTM distillation."""
    outputDirectory = runRoot / "direct" / datasetName / f"seed{seed}"
    return trainStage(
        sourceName=mapping["teacher"],
        targetName=mapping["student"],
        datasetName=datasetName,
        seed=seed,
        sourceCheckpoint=baselineCheckpointPath(
            checkpointRoot,
            mapping["teacher"],
            datasetName,
            seed,
        ),
        outputDirectory=outputDirectory,
        outputCheckpointName="student_checkpoint.pt",
        **trainingArguments,
    )


def runAssistantRoute(
    mapping: Dict,
    datasetName: str,
    seed: int,
    checkpointRoot: Path,
    runRoot: Path,
    **trainingArguments,
) -> Dict:
    """Run teacher to assistant, then assistant to MicroBiConvLSTM."""
    outputDirectory = runRoot / "assistant" / datasetName / f"seed{seed}"
    assistantResult = trainStage(
        sourceName=mapping["teacher"],
        targetName=mapping["assistant"],
        datasetName=datasetName,
        seed=seed,
        sourceCheckpoint=baselineCheckpointPath(
            checkpointRoot,
            mapping["teacher"],
            datasetName,
            seed,
        ),
        outputDirectory=outputDirectory,
        outputCheckpointName="assistant_checkpoint.pt",
        **trainingArguments,
    )
    studentResult = trainStage(
        sourceName=mapping["assistant"],
        targetName=mapping["student"],
        datasetName=datasetName,
        seed=seed,
        sourceCheckpoint=assistantCheckpointPath(runRoot, datasetName, seed),
        outputDirectory=outputDirectory / "student",
        outputCheckpointName="student_checkpoint.pt",
        **trainingArguments,
    )
    return {"assistant": assistantResult, "student": studentResult}


def syntheticSmoke(config: Dict, device: torch.device) -> Dict:
    """Run real model forwards and KD backward passes on synthetic windows."""
    results = {}
    for datasetName, mapping in config["datasets"].items():
        setSeed(int(config["masterSeed"]))
        dataConfig = datasetConfig(datasetName)
        inputs = torch.randn(
            2,
            int(dataConfig["seqLen"]),
            int(dataConfig["inputChannels"]),
            device=device,
        )
        labels = torch.arange(2, device=device) % int(dataConfig["numClasses"])
        teacher = createModel(mapping["teacher"], datasetName).to(device)
        assistant = createModel(mapping["assistant"], datasetName).to(device)
        student = createModel(mapping["student"], datasetName).to(device)
        freezeSource(teacher)
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            teacherLogits = teacher(inputs)
        assistantLogits = assistant(inputs)
        assistantLoss, _, _ = knowledgeDistillationLoss(
            assistantLogits,
            teacherLogits,
            labels,
            criterion,
            float(config["temperature"]),
            float(config["alpha"]),
        )
        assistantLoss.backward()
        if any(parameter.grad is not None for parameter in teacher.parameters()):
            raise AssertionError("Frozen teacher received gradients.")

        freezeSource(assistant)
        assistant.zero_grad(set_to_none=True)
        with torch.no_grad():
            assistantLogits = assistant(inputs)
        studentLogits = student(inputs)
        studentLoss, _, _ = knowledgeDistillationLoss(
            studentLogits,
            assistantLogits,
            labels,
            criterion,
            float(config["temperature"]),
            float(config["alpha"]),
        )
        studentLoss.backward()
        if any(parameter.grad is not None for parameter in assistant.parameters()):
            raise AssertionError("Frozen assistant received gradients.")

        results[datasetName] = {
            "teacher": mapping["teacher"],
            "assistant": mapping["assistant"],
            "student": mapping["student"],
            "logitShape": list(studentLogits.shape),
            "assistantLossFinite": bool(torch.isfinite(assistantLoss).item()),
            "studentLossFinite": bool(torch.isfinite(studentLoss).item()),
        }
    return results


def metricSummary(values: Sequence[float]) -> Dict[str, float]:
    """Summarize one metric across study seeds."""
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def summarizeDatasetResults(datasetResults: Dict[str, Dict]) -> Dict:
    """Create route summaries without discarding individual seed outcomes."""
    summary: Dict[str, Dict] = {}
    directF1 = [
        float(seedResult["direct"]["result"]["validationMacroF1"])
        for seedResult in datasetResults.values()
        if "direct" in seedResult
    ]
    if directF1:
        summary["directStudentValidationMacroF1"] = metricSummary(directF1)

    assistantF1 = [
        float(seedResult["assistant"]["assistant"]["result"]["validationMacroF1"])
        for seedResult in datasetResults.values()
        if "assistant" in seedResult
    ]
    assistantStudentF1 = [
        float(seedResult["assistant"]["student"]["result"]["validationMacroF1"])
        for seedResult in datasetResults.values()
        if "assistant" in seedResult
    ]
    if assistantF1:
        summary["distilledAssistantValidationMacroF1"] = metricSummary(assistantF1)
    if assistantStudentF1:
        summary["assistantStudentValidationMacroF1"] = metricSummary(
            assistantStudentF1
        )
    return summary


def main() -> int:
    global ACTIVE_STUDY_CONFIG
    parser = argparse.ArgumentParser(description="Train direct and teacher assistant HAR knowledge distillation routes.")
    parser.add_argument("--study-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--datasets", default="pamap2,wisdm")
    parser.add_argument("--routes", default="direct,assistant")
    parser.add_argument("--seed-list", default=None)
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=PACKAGE_ROOT / "results" / "checkpoints",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=PACKAGE_ROOT / "results" / "runs",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-validation-batches", type=int, default=None)
    parser.add_argument("--synthetic-smoke", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing stage instead of safely resuming it.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Required before any dataset training starts.",
    )
    args = parser.parse_args()
    ACTIVE_STUDY_CONFIG = args.study_config.resolve()

    config = loadStudyConfig(args.study_config)
    device = getDevice(args.device)
    if args.synthetic_smoke:
        results = syntheticSmoke(config, device)
        print(json.dumps(results, indent=2))
        return 0

    datasetNames = parseCsv(args.datasets, config["datasets"].keys(), "datasets")
    routes = parseCsv(args.routes, ["direct", "assistant"], "routes")
    seeds = parseSeedList(args.seed_list, config)
    epochs = args.epochs if args.epochs is not None else int(config["epochs"])
    patience = args.patience if args.patience is not None else int(config["patience"])
    temperature = (
        args.temperature if args.temperature is not None else float(config["temperature"])
    )
    alpha = args.alpha if args.alpha is not None else float(config["alpha"])

    plan = {
        "datasets": datasetNames,
        "routes": routes,
        "seedCount": len(seeds),
        "epochs": epochs,
        "patience": patience,
        "temperature": temperature,
        "alpha": alpha,
        "checkpointRoot": str(args.checkpoint_root.resolve()),
        "runRoot": str(args.run_root.resolve()),
        "datasetRuns": len(datasetNames) * len(routes) * len(seeds),
        "trainingStages": len(datasetNames)
        * len(seeds)
        * sum(1 if route == "direct" else 2 for route in routes),
    }
    print(json.dumps(plan, indent=2))
    if not args.execute:
        print("Dry plan only. Add --execute to start dataset training.")
        return 0

    trainingArguments = {
        "epochs": epochs,
        "patience": patience,
        "temperature": temperature,
        "alpha": alpha,
        "device": device,
        "maxTrainBatches": args.max_train_batches,
        "maxValidationBatches": args.max_validation_batches,
        "overwrite": args.overwrite,
    }
    allResults = {}
    for datasetName in datasetNames:
        mapping = config["datasets"][datasetName]
        allResults[datasetName] = {}
        for seed in seeds:
            allResults[datasetName][str(seed)] = {}
            if "direct" in routes:
                allResults[datasetName][str(seed)]["direct"] = runDirectRoute(
                    mapping=mapping,
                    datasetName=datasetName,
                    seed=seed,
                    checkpointRoot=args.checkpoint_root,
                    runRoot=args.run_root,
                    **trainingArguments,
                )
            if "assistant" in routes:
                allResults[datasetName][str(seed)]["assistant"] = runAssistantRoute(
                    mapping=mapping,
                    datasetName=datasetName,
                    seed=seed,
                    checkpointRoot=args.checkpoint_root,
                    runRoot=args.run_root,
                    **trainingArguments,
                )

    for datasetName, datasetResults in allResults.items():
        saveJson(
            args.run_root / "summaries" / f"{datasetName}_results.json",
            datasetResults,
        )
        saveJson(
            args.run_root / "summaries" / f"{datasetName}_summary.json",
            summarizeDatasetResults(datasetResults),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
