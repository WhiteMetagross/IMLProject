"""Recover teacher, assistant and normal student checkpoints for the KD study."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List


THIS_FILE = Path(__file__).resolve()
PACKAGE_ROOT = THIS_FILE.parent
PROJECT_ROOT = PACKAGE_ROOT.parent

sys.path.insert(0, str(PROJECT_ROOT))

from scripts import trainBaselines as baselineTraining
from scripts import trainMicroBiConvLstm as microTraining

from KnowledgeDistillation.train import (
    DEFAULT_CONFIG,
    MODEL_NAMES,
    loadStudyConfig,
    parseSeedList,
)


def baselineOverride(config: Dict, modelName: str, datasetName: str) -> Dict:
    """Return the exact HPO derived baseline settings used by the study."""
    try:
        return dict(config["baselineTrainingOverrides"][modelName][datasetName])
    except KeyError as error:
        raise ValueError(
            f"No baseline training override for {modelName} on {datasetName}."
        ) from error


def saveJson(path: Path, payload: Dict) -> None:
    """Save one JSON artifact with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def recoverMicroBiConvLstm(
    datasetName: str,
    seeds: List[int],
    epochs: int,
    patience: int,
    checkpointRoot: Path,
) -> Dict:
    """Train the normal MicroBiConvLSTM student for explicit study seeds."""
    outputDirectory = checkpointRoot / "microbi_convlstm" / datasetName
    result = microTraining.trainAllSeeds(
        datasetName=datasetName,
        nSeeds=len(seeds),
        explicitSeeds=seeds,
        epochs=epochs,
        patience=patience,
        saveDir=str(outputDirectory),
        verbose=True,
    )
    summary = {
        key: value for key, value in result.items() if key != "allResults"
    }
    saveJson(outputDirectory / "training_summary.json", summary)
    return summary


def recoverBaseline(
    config: Dict,
    configPath: Path,
    modelName: str,
    datasetName: str,
    seeds: List[int],
    epochs: int,
    patience: int,
    checkpointRoot: Path,
) -> Dict:
    """Train one baseline with recorded HPO derived settings."""
    override = baselineOverride(config, modelName, datasetName)
    originalLoader = baselineTraining.loadBestTrainingParams

    def exactTrainingParams(requestedModel: str, requestedDataset: str):
        if requestedModel != modelName or requestedDataset != datasetName:
            raise ValueError(
                "Recovery requested training settings for an unexpected model or dataset."
            )
        return dict(override), f"{configPath}#baselineTrainingOverrides"

    baselineTraining.loadBestTrainingParams = exactTrainingParams
    try:
        return baselineTraining.runMultiSeed(
            modelName=modelName,
            datasetName=datasetName,
            numSeeds=len(seeds),
            explicitSeeds=seeds,
            epochs=epochs,
            patience=patience,
            saveDir=str(checkpointRoot / "baselines"),
        )
    finally:
        baselineTraining.loadBestTrainingParams = originalLoader


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Recover source checkpoints for the knowledge distillation study."
    )
    parser.add_argument("--study-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model", required=True, choices=sorted(MODEL_NAMES))
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed-list", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=PACKAGE_ROOT / "results" / "checkpoints",
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    configPath = args.study_config.resolve()
    config = loadStudyConfig(configPath)
    if args.dataset not in config["datasets"]:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    mapping = config["datasets"][args.dataset]
    if args.model not in mapping.values():
        raise ValueError(
            f"{args.model} is not a study model for {args.dataset}."
        )

    seeds = parseSeedList(args.seed_list, config)
    epochs = args.epochs if args.epochs is not None else int(config["epochs"])
    patience = (
        args.patience if args.patience is not None else int(config["patience"])
    )
    checkpointRoot = args.checkpoint_root.resolve()

    manifest = {
        "model": args.model,
        "dataset": args.dataset,
        "masterSeed": int(config["masterSeed"]),
        "seeds": seeds,
        "epochs": epochs,
        "patience": patience,
        "studyConfig": str(configPath),
        "checkpointRoot": str(checkpointRoot),
    }
    print(json.dumps(manifest, indent=2))
    if not args.execute:
        print("Dry recovery plan only. No training started.")
        return 0

    previousDirectory = Path.cwd()
    try:
        os.chdir(PROJECT_ROOT)
        if args.model == "microbi_convlstm":
            result = recoverMicroBiConvLstm(
                datasetName=args.dataset,
                seeds=seeds,
                epochs=epochs,
                patience=patience,
                checkpointRoot=checkpointRoot,
            )
            outputDirectory = checkpointRoot / "microbi_convlstm" / args.dataset
        else:
            result = recoverBaseline(
                config=config,
                configPath=configPath,
                modelName=args.model,
                datasetName=args.dataset,
                seeds=seeds,
                epochs=epochs,
                patience=patience,
                checkpointRoot=checkpointRoot,
            )
            outputDirectory = checkpointRoot / "baselines" / args.model / args.dataset
    finally:
        os.chdir(previousDirectory)

    manifest["resultSummary"] = {
        key: value
        for key, value in result.items()
        if key not in {"allResults", "individual_results"}
    }
    saveJson(outputDirectory / "recovery_manifest.json", manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
