"""Prepare and launch the configured knowledge distillation study.

Default behavior is a dry plan. Dataset training starts only with --execute.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


THIS_FILE = Path(__file__).resolve()
PACKAGE_ROOT = THIS_FILE.parent
PROJECT_ROOT = PACKAGE_ROOT.parent

sys.path.insert(0, str(PROJECT_ROOT))

from KnowledgeDistillation.train import (
    DEFAULT_CONFIG,
    baselineCheckpointPath,
    loadStudyConfig,
    parseCsv,
    parseSeedList,
)


def recoveryModels(mapping: Dict) -> List[str]:
    """Return unique teacher, assistant and student models in role order."""
    models = []
    for role in ("teacher", "assistant", "student"):
        modelName = mapping[role]
        if modelName not in models:
            models.append(modelName)
    return models


def buildRecoveryCommand(
    pythonExecutable: str,
    studyConfig: Path,
    checkpointRoot: Path,
    modelName: str,
    datasetName: str,
    seeds: Sequence[int],
    epochs: int,
    patience: int,
) -> List[str]:
    """Build one checkpoint recovery command with recorded training settings."""
    seedList = ",".join(str(seed) for seed in seeds)
    return [
        pythonExecutable,
        str(PACKAGE_ROOT / "recover.py"),
        "--study-config",
        str(studyConfig),
        "--model",
        modelName,
        "--dataset",
        datasetName,
        "--seed-list",
        seedList,
        "--epochs",
        str(epochs),
        "--patience",
        str(patience),
        "--checkpoint-root",
        str(checkpointRoot),
        "--execute",
    ]


def buildDistillationCommand(
    pythonExecutable: str,
    studyConfig: Path,
    checkpointRoot: Path,
    runRoot: Path,
    datasetName: str,
    routes: Sequence[str],
    seeds: Sequence[int],
    epochs: int,
    patience: int,
    temperature: float,
    alpha: float,
) -> List[str]:
    """Build one dataset command containing requested KD routes."""
    return [
        pythonExecutable,
        str(PACKAGE_ROOT / "train.py"),
        "--study-config",
        str(studyConfig),
        "--datasets",
        datasetName,
        "--routes",
        ",".join(routes),
        "--seed-list",
        ",".join(str(seed) for seed in seeds),
        "--checkpoint-root",
        str(checkpointRoot),
        "--run-root",
        str(runRoot),
        "--epochs",
        str(epochs),
        "--patience",
        str(patience),
        "--temperature",
        str(temperature),
        "--alpha",
        str(alpha),
        "--execute",
    ]


def expectedRecoveryCheckpoints(
    checkpointRoot: Path,
    config: Dict,
    datasetNames: Sequence[str],
    seeds: Sequence[int],
) -> List[Path]:
    """Return every checkpoint required by the selected recovery plan."""
    expected = []
    for datasetName in datasetNames:
        for modelName in recoveryModels(config["datasets"][datasetName]):
            for seed in seeds:
                expected.append(
                    baselineCheckpointPath(checkpointRoot, modelName, datasetName, seed)
                )
    return expected


def missingRecoveryCheckpoints(
    checkpointRoot: Path,
    config: Dict,
    datasetNames: Sequence[str],
    seeds: Sequence[int],
) -> List[Path]:
    """List missing recovery checkpoints."""
    return [
        path
        for path in expectedRecoveryCheckpoints(
            checkpointRoot,
            config,
            datasetNames,
            seeds,
        )
        if not path.exists()
    ]


def executeCommands(commands: Sequence[Sequence[str]], stopOnError: bool = True) -> List[Dict]:
    """Execute commands sequentially from the repository root."""
    statuses = []
    for index, command in enumerate(commands, start=1):
        display = shlex.join(command)
        print(f"[{index}/{len(commands)}] {display}")
        completed = subprocess.run(command, cwd=PROJECT_ROOT)
        status = {
            "index": index,
            "command": display,
            "returnCode": completed.returncode,
        }
        statuses.append(status)
        if completed.returncode != 0 and stopOnError:
            break
    return statuses


def savePlan(planDirectory: Path, manifest: Dict, commands: Sequence[Sequence[str]]) -> None:
    """Save exact commands and study manifest."""
    planDirectory.mkdir(parents=True, exist_ok=True)
    commandLines = [shlex.join(command) for command in commands]
    (planDirectory / "commands.sh").write_text(
        "\n".join(commandLines) + "\n",
        encoding="utf-8",
    )
    manifest["commands"] = commandLines
    (planDirectory / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare the full HAR KD study.")
    parser.add_argument(
        "action",
        choices=["plan", "recover", "distill", "all"],
        default="plan",
        nargs="?",
    )
    parser.add_argument("--study-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--datasets", default="pamap2,wisdm")
    parser.add_argument("--routes", default="direct,assistant")
    parser.add_argument("--seed-list", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PACKAGE_ROOT / "results",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Required before recovery or KD training starts.",
    )
    args = parser.parse_args()

    config = loadStudyConfig(args.study_config)
    datasetNames = parseCsv(args.datasets, config["datasets"].keys(), "datasets")
    routes = parseCsv(args.routes, ["direct", "assistant"], "routes")
    seeds = parseSeedList(args.seed_list, config)

    epochs = args.epochs if args.epochs is not None else int(config["epochs"])
    patience = args.patience if args.patience is not None else int(config["patience"])
    temperature = (
        args.temperature if args.temperature is not None else float(config["temperature"])
    )
    alpha = args.alpha if args.alpha is not None else float(config["alpha"])
    outputRoot = args.output_root.resolve()
    checkpointRoot = outputRoot / "checkpoints"
    runRoot = outputRoot / "runs"

    recoveryCommands = []
    for datasetName in datasetNames:
        mapping = config["datasets"][datasetName]
        for modelName in recoveryModels(mapping):
            missingSeeds = [
                seed
                for seed in seeds
                if not baselineCheckpointPath(
                    checkpointRoot,
                    modelName,
                    datasetName,
                    seed,
                ).exists()
            ]
            if not missingSeeds:
                continue
            recoveryCommands.append(
                buildRecoveryCommand(
                    pythonExecutable=sys.executable,
                    studyConfig=args.study_config.resolve(),
                    checkpointRoot=checkpointRoot,
                    modelName=modelName,
                    datasetName=datasetName,
                    seeds=missingSeeds,
                    epochs=epochs,
                    patience=patience,
                )
            )

    distillationCommands = [
        buildDistillationCommand(
            pythonExecutable=sys.executable,
            studyConfig=args.study_config.resolve(),
            checkpointRoot=checkpointRoot,
            runRoot=runRoot,
            datasetName=datasetName,
            routes=routes,
            seeds=seeds,
            epochs=epochs,
            patience=patience,
            temperature=temperature,
            alpha=alpha,
        )
        for datasetName in datasetNames
    ]

    selectedCommands = []
    if args.action in {"plan", "all", "recover"}:
        selectedCommands.extend(recoveryCommands)
    if args.action in {"plan", "all", "distill"}:
        selectedCommands.extend(distillationCommands)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    planDirectory = outputRoot / "plans" / f"study_{timestamp}"
    expectedCount = len(expectedRecoveryCheckpoints(checkpointRoot, config, datasetNames, seeds))
    manifest = {
        "action": args.action,
        "createdUtc": timestamp,
        "projectRoot": str(PROJECT_ROOT),
        "packageRoot": str(PACKAGE_ROOT),
        "pythonExecutable": sys.executable,
        "studyConfig": str(args.study_config.resolve()),
        "datasets": datasetNames,
        "routes": routes,
        "masterSeed": int(config["masterSeed"]),
        "seeds": seeds,
        "epochs": epochs,
        "patience": patience,
        "temperature": temperature,
        "alpha": alpha,
        "checkpointRoot": str(checkpointRoot),
        "runRoot": str(runRoot),
        "recoveryCommandCount": len(recoveryCommands),
        "expectedRecoveryCheckpointCount": expectedCount,
        "directKdRuns": len(datasetNames) * len(seeds) if "direct" in routes else 0,
        "assistantKdRuns": len(datasetNames) * len(seeds) if "assistant" in routes else 0,
        "assistantKdTrainingStages": (
            2 * len(datasetNames) * len(seeds) if "assistant" in routes else 0
        ),
    }
    savePlan(planDirectory, manifest, selectedCommands)

    print(json.dumps(manifest, indent=2))
    print(f"Plan saved: {planDirectory}")
    if args.action == "plan" or not args.execute:
        print("Dry plan only. No training started.")
        return 0

    statuses = []
    if args.action in {"recover", "all"}:
        statuses.extend(executeCommands(recoveryCommands))
        if any(status["returnCode"] != 0 for status in statuses):
            (planDirectory / "status.json").write_text(
                json.dumps(statuses, indent=2),
                encoding="utf-8",
            )
            return 1
        missing = missingRecoveryCheckpoints(
            checkpointRoot,
            config,
            datasetNames,
            seeds,
        )
        if missing:
            print("Recovery commands ended without all expected checkpoints:")
            for path in missing:
                print(path)
            (planDirectory / "status.json").write_text(
                json.dumps(statuses, indent=2),
                encoding="utf-8",
            )
            return 2

    if args.action in {"distill", "all"}:
        missing = missingRecoveryCheckpoints(
            checkpointRoot,
            config,
            datasetNames,
            seeds,
        )
        if missing:
            print("KD blocked. Required recovered checkpoints are missing:")
            for path in missing:
                print(path)
            return 3
        statuses.extend(executeCommands(distillationCommands))

    (planDirectory / "status.json").write_text(
        json.dumps(statuses, indent=2),
        encoding="utf-8",
    )
    return 0 if all(status["returnCode"] == 0 for status in statuses) else 1


if __name__ == "__main__":
    raise SystemExit(main())
