"""Focused tests for knowledge distillation training support."""

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from KnowledgeDistillation.run_study import (
    buildDistillationCommand,
    buildRecoveryCommand,
    expectedRecoveryCheckpoints,
    recoveryModels,
)
from KnowledgeDistillation.recover import baselineOverride
from KnowledgeDistillation.train import (
    DEFAULT_CONFIG,
    baselineCheckpointPath,
    createModel,
    freezeSource,
    generateStudySeeds,
    knowledgeDistillationLoss,
    loadStudyConfig,
    parseSeedList,
    summarizeDatasetResults,
    targetTrainingConfig,
)


def testStudyConfigContainsRequestedMappings():
    config = loadStudyConfig(DEFAULT_CONFIG)
    assert config["datasets"]["pamap2"]["teacher"] == "tinierhar"
    assert config["datasets"]["pamap2"]["assistant"] == "tinyhar"
    assert config["datasets"]["wisdm"]["teacher"] == "tinierhar"
    assert config["datasets"]["wisdm"]["assistant"] == "deepconvlstm"
    assert config["datasets"]["unimib"]["teacher"] == "deepconvlstm"
    assert config["datasets"]["ucihar"]["teacher"] == "tinyhar"
    assert config["datasets"]["skoda"]["teacher"] == "tinyhar"


def testRecordedBaselineOverridesMatchCompletedStudy():
    config = loadStudyConfig(DEFAULT_CONFIG)
    pamap2 = baselineOverride(config, "tinierhar", "pamap2")
    wisdm = baselineOverride(config, "deepconvlstm", "wisdm")
    assert pamap2["lr"] == pytest.approx(0.0017304738170769488)
    assert pamap2["dropout"] == pytest.approx(0.35886703210228926)
    assert wisdm["lr"] == pytest.approx(0.0011525599308775199)
    assert wisdm["dropout"] == pytest.approx(0.4740895086687639)
    assert targetTrainingConfig("tinierhar", "pamap2")["lr"] == pytest.approx(
        pamap2["lr"]
    )


def testStudySeedsAreDeterministicAndUnique():
    first = generateStudySeeds(17, 5)
    second = generateStudySeeds(17, 5)
    assert first == second
    assert len(first) == 5
    assert len(set(first)) == 5


def testNonStudySeedIsRejected():
    config = loadStudyConfig(DEFAULT_CONFIG)
    with pytest.raises(ValueError, match="outside the configured"):
        parseSeedList("29", config)


def testKnowledgeDistillationLossBackpropagatesOnlyToTarget():
    source = nn.Linear(4, 3)
    target = nn.Linear(4, 3)
    freezeSource(source)
    inputs = torch.randn(5, 4)
    labels = torch.tensor([0, 1, 2, 0, 1])
    with torch.no_grad():
        sourceLogits = source(inputs)
    targetLogits = target(inputs)
    totalLoss, hardLoss, softLoss = knowledgeDistillationLoss(
        targetLogits,
        sourceLogits,
        labels,
        nn.CrossEntropyLoss(),
        temperature=4.0,
        alpha=0.5,
    )
    totalLoss.backward()
    assert torch.isfinite(totalLoss)
    assert torch.isfinite(hardLoss)
    assert torch.isfinite(softLoss)
    assert all(parameter.grad is None for parameter in source.parameters())
    assert all(parameter.grad is not None for parameter in target.parameters())


@pytest.mark.parametrize("datasetName", ["pamap2", "wisdm", "unimib", "ucihar", "skoda"])
def testMappedModelsReturnCompatibleLogits(datasetName):
    config = loadStudyConfig(DEFAULT_CONFIG)
    mapping = config["datasets"][datasetName]
    teacher = createModel(mapping["teacher"], datasetName)
    assistant = createModel(mapping["assistant"], datasetName)
    student = createModel(mapping["student"], datasetName)
    dataConfig = {
        "pamap2": (128, 19, 12),
        "wisdm": (128, 3, 6),
        "unimib": (128, 3, 9),
        "ucihar": (128, 9, 6),
        "skoda": (98, 30, 11),
    }[datasetName]
    sequenceLength, channels, classes = dataConfig
    inputs = torch.randn(2, sequenceLength, channels)
    with torch.no_grad():
        teacherShape = teacher(inputs).shape
        assistantShape = assistant(inputs).shape
        studentShape = student(inputs).shape
    assert teacherShape == (2, classes)
    assert assistantShape == teacherShape
    assert studentShape == teacherShape


def testRecoveryPlanContainsThreeRolesAndFiveSeeds(tmp_path):
    config = loadStudyConfig(DEFAULT_CONFIG)
    seeds = generateStudySeeds(17, 5)
    expected = expectedRecoveryCheckpoints(
        tmp_path,
        config,
        ["pamap2", "wisdm"],
        seeds,
    )
    assert len(recoveryModels(config["datasets"]["pamap2"])) == 3
    assert len(recoveryModels(config["datasets"]["wisdm"])) == 3
    assert len(expected) == 30
    assert baselineCheckpointPath(
        tmp_path,
        "microbi_convlstm",
        "pamap2",
        seeds[0],
    ) in expected


def testRecoveryCommandUsesRecordedOverrideEntryPoint(tmp_path):
    command = buildRecoveryCommand(
        pythonExecutable="python",
        studyConfig=DEFAULT_CONFIG,
        checkpointRoot=tmp_path / "checkpoints",
        modelName="tinierhar",
        datasetName="pamap2",
        seeds=generateStudySeeds(17, 5),
        epochs=200,
        patience=10,
    )
    assert command[1].endswith("KnowledgeDistillation/recover.py")
    assert "--study-config" in command
    assert "--checkpoint-root" in command
    assert "--execute" in command


def testDistillationCommandRequiresExecuteFlag(tmp_path):
    command = buildDistillationCommand(
        pythonExecutable="python",
        studyConfig=DEFAULT_CONFIG,
        checkpointRoot=tmp_path / "checkpoints",
        runRoot=tmp_path / "runs",
        datasetName="pamap2",
        routes=["direct", "assistant"],
        seeds=generateStudySeeds(17, 5),
        epochs=200,
        patience=10,
        temperature=4.0,
        alpha=0.5,
    )
    assert "--execute" in command
    assert "direct,assistant" in command


def testDatasetSummarySeparatesDirectAndAssistantRoutes():
    results = {
        "1": {
            "direct": {"result": {"validationMacroF1": 0.7}},
            "assistant": {
                "assistant": {"result": {"validationMacroF1": 0.8}},
                "student": {"result": {"validationMacroF1": 0.75}},
            },
        },
        "2": {
            "direct": {"result": {"validationMacroF1": 0.8}},
            "assistant": {
                "assistant": {"result": {"validationMacroF1": 0.9}},
                "student": {"result": {"validationMacroF1": 0.85}},
            },
        },
    }
    summary = summarizeDatasetResults(results)
    assert summary["directStudentValidationMacroF1"]["mean"] == pytest.approx(0.75)
    assert summary["distilledAssistantValidationMacroF1"]["mean"] == pytest.approx(0.85)
    assert summary["assistantStudentValidationMacroF1"]["mean"] == pytest.approx(0.8)
