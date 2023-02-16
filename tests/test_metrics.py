import numpy as np
import torch

from src.metrics import (
    AUC,
    FPR,
    ExactMatchRatio,
    F1Score,
    LossMetric,
    MeanAveragePrecision,
    MetricCollection,
    Precision,
    Precision_K,
    PrecisionAtRecall,
    Recall,
    Recall_K,
)


def test_recall():
    logits = torch.tensor([[0.2, 0.6, 0.8, 0.4], [0.1, 0.56, 0.9, 0.3]])
    targets = torch.tensor([[0, 1, 1, 1], [0, 1, 1, 0]])
    batch_1 = {"logits": logits[:1], "targets": targets[:1]}
    batch_2 = {"logits": logits[1:], "targets": targets[1:]}
    batch = {"logits": logits, "targets": targets}
    recall = Recall(number_of_classes=4, average="micro", threshold=0.5)
    recall.update(batch)

    assert len(recall._tp) == logits.size(1)
    assert recall.compute() == 4 / 5
    assert isinstance(recall.compute(), torch.Tensor)

    recall.reset()

    assert len(recall._tp) == 4
    assert recall.compute() == 0

    recall = Recall(number_of_classes=4, average="micro", threshold=0.6)
    recall.update(batch)

    assert len(recall._tp) == logits.size(1)
    assert recall.compute() == 2 / 5

    recall = Recall(number_of_classes=4, average="macro", threshold=0.5)
    recall.update(batch)

    assert len(recall._tp) == logits.size(1)
    assert recall.compute() == 0.5

    recall = Recall(number_of_classes=4, average="none", threshold=0.5)
    recall.update(batch)
    assert torch.all(recall.compute() == torch.tensor([0.0, 1.0, 1.0, 0.0]))

    recall = Recall(number_of_classes=4, average="micro", threshold=0.5)
    recall.update(batch_1)
    recall.update(batch_2)
    assert recall.compute() == 4 / 5


def test_precision():
    logits = torch.tensor([[0.2, 0.6, 0.8, 0.2], [0.6, 0.56, 0.9, 0.3]])
    targets = torch.tensor([[0, 1, 1, 1], [0, 1, 1, 0]])
    batch_1 = {"logits": logits[:1], "targets": targets[:1]}
    batch_2 = {"logits": logits[1:], "targets": targets[1:]}
    batch = {"logits": logits, "targets": targets}

    precision = Precision(number_of_classes=4, average="micro", threshold=0.5)
    precision.update(batch)

    assert len(precision._tp) == logits.size(1)
    assert precision.compute() == 4 / 5
    assert isinstance(precision.compute(), torch.Tensor)

    precision.reset()
    assert len(precision._tp) == 4
    assert precision.compute() == 0

    precision = Precision(number_of_classes=4, average="micro", threshold=0.6)
    precision.update(batch)

    assert len(precision._tp) == logits.size(1)
    assert precision.compute() == 1

    precision = Precision(number_of_classes=4, average="macro", threshold=0.5)
    precision.update(batch)
    assert len(precision._tp) == logits.size(1)
    assert precision.compute() == 0.5

    precision = Precision(number_of_classes=4, average="none", threshold=0.5)
    precision.update(batch)
    assert torch.all(precision.compute() == torch.tensor([0.0, 1.0, 1.0, 0.0]))

    precision = Precision(number_of_classes=4, average="micro", threshold=0.5)
    precision.update(batch_1)
    precision.update(batch_2)
    assert precision.compute() == 4 / 5


def test_f1():
    logits = torch.tensor([[0.2, 0.6, 0.8, 0.4], [0.1, 0.56, 0.9, 0.3]])
    targets = torch.tensor([[0, 1, 1, 1], [0, 1, 1, 0]])
    batch_1 = {"logits": logits[:1], "targets": targets[:1]}
    batch_2 = {"logits": logits[1:], "targets": targets[1:]}
    batch = {"logits": logits, "targets": targets}
    f1 = F1Score(number_of_classes=4, average="micro", threshold=0.5)
    f1.update(batch)

    assert round(f1.compute().item(), 1) == 0.9

    f1.reset()

    assert len(f1._tp) == 4
    assert f1.compute() == 0

    f1 = F1Score(number_of_classes=4, average="micro", threshold=0.6)
    f1.update(batch)
    assert round(f1.compute().item(), 1) == 0.6

    f1 = F1Score(number_of_classes=4, average="macro", threshold=0.5)
    f1.update(batch)
    assert f1.compute() == 0.5

    f1 = F1Score(number_of_classes=4, average="none", threshold=0.5)
    f1.update(batch)

    assert torch.all(f1.compute() == torch.tensor([0.0, 1.0, 1.0, 0.0]))

    f1 = F1Score(number_of_classes=4, average="micro", threshold=0.5)
    f1.update(batch_1)
    f1.update(batch_2)
    assert round(f1.compute().item(), 1) == 0.9


def test_exact_match_ratio():
    logits = torch.tensor(
        [[0.2, 0.6, 0.8, 0.4], [0.1, 0.56, 0.9, 0.3], [0.1, 0.56, 0.9, 0.3]]
    )
    targets = torch.tensor([[0, 1, 1, 1], [0, 1, 1, 0], [1, 1, 1, 0]])
    batch_1 = {"logits": logits[:1], "targets": targets[:1]}
    batch_2 = {"logits": logits[1:2], "targets": targets[1:2]}
    batch_3 = {"logits": logits[2:], "targets": targets[2:]}
    batch = {"logits": logits, "targets": targets}
    emr = ExactMatchRatio(number_of_classes=4, threshold=0.5)
    emr.update(batch)
    assert emr._num_exact_matches == 1
    assert emr._num_examples == 3
    assert round(emr.compute().item(), 2) == 0.33

    emr.reset()
    assert emr._num_exact_matches == 0

    emr = ExactMatchRatio(number_of_classes=4, threshold=0.6)
    emr.update(batch)
    assert emr.compute() == 0.0

    emr = ExactMatchRatio(number_of_classes=4, threshold=0.5)
    emr.update(batch_1)
    emr.update(batch_2)
    emr.update(batch_3)
    assert round(emr.compute().item(), 2) == 0.33


def test_fpr():
    logits = torch.tensor([[0.6, 0.6, 0.8, 0.4], [0.1, 0.56, 0.9, 0.3]])
    targets = torch.tensor([[0, 1, 1, 1], [0, 1, 1, 0]])
    batch_1 = {"logits": logits[:1], "targets": targets[:1]}
    batch_2 = {"logits": logits[1:], "targets": targets[1:]}
    batch = {"logits": logits, "targets": targets}
    fpr = FPR(number_of_classes=4, threshold=0.5)
    fpr.update(batch)
    assert fpr.compute() == 1 / 3

    fpr = FPR(number_of_classes=4, threshold=0.6)
    fpr.update(batch)
    assert fpr.compute() == 0

    fpr = FPR(number_of_classes=4, average="macro", threshold=0.5)
    fpr.update(batch)
    assert len(fpr._fp) == logits.size(1)
    assert fpr.compute() == 0.125

    fpr = FPR(number_of_classes=4, threshold=0.5)
    fpr.update(batch_1)
    fpr.update(batch_2)
    assert fpr.compute() == 1 / 3


def test_auc():
    logits = torch.tensor([[0.6, 0.6, 0.8, 0.4], [0.1, 0.56, 0.9, 0.3]])
    targets = torch.tensor([[0, 1, 1, 1], [0, 1, 1, 0]])
    auc = AUC(average="micro")

    assert round(auc.compute(logits=logits, targets=targets).item(), 2) == 0.83

    auc = AUC(average="macro")

    assert round(auc.compute(logits=logits, targets=targets).item(), 2) == 1.0


def test_precision_k():
    logits = torch.tensor([[0.6, 0.6, 0.8, 0.4], [0.1, 0.56, 0.9, 0.3]])
    targets = torch.tensor([[0, 1, 1, 1], [0, 1, 1, 0]])
    batch_1 = {"logits": logits[:1], "targets": targets[:1]}
    batch_2 = {"logits": logits[1:], "targets": targets[1:]}
    batch = {"logits": logits, "targets": targets}
    pk = Precision_K(k=2)
    pk.update(batch)

    assert pk.compute() == 1.0

    pk = Precision_K(k=4)
    pk.update(batch)
    assert pk.compute() == np.mean([(3 / 4), (2 / 4)])

    pk = Precision_K(k=2)
    pk.update(batch_1)
    pk.update(batch_2)
    assert pk.compute() == 1.0


def test_recall_k():
    logits = torch.tensor([[0.5, 0.6, 0.8, 0.4], [0.1, 0.56, 0.9, 0.3]])
    targets = torch.tensor([[0, 1, 1, 1], [0, 1, 1, 0]])
    batch_1 = {"logits": logits[:1], "targets": targets[:1]}
    batch_2 = {"logits": logits[1:], "targets": targets[1:]}
    batch = {"logits": logits, "targets": targets}
    recall = Recall_K(k=2)
    recall.update(batch)

    assert round(recall.compute().item(), 2) == round(np.mean([(2 / 3), (2 / 2)]), 2)

    recall = Recall_K(k=4)
    recall.update(batch)
    assert recall.compute() == 1

    recall = Recall_K(k=2)
    recall.update(batch_1)
    recall.update(batch_2)
    assert round(recall.compute().item(), 2) == round(np.mean([(2 / 3), (2 / 2)]), 2)


def test_metric_collection() -> None:
    logits = torch.tensor([[0.5, 0.6, 0.8, 0.4], [0.1, 0.56, 0.9, 0.3]])
    targets = torch.tensor([[0, 1, 1, 1], [0, 1, 1, 0]])

    batch_1 = {"logits": logits[:1], "targets": targets[:1]}
    batch_2 = {"logits": logits[1:], "targets": targets[1:]}
    batch = {"logits": logits, "targets": targets}
    threshold = 0.5
    num_classes = 4
    metrics = MetricCollection(
        [
            F1Score(
                threshold=threshold, number_of_classes=num_classes, average="micro"
            ),
            F1Score(
                threshold=threshold, number_of_classes=num_classes, average="macro"
            ),
        ]
    )
    metrics.update(batch)
    assert len(metrics.compute()) == 2
    assert round(metrics.compute()["f1_micro"].item(), 1) == 0.9
    assert round(metrics.compute()["f1_macro"].item(), 1) == 0.5

    metrics.reset()
    assert len(metrics.compute()) == 2
    assert metrics.compute() == {"f1_micro": 0.0, "f1_macro": 0.0}

    metrics.update(batch_1)
    metrics.update(batch_2)
    assert len(metrics.compute()) == 2

    metrics = MetricCollection(
        [
            AUC(average="micro"),
            FPR(threshold=0.3, number_of_classes=num_classes, average="micro"),
            F1Score(
                threshold=threshold, number_of_classes=num_classes, average="micro"
            ),
            F1Score(
                threshold=threshold, number_of_classes=num_classes, average="macro"
            ),
        ]
    )
    metrics.update(batch)
    assert len(metrics.compute()) == 3
    assert round(metrics.compute()["f1_micro"].item(), 1) == 0.9
    assert round(metrics.compute()["f1_macro"].item(), 1) == 0.5
    assert round(metrics.compute()["fpr_micro"].item(), 1) == 0.3

    assert len(metrics.compute(logits=logits, targets=targets)) == 4
    assert (
        round(
            metrics.compute(logits=logits, targets=targets)[
                "auc_micro"
            ].item(),
            2,
        )
        == 0.93
    )

    assert len(metrics.compute()) == 3

    assert round(metrics.get_best_metric("f1_micro").item(), 1) == 0.9
    assert round(metrics.get_best_metric("f1_macro").item(), 1) == 0.5
    assert round(metrics.get_best_metric("auc_micro").item(), 2) == 0.93
    assert round(metrics.get_best_metric("fpr_micro").item(), 2) == 0.33

    metrics.reset_metrics()

    assert round(metrics.get_best_metric("f1_micro").item(), 1) == 0.9
    assert round(metrics.get_best_metric("f1_macro").item(), 1) == 0.5
    assert round(metrics.get_best_metric("auc_micro").item(), 2) == 0.93
    assert round(metrics.get_best_metric("fpr_micro").item(), 2) == 0.33

    metrics2 = metrics.copy()

    assert round(metrics2.get_best_metric("f1_micro").item(), 1) == 0.9
    assert round(metrics2.get_best_metric("f1_macro").item(), 1) == 0.5
    assert round(metrics2.get_best_metric("auc_micro").item(), 2) == 0.93
    assert round(metrics2.get_best_metric("fpr_micro").item(), 2) == 0.33

    logits = torch.tensor([[0, 0.7, 0.8, 0.6], [0.1, 0.8, 0.9, 0.3]])
    targets = torch.tensor([[0, 1, 1, 1], [0, 1, 1, 0]])
    batch = {"logits": logits, "targets": targets}
    metrics.update(batch)
    assert (
        round(
            metrics.compute(logits=logits, targets=targets)[
                "auc_micro"
            ].item(),
            2,
        )
        == 1.0
    )

    assert round(metrics.get_best_metric("f1_micro").item(), 1) == 1.0
    assert round(metrics.get_best_metric("f1_macro").item(), 1) == 0.8
    assert round(metrics.get_best_metric("auc_micro").item(), 2) == 1.0
    assert round(metrics.get_best_metric("fpr_micro").item(), 2) == 0.0

    metrics.reset()

    assert metrics.get_best_metric("f1_micro") == None
    assert metrics.get_best_metric("f1_macro") == None
    assert metrics.get_best_metric("auc_micro") == None
    assert metrics.get_best_metric("fpr_micro") == None

    logits = torch.tensor([[0.5, 0.6, 0.8, 0.4], [0.1, 0.56, 0.9, 0.3]])
    targets = torch.tensor([[0, 1, 1, 1], [0, 1, 1, 0]])

    batch_1 = {"logits": logits[:1], "targets": targets[:1]}
    batch_2 = {"logits": logits[1:], "targets": targets[1:]}
    batch = {"logits": logits, "targets": targets}
    metrics = MetricCollection(
        [
            AUC(average="micro"),
            Precision(
                threshold=threshold, number_of_classes=num_classes, average="micro"
            ),
            Recall(threshold=threshold, number_of_classes=num_classes, average="micro"),
            Precision(
                threshold=threshold, number_of_classes=num_classes, average="macro"
            ),
            Recall(threshold=threshold, number_of_classes=num_classes, average="macro"),
            F1Score(
                threshold=threshold, number_of_classes=num_classes, average="micro"
            ),
            F1Score(
                threshold=threshold, number_of_classes=num_classes, average="macro"
            ),
        ],
        code_indices=torch.tensor([0, 2]),
        code_system_name="targets",
    )
    assert len(metrics.code_indices) == 2
    assert metrics.code_system_name == "targets"
    assert metrics.metrics[1].number_of_classes == 2
    assert len(metrics.metrics[1]._tp) == 2
    metrics.update(batch)
    assert len(metrics.compute()) == 6
    assert round(metrics.compute()["precision_micro"].item(), 2) == 1
    assert round(metrics.compute()["recall_micro"].item(), 2) == 1
    assert round(metrics.compute()["precision_macro"].item(), 2) == 0.5
    assert round(metrics.compute()["recall_macro"].item(), 2) == 0.5
    assert round(metrics.compute()["f1_micro"].item(), 2) == 1
    assert round(metrics.compute()["f1_macro"].item(), 2) == 0.5
    assert (
        round(
            metrics.compute(logits=logits, targets=targets)[
                "auc_micro"
            ].item(),
            2,
        )
        == 0.93
    )

    metrics.reset()

    metrics.update(batch_1)
    metrics.update(batch_2)

    assert len(metrics.compute()) == 6
    assert round(metrics.compute()["precision_micro"].item(), 2) == 1
    assert round(metrics.compute()["recall_micro"].item(), 2) == 1
    assert round(metrics.compute()["precision_macro"].item(), 2) == 0.5
    assert round(metrics.compute()["recall_macro"].item(), 2) == 0.5
    assert round(metrics.compute()["f1_micro"].item(), 2) == 1
    assert round(metrics.compute()["f1_macro"].item(), 2) == 0.5

    logits = torch.tensor([[0.5, 0.6, 0.8, 0.4], [0.1, 0.56, 0.9, 0.3]])
    targets = torch.tensor([[0, 1, 0, 1], [1, 1, 1, 0]])
    batch = {"logits": logits, "targets": targets}
    metrics.reset()

    assert len(metrics.code_indices) == 2
    assert metrics.code_system_name == "targets"
    assert metrics.metrics[1].number_of_classes == 2
    assert len(metrics.metrics[1]._tp) == 2
    metrics.update(batch)
    assert len(metrics.compute()) == 6
    assert round(metrics.compute()["precision_micro"].item(), 2) == 1
    assert round(metrics.compute()["recall_micro"].item(), 2) == 0.5
    assert round(metrics.compute()["precision_macro"].item(), 2) == 0.5
    assert round(metrics.compute()["recall_macro"].item(), 2) == 0.5
    assert round(metrics.compute()["f1_micro"].item(), 2) == 0.67

    assert round(metrics.compute()["f1_macro"].item(), 2) == 0.5
    assert (
        round(
            metrics.compute(logits=logits, targets=targets)[
                "auc_micro"
            ].item(),
            2,
        )
        == 0.53
    )


def test_loss_metric() -> None:
    logits = torch.tensor([[0.5, 0.6, 0.8, 0.4], [0.1, 0.56, 0.9, 0.3]])
    targets = torch.tensor([[0, 1, 1, 1], [0, 1, 1, 0]])
    loss = torch.tensor([0.5, 0.6])
    batch = {"logits": logits, "targets": targets, "loss": loss}
    loss_metric = LossMetric()
    loss_metric.update(batch)
    assert loss_metric.compute() == 0.55


def test_map_metric() -> None:
    logits = torch.tensor([[0.5, 0.6, 0.8, 0.4], [0.1, 0.56, 0.9, 0.3]])
    targets = torch.tensor([[0, 1, 1, 1], [0, 1, 1, 0]])
    batch = {"logits": logits, "targets": targets}
    map_metric = MeanAveragePrecision()
    map_metric.update(batch)

    assert round(map_metric.compute().item(), 2) == 0.96

    map_metric.reset()

    logits = torch.tensor([[0.9, 0.6, 0.8, 0.4], [0.1, 0.56, 0.4, 0.3]])
    targets = torch.tensor([[0, 1, 0, 1], [0, 1, 1, 1]])
    batch = {"logits": logits, "targets": targets}

    map_metric.update(batch)

    assert round(map_metric.compute().item(), 3) == 0.708


def test_precision_at_recall() -> None:
    logits = torch.tensor([[0.5, 0.6, 0.8, 0.4], [0.1, 0.56, 0.9, 0.3]])
    targets = torch.tensor([[0, 1, 1, 1], [0, 1, 1, 0]])
    batch = {"logits": logits, "targets": targets}
    precision_at_recall = PrecisionAtRecall()
    precision_at_recall.update(batch)

    assert round(precision_at_recall.compute().item(), 2) == 0.83

    precision_at_recall.reset()

    logits = torch.tensor([[0.2, 0.6, 0.9, 0.4], [0.9, 0.56, 0.9, 0.0]])
    targets = torch.tensor([[0, 1, 0, 1], [0, 1, 1, 1]])
    batch = {"logits": logits, "targets": targets}
    precision_at_recall.update(batch)

    assert round(precision_at_recall.compute().item(), 2) == round((0.5 + 0.66) / 2, 2)
