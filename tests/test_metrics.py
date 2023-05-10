"""Unit tests for metrics"""

from typing import Tuple

import numpy as np
import torch
from numpy.random import default_rng
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    hamming_loss,
    multilabel_confusion_matrix,
)

from model.net import (
    avg_acc_gpu,
    avg_f1_score_gpu,
    avg_precision_gpu,
    confusion_matrix,
    match_gpu,
)

RNG = default_rng()


def _create_data() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Create data to test"""
    high_val = RNG.integers(3, 50)
    threshold = RNG.uniform()

    num_classes = RNG.integers(2, high_val)
    num_examples = RNG.integers(1, high_val)

    output = torch.randn(num_examples, num_classes)
    preds = (torch.sigmoid(output) > threshold).to(torch.int32)
    labels = torch.randint(0, 2, (num_examples, num_classes), dtype=torch.int32)

    return output, preds, labels, threshold


def test_accuracy() -> None:
    """Test implementation of accuracy"""
    output, preds, labels, threshold = _create_data()

    sk_acc = 1.0 - hamming_loss(labels.numpy(), preds.numpy())
    my_acc = avg_acc_gpu(output, labels, threshold).item()

    assert np.isclose(my_acc, sk_acc)


def test_match() -> None:
    """Test implementation of match accuracy"""
    output, preds, labels, threshold = _create_data()

    sk_match = accuracy_score(labels.numpy(), preds.numpy())
    my_match = match_gpu(output, labels, threshold).item()

    assert np.isclose(my_match, sk_match)


def test_f1() -> None:
    """Test f1 score calculation"""
    output, preds, labels, threshold = _create_data()

    sk_f1 = f1_score(labels.numpy(), preds.numpy(), average="weighted")
    my_f1 = avg_f1_score_gpu(output, labels, threshold).item()

    assert np.isclose(my_f1, sk_f1)


def test_ap() -> None:
    """Test ap score calculation"""
    output, _, labels, _ = _create_data()

    sk_ap = average_precision_score(labels.numpy(), output.numpy(), average="weighted")
    my_ap = avg_precision_gpu(output, labels).item()

    assert np.isclose(my_ap, sk_ap, atol=1e-1)


def test_conf_mat() -> None:
    """Test f1 score calculation"""
    output, preds, labels, threshold = _create_data()

    sk_cm = multilabel_confusion_matrix(labels.numpy(), preds.numpy())
    my_cm = confusion_matrix(output, labels, threshold)

    assert (sk_cm == my_cm.numpy()).all()
