"""Unit tests for metrics"""

from typing import Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, multilabel_confusion_matrix

from model.net import avg_acc_gpu, avg_f1_score_gpu
from model.net import confusion_matrix as conf_mat

HIGH_VAL = 30
THRESHOLD = 0.5


def _create_data() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create data to test"""
    num_classes = np.random.randint(1, HIGH_VAL)
    num_examples = np.random.randint(1, HIGH_VAL)

    output = torch.randn(num_examples, num_classes)
    preds = (torch.sigmoid(output) > THRESHOLD).to(torch.int32)
    labels = torch.randint(0, 2, (num_examples, num_classes))

    return output, preds, labels


def test_accuracy() -> None:
    """Test implementation of accuracy"""
    output, preds, labels = _create_data()

    sk_acc = accuracy_score(labels.numpy(), preds.numpy())
    my_acc = avg_acc_gpu(output, labels).item()

    assert np.isclose(sk_acc, my_acc)


def test_f1() -> None:
    """Test f1 score calculation"""
    output, preds, labels = _create_data()

    sk_f1 = f1_score(labels.numpy(), preds.numpy(), average="weighted")
    my_f1 = avg_f1_score_gpu(output, labels).item()

    assert np.isclose(sk_f1, my_f1)


def test_conf_mat() -> None:
    """Test f1 score calculation"""
    output, preds, labels = _create_data()

    sk_cm = multilabel_confusion_matrix(labels.numpy(), preds.numpy())
    my_cm = conf_mat(output, labels)

    assert (sk_cm == my_cm.numpy()).all()
