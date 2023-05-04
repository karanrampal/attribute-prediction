"""Define the Network, loss and metrics"""

from typing import Callable, Dict

import torch
import torch.nn as tnn
from torchvision import models

from utils.utils import Params


class Net(tnn.Module):
    """Extend the torch.nn.Module class to define a custom neural network"""

    def __init__(self, params: Params) -> None:
        """Initialize the different layers in the neural network
        Args:
            params: Hyperparameters
        """
        super().__init__()

        self.model = models.resnet18(pretrained=True)
        in_feats = self.model.fc.in_features
        self.model.fc = tnn.Linear(in_features=in_feats, out_features=params.num_classes)
        self.dropout_rate = params.dropout

    def forward(self, x_inp: torch.Tensor) -> torch.Tensor:
        """Defines the forward propagation through the network
        Args:
            x_inp: Batch of images
        Returns:
            logits
        """
        return self.model(x_inp)


def loss_fn(outputs: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:
    """Compute the loss given outputs and ground_truth.
    Args:
        outputs: Logits of network forward pass
        ground_truth: Batch of ground truth
    Returns:
        loss for all the inputs in the batch
    """
    criterion = tnn.BCEWithLogitsLoss()
    loss = criterion(outputs, ground_truth)
    return loss


def avg_acc_gpu(outputs: torch.Tensor, labels: torch.Tensor, thr: float = 0.5) -> float:
    """Compute the accuracy, given the outputs and labels for all images.
    Args:
        outputs: Logits of the network
        labels: Ground truth labels
        thr: Threshold
    Returns:
        average accuracy in [0,1]
    """
    outputs = (torch.sigmoid(outputs) > thr).to(torch.int32)
    avg_acc = (outputs == labels).all(1).to(torch.float32).mean()
    return avg_acc.item()


def avg_f1_score_gpu(
    outputs: torch.Tensor, labels: torch.Tensor, thr: float = 0.5, eps: float = 1e-7
) -> float:
    """Compute the F1 score, given the outputs and labels for all images.
    Args:
        outputs: Logits of the network
        labels: Ground truth labels
        thr: Threshold
        eps: Epsilon
    Returns:
        average f1 score
    """
    outputs = (torch.sigmoid(outputs) > thr).to(torch.int32)

    true_pos = (labels * outputs).sum(0)
    false_pos = ((1 - labels) * outputs).sum(0)
    false_neg = (labels * (1 - outputs)).sum(0)

    precision = true_pos / (true_pos + false_pos + eps)
    recall = true_pos / (true_pos + false_neg + eps)
    avg_f1 = 2 * (precision * recall) / (precision + recall + eps)
    wts = labels.sum(0)
    wtd_macro_f1 = (avg_f1 * wts).sum() / wts.sum()

    return wtd_macro_f1.item()


def confusion_matrix(outputs: torch.Tensor, labels: torch.Tensor, thr: float = 0.5) -> torch.Tensor:
    """Create confusion matrix
    Args:
        outputs: Logits of the network
        labels: Ground truth labels
        thr: Threshold
    Returns:
        Confusion matrix as a tensor
    """
    conf_mat = torch.zeros((labels.shape[1], 2, 2), dtype=torch.int32)
    outputs = (torch.sigmoid(outputs) > thr).to(torch.int32)

    conf_mat[:, 0, 0] = ((1 - labels) * (1 - outputs)).sum(0)
    conf_mat[:, 0, 1] = ((1 - labels) * outputs).sum(0)
    conf_mat[:, 1, 0] = (labels * (1 - outputs)).sum(0)
    conf_mat[:, 1, 1] = (labels * outputs).sum(0)

    return conf_mat


# Maintain all metrics required during training and evaluation.
def get_metrics() -> Dict[str, Callable]:
    """Returns a dictionary of all the metrics to be used"""
    metrics: Dict[str, Callable] = {
        "accuracy": avg_acc_gpu,
        "f1-score": avg_f1_score_gpu,
    }
    return metrics
