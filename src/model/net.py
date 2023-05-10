"""Define the Network, loss and metrics"""

from typing import Callable, Dict

import torch
import torch.nn as tnn
from torchvision.models import ResNet50_Weights, resnet50

from utils.utils import Params


class Net(tnn.Module):
    """Extend the torch.nn.Module class to define a custom neural network"""

    def __init__(self, params: Params) -> None:
        """Initialize the different layers in the neural network
        Args:
            params: Hyperparameters
        """
        super().__init__()

        self.preprocess = ResNet50_Weights.DEFAULT.transforms()
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
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


def loss_fn(params: Params) -> Callable:
    """Compute the loss given outputs and ground_truth.
    Args:
        params: Hyper-parameters
    Returns:
        loss function
    """
    wts = [
        1.0,
        10.0,
        10.0,
        10.0,
        7.0,
        10.0,
        10.0,
        10.0,
        10.0,
        10.0,
        4.0,
        8.0,
        10.0,
        10.0,
        10.0,
        10.0,
        10.0,
        10.0,
        8.0,
        10.0,
        10.0,
        10.0,
        6.0,
        9.0,
        10.0,
        10.0,
        10.0,
        10.0,
        10.0,
        8.0,
        10.0,
        10.0,
        10.0,
        10.0,
        10.0,
        10.0,
        10.0,
        10.0,
        10.0,
        10.0,
        10.0,
        10.0,
        10.0,
        10.0,
        10.0,
        10.0,
        10.0,
        10.0,
        10.0,
        10.0,
        7.0,
        10.0,
        10.0,
        10.0,
        10.0,
        10.0,
        10.0,
        10.0,
        10.0,
        10.0,
        10.0,
        10.0,
        10.0,
        10.0,
        10.0,
        10.0,
        10.0,
        10.0,
        10.0,
        10.0,
        10.0,
        10.0,
    ]
    criterion = tnn.BCEWithLogitsLoss(pos_weight=torch.as_tensor(wts, device=params.device))
    return criterion


def avg_acc_gpu(outputs: torch.Tensor, labels: torch.Tensor, thr: float = 0.5) -> torch.Tensor:
    """Compute the accuracy, given the outputs and labels for all images.
    Args:
        outputs: Logits of the network
        labels: Ground truth labels
        thr: Threshold
    Returns:
        average accuracy in [0,1]
    """
    outputs = (torch.sigmoid(outputs) > thr).to(torch.float32)
    avg_acc = (outputs == labels).to(torch.float32).mean()
    return avg_acc


def match_gpu(outputs: torch.Tensor, labels: torch.Tensor, thr: float = 0.5) -> torch.Tensor:
    """Compute the match accuracy, given the outputs and labels for all images.
    Args:
        outputs: Logits of the network
        labels: Ground truth labels
        thr: Threshold
    Returns:
        average match accuracy in [0,1]
    """
    outputs = (torch.sigmoid(outputs) > thr).to(torch.float32)
    avg_acc = (outputs == labels).all(1).to(torch.float32).mean()
    return avg_acc


def avg_f1_score_gpu(
    outputs: torch.Tensor, labels: torch.Tensor, thr: float = 0.5, eps: float = 1e-7
) -> torch.Tensor:
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
    wtd_macro_f1 = (avg_f1 * wts).sum() / (wts.sum() + eps)

    return wtd_macro_f1


def avg_precision_gpu(
    outputs: torch.Tensor, labels: torch.Tensor, step: float = 0.1, eps: float = 1e-7
) -> torch.Tensor:
    """Compute the average precision, given the outputs and labels for all images.
    Args:
        outputs: Logits of the network
        labels: Ground truth labels
        step: Step size
        eps: Epsilon
    Returns:
        average precision
    """
    thr = torch.arange(0.0, 1.0 + step, step, device=labels.device).view(-1, 1, 1)
    outputs = (torch.sigmoid(outputs).unsqueeze(0) > thr).to(torch.int32)

    true_pos = (labels * outputs).sum(1)
    false_pos = ((1 - labels) * outputs).sum(1)
    false_neg = (labels * (1 - outputs)).sum(1)

    precision = true_pos / (true_pos + false_pos + eps)
    recall = true_pos / (true_pos + false_neg + eps)

    recall[:-1, :] = recall[:-1, :] - recall[1:, :]
    avg_precision = (recall * precision).sum(0)
    wts = labels.sum(0)
    wtd_avg_prec = (avg_precision * wts).sum() / (wts.sum() + eps)

    return wtd_avg_prec


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
        "match_acc": match_gpu,
        "f1-score": avg_f1_score_gpu,
        "avg_precision": avg_precision_gpu,
    }
    return metrics
