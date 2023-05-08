"""Define the Network, loss and metrics"""

from typing import Callable, Dict

import torch
import torch.nn as tnn
from torchvision.models import EfficientNet_V2_M_Weights, efficientnet_v2_m

from utils.utils import Params


class Net(tnn.Module):
    """Extend the torch.nn.Module class to define a custom neural network"""

    def __init__(self, params: Params) -> None:
        """Initialize the different layers in the neural network
        Args:
            params: Hyperparameters
        """
        super().__init__()

        self.preprocess = EfficientNet_V2_M_Weights.DEFAULT.transforms()
        self.model = efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.DEFAULT)
        in_feats = self.model.classifier[1].in_features
        self.model.classifier[1] = tnn.Linear(in_features=in_feats, out_features=params.num_classes)
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
        22.0,
        37.0,
        11.0,
        7.0,
        17.0,
        65.0,
        10.0,
        366.0,
        252.0,
        4.0,
        8.0,
        12.0,
        122.0,
        50.0,
        55.0,
        844.0,
        50.0,
        8.0,
        1440.0,
        31.0,
        985.0,
        6.0,
        9.0,
        669.0,
        17.0,
        19.0,
        771.0,
        26.0,
        8.0,
        20.0,
        80.0,
        21.0,
        368.0,
        111.0,
        1822.0,
        541.0,
        340.0,
        1329.0,
        266.0,
        83.0,
        48.0,
        1329.0,
        3814.0,
        432.0,
        1622.0,
        163.0,
        80.0,
        260.0,
        187.0,
        7.0,
        102.0,
        122.0,
        262.0,
        17.0,
        322.0,
        2101.0,
        438.0,
        8238.0,
        2145.0,
        592.0,
        936.0,
        353.0,
        375.0,
        1775.0,
        2019.0,
        261.0,
        1157.0,
        9361.0,
        940.0,
        37.0,
        40.0,
    ]
    criterion = tnn.BCEWithLogitsLoss(pos_weight=torch.as_tensor(wts, device=params.device))
    return criterion


def avg_acc_gpu(outputs: torch.Tensor, labels: torch.Tensor, thr: float = 0.5) -> float:
    """Compute the accuracy, given the outputs and labels for all images.
    Args:
        outputs: Logits of the network
        labels: Ground truth labels
        thr: Threshold
    Returns:
        average accuracy in [0,1]
    """
    outputs = (torch.sigmoid(outputs) > thr).to(torch.float32)
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
