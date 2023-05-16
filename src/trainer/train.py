#!/usr/bin/env python3
"""Script to train a model in pytorch"""

import argparse
import logging
import os
from typing import Any, Callable, Dict, Optional, Union

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from dataloader.data_loader import get_dataloader
from model.net import Net, get_metrics, loss_fn
from trainer.evaluate import evaluate
from utils import utils


def args_parser() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Training utility")
    parser.add_argument(
        "-d",
        "--data_dir",
        default="gs://hm_images",
        type=str,
        help="Directory containing the dataset",
    )
    parser.add_argument(
        "-m",
        "--model_dir",
        default=os.getenv("AIP_MODEL_DIR", "gs://attributes_models/base_model/model"),
        type=str,
        help="Directory containing model",
    )
    parser.add_argument(
        "--tb_log_dir",
        default=os.getenv("AIP_TENSORBOARD_LOG_DIR", "gs://attributes_models/base_model/logs"),
        type=str,
        help="TensorBoard summarywriter directory",
    )
    parser.add_argument("--locally", action="store_true", help="Train locally")
    parser.add_argument(
        "-r",
        "--restore_file",
        default=None,
        choices=["best", "last"],
        type=str,
        help="Optional, name of the file in --model_dir containing weights to restore",
    )
    parser.add_argument(
        "--world_size",
        type=int,
        default=os.environ.get("WORLD_SIZE", 1),
        help="The total number of nodes in the cluster",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=os.environ.get("RANK", 0),
        help="Identifier for each node",
    )
    parser.add_argument("--height", default=232, type=int, help="Image height")
    parser.add_argument("-w", "--width", default=232, type=int, help="Image width")
    parser.add_argument("--crop", default=224, type=int, help="Center crop image")
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size")
    parser.add_argument("--num_workers", default=2, type=int, help="Number of workers to load data")
    parser.add_argument(
        "--pin_memory",
        default=True,
        type=bool,
        help="Pin memory for faster load on GPU",
    )
    parser.add_argument("--num_classes", default=90, type=int, help="Number of classes")
    parser.add_argument("--dropout", default=0.5, type=float, help="Dropout rate")
    parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate")
    parser.add_argument("--decay", default=0.0, type=float, help="Decay rate")
    parser.add_argument("--policy", default="steps", type=str, help="Learning rate scheduler")
    parser.add_argument("--steps", default=[10, 20], help="Steps for learning rate scheduler")
    parser.add_argument(
        "--save_summary_steps", default=500, type=int, help="Save after number of steps"
    )
    parser.add_argument("--num_epochs", default=50, type=int, help="Number of epochs")

    # Augmentation related arguments
    parser.add_argument("-f", "--flip", default=0.5, type=float, help="Probability to flip image")
    parser.add_argument(
        "-b",
        "--brightness",
        default=0.3,
        type=float,
        help="Brightness level for augmentation",
    )
    parser.add_argument(
        "-c",
        "--contrast",
        default=1.5,
        type=float,
        help="Contrast level for augmentation",
    )
    parser.add_argument(
        "-s",
        "--saturation",
        default=1.5,
        type=float,
        help="Saturation level for augmentation",
    )
    parser.add_argument("--hue", default=0.0, type=float, help="Hue level for augmentation")
    parser.add_argument("--degree", default=1.5, type=float, help="Degree level for augmentation")
    return parser.parse_args()


def train(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: Callable,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    dataloader: DataLoader,
    metrics: Dict[str, Any],
    params: utils.Params,
    writer: Optional[SummaryWriter],
    epoch: int,
) -> None:
    """Train the model.
    Args:
        model: Neural network
        optimizer: Optimizer for parameters of model
        criterion: A function that computes the loss for the batch
        scheduler: Learning rate scheduler
        dataloader: Training data
        metrics: A dictionary of metrics
        params: Hyperparameters
        writer : Summary writer for tensorboard
        epoch: Value of Epoch
    """
    model.train()
    summ = []

    for i, (train_batch, labels) in enumerate(dataloader):
        if params.cuda:
            train_batch = train_batch.to(params.device)
            labels = labels.to(params.device)

        output_batch = model(train_batch)
        loss = criterion(output_batch, labels)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        if i % params.save_summary_steps == 0:
            output_batch = output_batch.detach()
            labels = labels.detach()

            summary_batch = {metric: metrics[metric](output_batch, labels) for metric in metrics}
            summary_batch["loss"] = loss.detach()
            if params.distributed:
                summary_batch = utils.reduce_dict(summary_batch)
            summ.append(summary_batch)

            if params.rank == 0 and writer:
                tmp = {k: v.item() for k, v in summary_batch.items()}
                writer.add_scalars("train", tmp, epoch * len(dataloader) + i)

    scheduler.step()
    metrics_mean = {metric: np.mean([x[metric].item() for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join(f"{k}: {v:05.3f}" for k, v in metrics_mean.items())
    logging.info("- Train metrics: %s", metrics_string)


def train_and_evaluate(
    model: torch.nn.Module,
    train_dataloader: DataLoader,
    train_sampler: Optional[DistributedSampler],
    val_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: Callable,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    metrics: Dict[str, Any],
    params: utils.Params,
    writer: Optional[SummaryWriter],
) -> None:
    """Train the model and evaluate every epoch.
    Args:
        model: Neural network
        train_dataloader: Training dataloader
        train_dampler: Distributed data sampler
        val_dataloader: Validation dataloader
        optimizer: Optimizer for parameters of model
        criterion: A function to compute the loss for the batch
        scheduler: Learning rate scheduler
        metrics: A dictionary of metric functions
        params: Hyperparameters
        writer : Summary writer for tensorboard
    """
    if params.restore_file is not None:
        restore_path = os.path.join(params.model_dir, params.restore_file + ".pth.tar")
        logging.info("Restoring parameters from %s", restore_path)
        utils.load_checkpoint(restore_path, model, optimizer)

    best_val_acc = 0.0

    for epoch in range(params.num_epochs):
        if train_sampler:
            train_sampler.set_epoch(epoch)
        logging.info("Epoch %d / %d", epoch + 1, params.num_epochs)

        train(
            model,
            optimizer,
            criterion,
            scheduler,
            train_dataloader,
            metrics,
            params,
            writer,
            epoch,
        )

        val_metrics = evaluate(model, criterion, val_dataloader, metrics, params, writer, epoch)

        val_acc = val_metrics.get("f1-score", 0.0)
        is_best = val_acc > best_val_acc

        if params.rank == 0:
            utils.save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "optim_dict": optimizer.state_dict(),
                },
                is_best=is_best,
                checkpoint=params.model_dir,
            )

        if is_best:
            logging.info("- Found new best accuracy")
            best_val_acc = val_acc

            best_yml_path = os.path.join(params.model_dir, "metrics_val_best.yml")
            if params.rank == 0:
                utils.save_dict_to_yaml(val_metrics, best_yml_path)

    last_yml_path = os.path.join(params.model_dir, "metrics_val_last.yml")
    if params.rank == 0:
        utils.save_dict_to_yaml(val_metrics, last_yml_path)


def main() -> None:
    """Main function"""
    args = args_parser()
    params = utils.Params(vars(args))

    if not params.locally:
        params.data_dir.replace("gs://", "/gcs/")
        params.model_dir.replace("gs://", "/gcs/")
        params.tb_log_dir.replace("gs://", "/gcs/")

    params.cuda = torch.cuda.is_available()
    utils.setup_distributed(params)

    writer = SummaryWriter(params.tb_log_dir) if params.rank == 0 else None

    torch.manual_seed(230)
    if params.cuda:
        torch.cuda.manual_seed(230)
        params.device = f"cuda:{params.local_rank}"
    else:
        params.device = "cpu"

    utils.set_logger()

    logging.info("Configurations: %s", str(params))
    logging.info("Loading the datasets...")

    dataloaders = get_dataloader(["train", "val"], params)
    train_dl, train_sampler = dataloaders["train"]
    val_dl, _ = dataloaders["val"]

    logging.info("- done.")

    model: Union[DistributedDataParallel, torch.nn.Module] = Net(params)
    if params.cuda:
        model = model.to(params.device)
    if params.rank == 0 and writer:
        writer.add_graph(model, next(iter(train_dl))[0].to(params.device))
    if params.distributed:
        model = DistributedDataParallel(model, device_ids=[params.local_rank])

    optimizer = torch.optim.Adam(
        model.parameters(), lr=params.learning_rate, weight_decay=params.decay
    )
    if params.policy == "steps":
        scheduler: torch.optim.lr_scheduler._LRScheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=params.steps, gamma=0.1, verbose=True
        )
    else:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9, verbose=True)

    criterion = loss_fn(params)
    metrics = get_metrics()

    logging.info("Starting training for %d epoch(s)", params.num_epochs)
    train_and_evaluate(
        model,
        train_dl,
        train_sampler,
        val_dl,
        optimizer,
        criterion,
        scheduler,
        metrics,
        params,
        writer,
    )

    if params.rank == 0 and writer:
        writer.close()
    if params.distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
