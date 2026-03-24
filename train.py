"""Training script for VRDL HW1 image classification.

This script trains a ResNet-based classifier on an ImageFolder dataset using
train/val splits stored in ``data/train`` and ``data/val`` by default.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.amp import GradScaler, autocast
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
SUPPORTED_MODELS = {"resnet18", "resnet50"}


@dataclass
class TrainConfig:
    """Configuration for training."""

    train_dir: str = "data/train"
    val_dir: str = "data/val"
    output_dir: str = "outputs"
    model_name: str = "resnet50"
    epochs: int = 25
    batch_size: int = 32
    num_workers: int = 4
    patience: int = 5
    fc_lr: float = 4e-4
    backbone_lr: float = 8e-5
    weight_decay: float = 1e-2
    dropout: float = 0.5
    label_smoothing: float = 0.1
    pretrained: bool = True
    freeze_backbone: bool = False
    image_size: int = 224
    seed: int = 42


@dataclass
class EpochMetrics:
    """Container for epoch metrics."""

    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float


class TrainerError(RuntimeError):
    """Raised when training preconditions are not met."""


def parse_args() -> TrainConfig:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train a ResNet model for HW1.")
    parser.add_argument("--train-dir", default="data/train")
    parser.add_argument("--val-dir", default="data/val")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--model-name", choices=sorted(SUPPORTED_MODELS), default="resnet50")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--fc-lr", type=float, default=4e-4)
    parser.add_argument("--backbone-lr", type=float, default=8e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--freeze-backbone", action="store_true")
    args = parser.parse_args()

    return TrainConfig(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        output_dir=args.output_dir,
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        patience=args.patience,
        fc_lr=args.fc_lr,
        backbone_lr=args.backbone_lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        label_smoothing=args.label_smoothing,
        pretrained=not args.no_pretrained,
        freeze_backbone=args.freeze_backbone,
        image_size=args.image_size,
        seed=args.seed,
    )


def set_reproducibility(seed: int) -> None:
    """Set random seeds for reproducible results."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Return the active device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    """Create training and validation transforms."""
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.8, 1.0),
                ratio=(3.0 / 4.0, 4.0 / 3.0),
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(15),
            transforms.RandomAffine(
                degrees=10,
                translate=(0.05, 0.05),
                scale=(0.9, 1.1),
                shear=5,
            ),
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=0.15,
                        contrast=0.15,
                        saturation=0.15,
                        hue=0.05,
                    )
                ],
                p=0.6,
            ),
            transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            transforms.RandomErasing(
                p=0.2,
                scale=(0.02, 0.2),
                ratio=(0.3, 3.3),
                value="random",
            ),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return train_transform, val_transform


def build_dataloaders(
    config: TrainConfig,
    device: torch.device,
) -> tuple[DataLoader[Any], DataLoader[Any], list[str]]:
    """Build dataloaders for training and validation."""
    train_transform, val_transform = build_transforms(config.image_size)

    train_dataset = datasets.ImageFolder(config.train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(config.val_dir, transform=val_transform)

    if len(train_dataset) == 0:
        raise TrainerError(f"Training dataset is empty: {config.train_dir}")
    if len(val_dataset) == 0:
        raise TrainerError(f"Validation dataset is empty: {config.val_dir}")

    classes = train_dataset.classes
    if classes != val_dataset.classes:
        raise TrainerError("Train and validation class folders do not match.")

    pin_memory = device.type == "cuda"
    loader_kwargs = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": config.num_workers > 0,
    }

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, classes


def build_model(
    model_name: str,
    num_classes: int,
    dropout: float,
    pretrained: bool,
    freeze_backbone: bool,
    device: torch.device,
) -> nn.Module:
    """Create a ResNet model with a custom classification head."""
    if model_name == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
    elif model_name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=weights)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )

    if freeze_backbone:
        for parameter in model.fc.parameters():
            parameter.requires_grad = True

    return model.to(device)


def build_optimizer(
    model: nn.Module,
    config: TrainConfig,
) -> tuple[optim.Optimizer, optim.lr_scheduler.ReduceLROnPlateau]:
    """Build optimizer and scheduler."""
    fc_params = []
    backbone_params = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("fc"):
            fc_params.append(parameter)
        else:
            backbone_params.append(parameter)

    param_groups = []
    if fc_params:
        param_groups.append({"params": fc_params, "lr": config.fc_lr})
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": config.backbone_lr})

    optimizer = optim.AdamW(
        param_groups,
        weight_decay=config.weight_decay,
        eps=1e-8,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
        min_lr=1e-7,
    )
    return optimizer, scheduler


def accuracy_from_logits(outputs: Tensor, labels: Tensor) -> int:
    """Count correct predictions from model outputs."""
    predictions = outputs.argmax(dim=1)
    return int((predictions == labels).sum().item())


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader[Any],
    criterion: nn.Module,
    device: torch.device,
    optimizer: optim.Optimizer | None = None,
    scaler: GradScaler | None = None,
    epoch_index: int = 0,
    total_epochs: int = 1,
    split_name: str = "Train",
) -> tuple[float, float]:
    """Run a single epoch for training or validation."""
    is_training = optimizer is not None
    if is_training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    progress_bar = tqdm(
        dataloader,
        desc=f"Epoch {epoch_index + 1}/{total_epochs} [{split_name}]",
        leave=False,
    )

    for images, labels in progress_bar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        autocast_enabled = device.type == "cuda"
        with torch.set_grad_enabled(is_training):
            with autocast(device_type=device.type, enabled=autocast_enabled):
                outputs = model(images)
                loss = criterion(outputs, labels)

            if is_training:
                if scaler is None:
                    raise TrainerError("GradScaler is required during training.")
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += accuracy_from_logits(outputs, labels)
        total_samples += batch_size

        progress_bar.set_postfix(
            loss=f"{total_loss / total_samples:.4f}",
            acc=f"{total_correct / total_samples:.4f}",
        )

    average_loss = total_loss / total_samples
    average_acc = total_correct / total_samples
    return average_loss, average_acc


def save_json(path: Path, data: Any) -> None:
    """Save a JSON file with indentation."""
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def save_checkpoint(
    path: Path,
    model: nn.Module,
    config: TrainConfig,
    classes: list[str],
    best_val_acc: float,
) -> None:
    """Save a training checkpoint."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_name": config.model_name,
        "num_classes": len(classes),
        "classes": classes,
        "dropout": config.dropout,
        "best_val_acc": best_val_acc,
        "config": asdict(config),
    }
    torch.save(checkpoint, path)


def to_float_list(values: list[float], name: str) -> list[float]:
    """Validate metric lists before plotting."""
    cleaned = []
    for index, value in enumerate(values):
        value = float(value)
        if not isfinite(value):
            raise ValueError(f"{name}[{index}] is not finite: {value}")
        cleaned.append(value)
    return cleaned


def save_training_curves(history: dict[str, list[float]], output_path: Path) -> None:
    """Save training curves to disk."""
    train_loss = to_float_list(history["train_loss"], "train_loss")
    val_loss = to_float_list(history["val_loss"], "val_loss")
    train_acc = to_float_list(history["train_acc"], "train_acc")
    val_acc = to_float_list(history["val_acc"], "val_acc")

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(train_loss, label="train_loss", marker="o")
    axes[0].plot(val_loss, label="val_loss", marker="o")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(train_acc, label="train_acc", marker="o")
    axes[1].plot(val_acc, label="val_acc", marker="o")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True)

    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    """Run the training pipeline."""
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    config = parse_args()
    set_reproducibility(config.seed)

    device = get_device()
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        cudnn.benchmark = True

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, classes = build_dataloaders(config, device)
    print(f"Number of classes: {len(classes)}")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")

    model = build_model(
        model_name=config.model_name,
        num_classes=len(classes),
        dropout=config.dropout,
        pretrained=config.pretrained,
        freeze_backbone=config.freeze_backbone,
        device=device,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    optimizer, scheduler = build_optimizer(model, config)
    scaler = GradScaler(device.type, enabled=device.type == "cuda")

    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }
    best_val_acc = 0.0
    early_stop_counter = 0

    for epoch in range(config.epochs):
        train_loss, train_acc = run_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            epoch_index=epoch,
            total_epochs=config.epochs,
            split_name="Train",
        )
        val_loss, val_acc = run_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            epoch_index=epoch,
            total_epochs=config.epochs,
            split_name="Val",
        )

        metrics = EpochMetrics(
            train_loss=train_loss,
            train_acc=train_acc,
            val_loss=val_loss,
            val_acc=val_acc,
        )
        history["train_loss"].append(metrics.train_loss)
        history["train_acc"].append(metrics.train_acc)
        history["val_loss"].append(metrics.val_loss)
        history["val_acc"].append(metrics.val_acc)

        scheduler.step(metrics.val_loss)

        is_best = metrics.val_acc > best_val_acc
        if is_best:
            best_val_acc = metrics.val_acc
            early_stop_counter = 0
            save_checkpoint(
                output_dir / "best_model.pth",
                model,
                config,
                classes,
                best_val_acc,
            )
        else:
            early_stop_counter += 1

        print(
            f"Epoch {epoch + 1}/{config.epochs} | "
            f"Train Loss: {metrics.train_loss:.4f} | "
            f"Train Acc: {metrics.train_acc:.4f} | "
            f"Val Loss: {metrics.val_loss:.4f} | "
            f"Val Acc: {metrics.val_acc:.4f}"
            f"{' <-- best model saved' if is_best else ''}"
        )

        if early_stop_counter >= config.patience:
            print("Early stopping triggered.")
            break

    save_json(output_dir / "classes.json", classes)
    save_json(output_dir / "history.json", history)
    save_json(output_dir / "train_config.json", asdict(config))
    save_training_curves(history, output_dir / "training_curves.png")

    print(f"Training finished. Best validation accuracy: {best_val_acc:.4f}")
    print(f"Saved outputs to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
