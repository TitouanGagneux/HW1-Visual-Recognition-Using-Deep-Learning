"""Inference script for VRDL HW1 image classification.

This script loads a trained checkpoint and generates a ``prediction.csv`` file
for the test set.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import nn
from torch.amp import autocast
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms
from tqdm import tqdm


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
SUPPORTED_MODELS = {"resnet18", "resnet50"}
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


class InferenceError(RuntimeError):
    """Raised when inference cannot proceed."""


class TestDataset(Dataset[tuple[torch.Tensor, str]]):
    """Custom dataset for unlabeled test images."""

    def __init__(self, image_dir: str, transform: transforms.Compose) -> None:
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.image_names = sorted(
            file_name
            for file_name in os.listdir(self.image_dir)
            if file_name.lower().endswith(IMAGE_EXTENSIONS)
        )
        if not self.image_names:
            raise InferenceError(
                f"No images found in {self.image_dir}. "
                "Check the path and image extensions."
            )

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        image_name = self.image_names[index]
        image_path = self.image_dir / image_name
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image)
        return image_tensor, image_name


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run inference for HW1.")
    parser.add_argument("--test-dir", default="data/test")
    parser.add_argument("--checkpoint", default="outputs/best_model.pth")
    parser.add_argument("--output-csv", default="prediction.csv")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--model-name", choices=sorted(SUPPORTED_MODELS), default=None)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--classes-json", default=None)
    parser.add_argument("--train-dir", default="data/train")
    parser.add_argument("--dropout", type=float, default=None)
    return parser.parse_args()


def get_device() -> torch.device:
    """Return the active device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_test_transform(image_size: int) -> transforms.Compose:
    """Build the test-time transform."""
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> dict[str, Any]:
    """Load a saved checkpoint."""
    if not checkpoint_path.exists():
        raise InferenceError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise InferenceError("Checkpoint format is invalid.")
    return checkpoint


def resolve_classes(
    args: argparse.Namespace,
    checkpoint: dict[str, Any],
) -> list[str]:
    """Resolve class names from checkpoint, JSON file, or train directory."""
    classes = checkpoint.get("classes")
    if isinstance(classes, list) and classes:
        return [str(class_name) for class_name in classes]

    if args.classes_json is not None:
        classes_path = Path(args.classes_json)
        if not classes_path.exists():
            raise InferenceError(f"Classes JSON not found: {classes_path}")
        return json.loads(classes_path.read_text(encoding="utf-8"))

    train_dir = Path(args.train_dir)
    if not train_dir.exists():
        raise InferenceError(
            "Class names are missing. Provide --classes-json or a checkpoint "
            "that contains class names."
        )

    train_dataset = datasets.ImageFolder(train_dir)
    if not train_dataset.classes:
        raise InferenceError("No classes found in the training directory.")
    return train_dataset.classes


def build_model(
    model_name: str,
    num_classes: int,
    dropout: float,
    device: torch.device,
) -> nn.Module:
    """Build a ResNet model matching the training configuration."""
    if model_name == "resnet18":
        model = models.resnet18(weights=None)
    elif model_name == "resnet50":
        model = models.resnet50(weights=None)
    else:
        raise InferenceError(f"Unsupported model: {model_name}")

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )
    return model.to(device)


def main() -> None:
    """Run the inference pipeline."""
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    args = parse_args()
    device = get_device()
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    checkpoint_path = Path(args.checkpoint)
    checkpoint = load_checkpoint(checkpoint_path, device)

    classes = resolve_classes(args, checkpoint)
    model_name = args.model_name or checkpoint.get("model_name", "resnet50")
    dropout = args.dropout
    if dropout is None:
        dropout = float(checkpoint.get("dropout", 0.5))

    model = build_model(
        model_name=model_name,
        num_classes=len(classes),
        dropout=dropout,
        device=device,
    )

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    test_dataset = TestDataset(
        image_dir=args.test_dir,
        transform=build_test_transform(args.image_size),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    predictions: list[list[str]] = []
    progress_bar = tqdm(test_loader, desc="Inference", leave=False)

    with torch.no_grad():
        for images, image_names in progress_bar:
            images = images.to(device, non_blocking=True)
            with autocast(device_type=device.type, enabled=device.type == "cuda"):
                outputs = model(images)
                predicted_indices = outputs.argmax(dim=1).cpu().tolist()

            for image_name, predicted_index in zip(image_names, predicted_indices):
                image_stem = Path(image_name).stem
                predicted_label = str(classes[predicted_index])
                predictions.append([image_stem, predicted_label])

    output_csv = Path(args.output_csv)
    with output_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["image_name", "pred_label"])
        writer.writerows(predictions)

    print(f"Prediction complete: {len(predictions)} images")
    print(f"CSV saved to: {output_csv.resolve()}")


if __name__ == "__main__":
    main()
