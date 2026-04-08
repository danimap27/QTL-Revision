"""Unified dataset loading for QTL experiments."""

import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _get_transforms(image_size=224):
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_transform, eval_transform


def _resolve_dataset_path(name, config_path=None):
    """Try multiple candidate paths to find the dataset directory."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)  # qtl_experiments/
    repo_dir = os.path.dirname(base_dir)     # QTL_Revision/

    candidates = []
    if config_path:
        candidates.append(config_path)
    candidates += [
        os.path.join(base_dir, "data", "datasets", name),
        os.path.join(repo_dir, "datasets", name),
        os.path.join(repo_dir, "Resultados", "datasets", name),
        os.path.join(os.getcwd(), "datasets", name),
        os.path.join(os.getcwd(), "data", "datasets", name),
    ]
    for c in candidates:
        if os.path.isdir(os.path.join(c, "train")) and os.path.isdir(os.path.join(c, "test")):
            return c
    raise FileNotFoundError(
        f"Dataset '{name}' not found. Checked: {candidates}"
    )


def load_dataset(dataset_cfg, training_cfg, seed):
    """Load train/val/test data loaders for a dataset.

    Args:
        dataset_cfg: dict with keys name, path, num_classes, image_size
        training_cfg: dict with key batch_size
        seed: int for reproducible train/val split

    Returns:
        train_loader, val_loader, test_loader, num_classes
    """
    name = dataset_cfg["name"]
    image_size = dataset_cfg.get("image_size", 224)
    config_path = dataset_cfg.get("path")
    batch_size = training_cfg["batch_size"]

    data_dir = _resolve_dataset_path(name, config_path)
    train_tf, eval_tf = _get_transforms(image_size)

    full_train = datasets.ImageFolder(os.path.join(data_dir, "train"), train_tf)
    test_ds = datasets.ImageFolder(os.path.join(data_dir, "test"), eval_tf)

    num_classes = len(full_train.classes)

    # Deterministic 80/20 train/val split
    train_size = int(0.8 * len(full_train))
    val_size = len(full_train) - train_size
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full_train, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=0, pin_memory=True)

    return train_loader, val_loader, test_loader, num_classes
