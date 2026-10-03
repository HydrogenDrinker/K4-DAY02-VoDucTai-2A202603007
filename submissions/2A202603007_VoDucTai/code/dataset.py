"""dataset.py - đọc DeepWeeds, kiểm tra chia dữ liệu, transform, DataLoader.

Giao diện bắt buộc:
    load_split(labels_dir, fold=0)            -> (train_df, val_df, test_df)
    check_split(train_df, val_df, test_df, images_dir) -> dict  (số liệu để ghi báo cáo)
    build_transforms(train, img_size, aug)    -> torchvision transform
    DeepWeedsDataset[i]                       -> (image_tensor, label:int, filename:str)
    make_loader(df, images_dir, transform, batch_size, train, sampler, num_workers)
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T

NUM_CLASSES = 9
# Thứ tự lớp theo cột `Label` của labels.csv (0 = Chinee Apple ... 7 = Snake Weed, 8 = Negatives).
CLASS_NAMES = [
    "Chinee Apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly Acacia",
    "Rubber Vine", "Siam Weed", "Snake Weed", "Negatives",
]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_split(labels_dir: str | Path, fold: int = 0) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Đọc train_subset{fold}.csv, val_subset{fold}.csv, test_subset{fold}.csv (S1).
    Mỗi file có cột `Filename, Label, Species`. Trả về ba DataFrame.
    """
    labels_dir = Path(labels_dir)
    train_path = labels_dir / f"train_subset{fold}.csv"
    val_path = labels_dir / f"val_subset{fold}.csv"
    test_path = labels_dir / f"test_subset{fold}.csv"

    if not train_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file split: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file split: {val_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file split: {test_path}")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    for df, name in [(train_df, "train"), (val_df, "val"), (test_df, "test")]:
        for col in ["Filename", "Label"]:
            if col not in df.columns:
                raise ValueError(f"Thiếu cột {col} trong {name} split")
        df["Label"] = df["Label"].astype(int)

    return train_df, val_df, test_df


def check_split(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                images_dir: str | Path) -> dict:
    """Kiểm tra bắt buộc trước khi train (README.md, mục 2.1). In ra và trả về dict số liệu.
    1. số ảnh mỗi tập và số ảnh mỗi lớp trong từng tập (kỳ vọng xấp xỉ 60/20/20)
    2. giao của từng cặp tập theo Filename phải RỖNG (train∩val, train∩test, val∩test)
    3. hợp ba tập phải bằng đúng 17.509 ảnh
    4. mọi Filename đều tồn tại trong `images_dir`
    """
    images_dir = Path(images_dir)
    n_train = len(train_df)
    n_val = len(val_df)
    n_test = len(test_df)
    total_imgs = n_train + n_val + n_test

    # 1. Kiểm tra tổng số ảnh
    assert total_imgs == 17509, f"Tổng số ảnh phải là 17.509, nhưng là {total_imgs}"

    # 2. Kiểm tra giao rỗng
    train_files = set(train_df["Filename"])
    val_files = set(val_df["Filename"])
    test_files = set(test_df["Filename"])

    train_val_overlap = train_files & val_files
    train_test_overlap = train_files & test_files
    val_test_overlap = val_files & test_files

    assert len(train_val_overlap) == 0, f"train và val bị trùng {len(train_val_overlap)} ảnh"
    assert len(train_test_overlap) == 0, f"train và test bị trùng {len(train_test_overlap)} ảnh"
    assert len(val_test_overlap) == 0, f"val và test bị trùng {len(val_test_overlap)} ảnh"

    # 3. Kiểm tra hợp bằng 17.509
    union_files = train_files | val_files | test_files
    assert len(union_files) == 17509, f"Hợp ba tập là {len(union_files)} != 17509"

    # 4. Kiểm tra file tồn tại trong images_dir
    # Kiểm tra nhanh: kiểm tra mẫu hoặc toàn bộ
    for fn in union_files:
        p = images_dir / fn
        if not p.exists():
            raise FileNotFoundError(f"Ảnh {fn} không tồn tại trong {images_dir}")

    # Thống kê theo lớp
    train_counts = train_df["Label"].value_counts().to_dict()
    val_counts = val_df["Label"].value_counts().to_dict()
    test_counts = test_df["Label"].value_counts().to_dict()

    per_class = {}
    for c in range(NUM_CLASSES):
        c_name = CLASS_NAMES[c] if c < len(CLASS_NAMES) else f"Class_{c}"
        per_class[c] = {
            "name": c_name,
            "train": train_counts.get(c, 0),
            "val": val_counts.get(c, 0),
            "test": test_counts.get(c, 0),
            "total": train_counts.get(c, 0) + val_counts.get(c, 0) + test_counts.get(c, 0)
        }

    stats = {
        "n": {"train": n_train, "val": n_val, "test": n_test, "total": total_imgs},
        "per_class": per_class,
        "overlap": {
            "train_val": len(train_val_overlap),
            "train_test": len(train_test_overlap),
            "val_test": len(val_test_overlap)
        }
    }
    return stats


def build_transforms(train: bool, img_size: int = 224, aug: str = "basic"):
    """Tạo transform. `aug` chọn mức augmentation: "basic", "color", "trivial", "randaug"."""
    normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    if train:
        if aug == "basic":
            return T.Compose([
                T.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                normalize,
            ])
        elif aug == "color":
            return T.Compose([
                T.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                T.RandomHorizontalFlip(),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                T.ToTensor(),
                normalize,
            ])
        elif aug == "trivial":
            return T.Compose([
                T.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                T.RandomHorizontalFlip(),
                T.TrivialAugmentWide(),
                T.ToTensor(),
                normalize,
            ])
        elif aug == "randaug":
            return T.Compose([
                T.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                T.RandomHorizontalFlip(),
                T.RandAugment(num_ops=2, magnitude=9),
                T.ToTensor(),
                normalize,
            ])
        else:
            return T.Compose([
                T.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                normalize,
            ])
    else:
        # Validation / Test: không dùng ngẫu nhiên
        if img_size == 256:
            return T.Compose([
                T.Resize((256, 256)),
                T.ToTensor(),
                normalize,
            ])
        else:
            return T.Compose([
                T.Resize(256),
                T.CenterCrop(img_size),
                T.ToTensor(),
                normalize,
            ])


class DeepWeedsDataset(Dataset):
    """Dataset đọc ảnh từ `images_dir` theo DataFrame (Filename, Label)."""

    def __init__(self, df: pd.DataFrame, images_dir: str | Path, transform=None):
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.filenames = self.df["Filename"].tolist()
        self.labels = self.df["Label"].to_numpy(dtype=np.int64)

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, int, str]:
        fn = self.filenames[i]
        path = self.images_dir / fn
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        label = int(self.labels[i])
        return img, label, fn


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)


def make_loader(df: pd.DataFrame, images_dir: str | Path, transform, batch_size: int,
                train: bool, sampler: str | None = None, num_workers: int = 2) -> DataLoader:
    """Tạo DataLoader."""
    dataset = DeepWeedsDataset(df, images_dir, transform)

    sampler_obj = None
    shuffle = False

    if train:
        if sampler == "balanced":
            counts = np.bincount(df["Label"].to_numpy(dtype=np.int64), minlength=NUM_CLASSES)
            weights_per_class = 1.0 / np.maximum(counts, 1)
            sample_weights = weights_per_class[df["Label"].to_numpy(dtype=np.int64)]
            sampler_obj = WeightedRandomSampler(
                weights=torch.as_tensor(sample_weights, dtype=torch.double),
                num_samples=len(sample_weights),
                replacement=True
            )
            shuffle = False
        else:
            shuffle = True
    else:
        shuffle = False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler_obj,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=(train and len(dataset) > batch_size),
        worker_init_fn=seed_worker,
    )
