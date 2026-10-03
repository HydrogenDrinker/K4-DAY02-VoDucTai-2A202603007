"""train.py - vòng huấn luyện cho mọi thí nghiệm (B, T, F).

Dùng MỘT hàm `run(cfg)` cho mọi cấu hình (RUBRIC mục H):
đổi thí nghiệm chỉ bằng cách đổi `Config`.

Chạy một thí nghiệm từ dòng lệnh:
    python train.py --set exp_id=B01 backbone=resnet50 seed=0
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

# Import local modules
import dataset
import model as model_lib
import losses as losses_lib

# Import eval functions from repo root
try:
    import eval as ev
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
    import eval as ev


@dataclass
class Config:
    # --- định danh ---
    exp_id: str = "T00"
    seed: int = 0
    fold: int = 0
    # --- mô hình ---
    backbone: str = "resnet50"
    init: str = "finetune"            # scratch | frozen | finetune
    drop_rate: float = 0.0
    # --- dữ liệu / augmentation ---
    img_size: int = 224
    aug: str = "basic"                # basic | color | trivial | randaug
    sampler: str | None = None        # None | balanced
    mix: str | None = None            # None | mixup | cutmix
    mix_alpha: float = 1.0
    # --- loss ---
    loss: str = "ce"                  # ce | ls | focal | ce_weighted
    label_smoothing: float = 0.0
    focal_gamma: float = 2.0
    class_weight_beta: float | None = None
    # --- tối ưu (công thức nền, GUIDE.md mục 1.4) ---
    epochs: int = 12
    batch_size: int = 64
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    ema_decay: float | None = None
    amp: bool = True
    num_workers: int = 2
    # --- đường dẫn ---
    images_dir: str = "data/images"
    labels_dir: str = "data/labels"
    out_dir: str = "runs"             # config.json, history.csv, checkpoint, logit của từng lần chạy
    pred_dir: str = "predictions"     # file dự đoán đúng định dạng eval.py (nộp cùng bài)
    # --- chỉ bật ở Bước 4 (chung kết): ghi predictions trên TEST. Mặc định TẮT (quy tắc S4). ---
    save_test_predictions: bool = False


def run_dir(cfg: Config) -> Path:
    """Thư mục kết quả của một lần chạy: <out_dir>/<exp_id>/seed<k>/ ."""
    return Path(cfg.out_dir) / cfg.exp_id / f"seed{cfg.seed}"


def pred_path(cfg: Config, split: str) -> Path:
    """Đường dẫn chuẩn của file dự đoán: <pred_dir>/<exp_id>_seed<k>_<split>.csv (split = val | test)."""
    return Path(cfg.pred_dir) / f"{cfg.exp_id}_seed{cfg.seed}_{split}.csv"


def set_seed(seed: int) -> None:
    """Cố định mọi nguồn ngẫu nhiên."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def build_optimizer(model: nn.Module, cfg: Config) -> torch.optim.Optimizer:
    """AdamW với 3 nhóm tham số (xem model.param_groups)."""
    groups = model_lib.param_groups(
        model,
        lr_backbone=cfg.lr_backbone,
        lr_head=cfg.lr_head,
        weight_decay=cfg.weight_decay
    )
    return torch.optim.AdamW(groups, eps=1e-8)


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: Config, steps_per_epoch: int):
    """Warmup tuyến tính rồi cosine về ~0 (slide trang 55)."""
    warmup_steps = max(1, int(cfg.warmup_epochs * steps_per_epoch))
    total_steps = max(warmup_steps + 1, cfg.epochs * steps_per_epoch)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(1e-4, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class EMA:
    """Trung bình động trọng số: W_ema <- d * W_ema + (1 - d) * W (slide trang 56)."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}

    def update(self, model: nn.Module) -> None:
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if k in self.shadow:
                    if v.dtype.is_floating_point:
                        self.shadow[k].mul_(self.decay).add_(v, alpha=1.0 - self.decay)
                    else:
                        self.shadow[k].copy_(v)

    def copy_to(self, model: nn.Module) -> None:
        model.load_state_dict(self.shadow)


def train_one_epoch(model: nn.Module, loader, criterion, optimizer, scheduler, scaler,
                    cfg: Config, device: torch.device, ema: EMA | None = None) -> dict:
    """Một epoch huấn luyện."""
    model_lib.apply_train_mode(model)
    total_loss = 0.0
    num_samples = 0

    for batch in loader:
        images, targets, _ = batch
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if cfg.mix and cfg.mix != "none":
            images_mixed, mixed_targets = losses_lib.mix_batch(
                images, targets, alpha=cfg.mix_alpha, mode=cfg.mix
            )
            optimizer.zero_grad()
            with autocast(enabled=cfg.amp and device.type == "cuda"):
                logits = model(images_mixed)
                loss = losses_lib.mixed_loss(criterion, logits, mixed_targets)
        else:
            optimizer.zero_grad()
            with autocast(enabled=cfg.amp and device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, targets)

        if scaler is not None and cfg.amp and device.type == "cuda":
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        if ema is not None:
            ema.update(model)

        b_size = images.size(0)
        total_loss += loss.item() * b_size
        num_samples += b_size

    avg_loss = total_loss / max(1, num_samples)
    curr_lr = optimizer.param_groups[0]["lr"]
    return {"train_loss": avg_loss, "lr": curr_lr}


def evaluate(model: nn.Module, loader, criterion, device: torch.device) -> Tuple[list[str], np.ndarray, np.ndarray, float]:
    """Chạy model trên một loader ở chế độ eval, KHÔNG tính gradient."""
    model.eval()
    all_filenames = []
    all_targets = []
    all_logits = []
    total_loss = 0.0
    num_samples = 0

    with torch.inference_mode():
        for batch in loader:
            images, targets, filenames = batch
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with autocast(enabled=torch.cuda.is_available() and device.type == "cuda"):
                logits = model(images)
                if criterion is not None:
                    loss = criterion(logits, targets)
                    total_loss += loss.item() * images.size(0)

            all_filenames.extend(filenames)
            all_targets.append(targets.cpu().numpy())
            all_logits.append(logits.float().cpu().numpy())
            num_samples += images.size(0)

    y_true = np.concatenate(all_targets, axis=0) if all_targets else np.array([], dtype=int)
    logits_arr = np.concatenate(all_logits, axis=0) if all_logits else np.empty((0, 9))
    avg_loss = total_loss / max(1, num_samples) if num_samples > 0 else 0.0

    return all_filenames, y_true, logits_arr, avg_loss


def plot_curves(history: list[dict], path: str | Path, title: str) -> None:
    """Vẽ đường cong training của một thí nghiệm -> curves/<exp_id>_<mota>.png."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h.get("val_loss", 0.0) for h in history]
    val_macro_f1 = [h.get("val_macro_f1", 0.0) for h in history]
    val_top1 = [h.get("val_top1", 0.0) for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Loss curve
    ax1.plot(epochs, train_loss, label="Train Loss", marker="o", color="#1f77b4")
    ax1.plot(epochs, val_loss, label="Val Loss", marker="s", color="#ff7f0e")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"Loss Curves - {title}")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()

    # Metric curve
    ax2.plot(epochs, val_macro_f1, label="Val Macro-F1", marker="^", color="#2ca02c")
    ax2.plot(epochs, val_top1, label="Val Top-1 Acc (%)", marker="d", color="#d62728")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Score / %")
    ax2.set_title(f"Validation Metrics - {title}")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def run(cfg: Config) -> dict:
    """Huấn luyện một cấu hình và lưu mọi thứ cần thiết. Trả về dict kết quả tóm tắt."""
    start_time = time.time()
    set_seed(cfg.seed)

    r_dir = run_dir(cfg)
    r_dir.mkdir(parents=True, exist_ok=True)
    Path(cfg.pred_dir).mkdir(parents=True, exist_ok=True)
    Path("curves").mkdir(parents=True, exist_ok=True)

    # 1. Ghi config.json
    cfg_dict = dataclasses.asdict(cfg)
    with open(r_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg_dict, f, indent=2)

    # 2. Dữ liệu: load_split và check_split
    train_df, val_df, test_df = dataset.load_split(cfg.labels_dir, fold=cfg.fold)
    split_stats = dataset.check_split(train_df, val_df, test_df, cfg.images_dir)

    train_tf = dataset.build_transforms(train=True, img_size=cfg.img_size, aug=cfg.aug)
    val_tf = dataset.build_transforms(train=False, img_size=cfg.img_size)

    train_loader = dataset.make_loader(
        train_df, cfg.images_dir, train_tf,
        batch_size=cfg.batch_size, train=True,
        sampler=cfg.sampler, num_workers=cfg.num_workers
    )
    val_loader = dataset.make_loader(
        val_df, cfg.images_dir, val_tf,
        batch_size=cfg.batch_size, train=False,
        num_workers=cfg.num_workers
    )

    # 3. Model, Criterion, Optimizer, Scheduler, EMA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_lib.build_model(
        cfg.backbone,
        pretrained=True,
        num_classes=dataset.NUM_CLASSES,
        drop_rate=cfg.drop_rate,
        init=cfg.init
    ).to(device)

    # GMACs & Params count
    params_m = model_lib.count_params(model)
    gmacs = model_lib.count_gmacs(model, img_size=cfg.img_size)

    # Criterion
    loss_kwargs = {}
    if cfg.loss == "ls":
        loss_kwargs["smoothing"] = cfg.label_smoothing or 0.1
    elif cfg.loss == "focal":
        loss_kwargs["gamma"] = cfg.focal_gamma
    elif cfg.loss == "ce_weighted":
        counts = train_df["Label"].value_counts().sort_index().values
        beta = cfg.class_weight_beta if cfg.class_weight_beta is not None else 0.0
        loss_kwargs["weight"] = losses_lib.class_weights(counts, beta=beta).to(device)

    criterion = losses_lib.build_criterion(cfg.loss, **loss_kwargs).to(device)
    eval_criterion = nn.CrossEntropyLoss().to(device)

    optimizer = build_optimizer(model, cfg)
    steps_per_epoch = len(train_loader)
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch)
    scaler = torch.amp.GradScaler("cuda") if (cfg.amp and device.type == "cuda") else None
    ema = EMA(model, decay=cfg.ema_decay) if cfg.ema_decay is not None else None

    # 4. Huấn luyện từng epoch
    history = []
    best_macro_f1 = -1.0
    best_epoch = 0
    best_val_logits = None
    best_val_y = None
    best_val_names = None
    epoch_times = []

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        train_res = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler, cfg, device, ema
        )
        t_epoch = time.time() - t0
        epoch_times.append(t_epoch)

        # Đánh giá bằng EMA nếu có, nếu không thì dùng model hiện tại
        eval_model = model
        if ema is not None:
            eval_model = copy.deepcopy(model)
            ema.copy_to(eval_model)

        val_names, val_y, val_logits, val_loss = evaluate(
            eval_model, val_loader, eval_criterion, device
        )

        # Tính metrics qua eval.py chuẩn
        val_probs = np.exp(val_logits - np.max(val_logits, axis=-1, keepdims=True))
        val_probs /= np.sum(val_probs, axis=-1, keepdims=True)
        val_preds = val_probs.argmax(axis=-1)

        val_metrics = ev.compute_metrics(val_y, val_preds, val_probs)
        macro_f1 = val_metrics["macro_f1"]
        top1_acc = val_metrics["top1"]

        hist_item = {
            "epoch": epoch,
            "train_loss": round(train_res["train_loss"], 4),
            "val_loss": round(val_loss, 4),
            "val_macro_f1": round(macro_f1, 4),
            "val_top1": round(top1_acc, 2),
            "lr": train_res["lr"],
            "time_s": round(t_epoch, 2),
        }
        history.append(hist_item)

        # Chọn checkpoint theo MACRO-F1 VAL (nếu bằng nhau thì ưu tiên epoch sớm hơn)
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_epoch = epoch
            best_val_logits = val_logits
            best_val_y = val_y
            best_val_names = val_names
            # Lưu checkpoint
            save_state = {
                "epoch": epoch,
                "model_state": eval_model.state_dict(),
                "macro_f1": macro_f1,
                "config": cfg_dict,
            }
            torch.save(save_state, r_dir / "best_model.pt")

    # 5. Lưu val predictions và logit tốt nhất
    np.save(r_dir / "val_logits.npy", best_val_logits)
    best_val_probs = np.exp(best_val_logits - np.max(best_val_logits, axis=-1, keepdims=True))
    best_val_probs /= np.sum(best_val_probs, axis=-1, keepdims=True)
    ev.save_predictions(pred_path(cfg, "val"), best_val_names, best_val_y, best_val_probs)

    # 6. Đánh giá TEST nếu save_test_predictions=True (chỉ Bước 4 chung kết)
    test_metrics = None
    if cfg.save_test_predictions:
        test_tf = dataset.build_transforms(train=False, img_size=cfg.img_size)
        test_loader = dataset.make_loader(
            test_df, cfg.images_dir, test_tf,
            batch_size=cfg.batch_size, train=False,
            num_workers=cfg.num_workers
        )
        # Nạp best checkpoint
        best_ckpt = torch.load(r_dir / "best_model.pt", map_location=device)
        model.load_state_dict(best_ckpt["model_state"])
        test_names, test_y, test_logits, test_loss = evaluate(
            model, test_loader, eval_criterion, device
        )
        np.save(r_dir / "test_logits.npy", test_logits)
        test_probs = np.exp(test_logits - np.max(test_logits, axis=-1, keepdims=True))
        test_probs /= np.sum(test_probs, axis=-1, keepdims=True)
        ev.save_predictions(pred_path(cfg, "test"), test_names, test_y, test_probs)

        test_preds = test_probs.argmax(axis=-1)
        test_metrics = ev.compute_metrics(test_y, test_preds, test_probs)

    # 7. Lưu history.csv và vẽ biểu đồ curves/
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(r_dir / "history.csv", index=False)

    curve_path = Path("curves") / f"{cfg.exp_id}_{cfg.backbone}.png"
    plot_curves(history, curve_path, f"{cfg.exp_id} ({cfg.backbone})")

    avg_epoch_time = float(np.mean(epoch_times)) if epoch_times else 0.0
    summary = {
        "exp_id": cfg.exp_id,
        "backbone": cfg.backbone,
        "tag": getattr(model, "pretrained_tag", "unknown"),
        "seed": cfg.seed,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_macro_f1,
        "best_val_top1": hist_df.loc[hist_df["epoch"] == best_epoch, "val_top1"].values[0] if len(hist_df) else 0.0,
        "avg_epoch_time_s": round(avg_epoch_time, 2),
        "params_m": round(params_m, 2),
        "gmacs": round(gmacs, 3),
        "test_macro_f1": test_metrics["macro_f1"] if test_metrics else None,
        "test_top1": test_metrics["top1"] if test_metrics else None,
    }

    with open(r_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def parse_overrides(pairs: list[str]) -> dict:
    """Biến ['seed=1', 'loss=focal', 'ema_decay=none'] thành dict."""
    cfg_fields = {f.name: f.type for f in dataclasses.fields(Config)}
    overrides = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Tham số không hợp lệ (phải có dạng KEY=VAL): {pair}")
        key, val = pair.split("=", 1)
        key = key.strip()
        val = val.strip()

        if key not in cfg_fields:
            raise ValueError(f"Key '{key}' không tồn tại trong Config. Các key hợp lệ: {list(cfg_fields.keys())}")

        if val.lower() == "none":
            overrides[key] = None
        elif val.lower() == "true":
            overrides[key] = True
        elif val.lower() == "false":
            overrides[key] = False
        else:
            try:
                overrides[key] = int(val)
            except ValueError:
                try:
                    overrides[key] = float(val)
                except ValueError:
                    overrides[key] = val
    return overrides


def main() -> None:
    """Điểm vào dòng lệnh: `python train.py --set exp_id=B01 backbone=resnet50 seed=0`."""
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình DeepWeeds")
    parser.add_argument("--set", nargs="*", default=[], help="Ghi đè tham số: key=val")
    args = parser.parse_args()

    overrides = parse_overrides(args.set)
    cfg = Config(**overrides)
    print(f"=== Chạy thí nghiệm {cfg.exp_id} ({cfg.backbone}, seed={cfg.seed}) ===")
    res = run(cfg)
    print("Kết quả:", res)


if __name__ == "__main__":
    main()
