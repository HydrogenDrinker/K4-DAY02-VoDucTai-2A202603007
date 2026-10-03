"""losses.py - các hàm loss và trộn mẫu (Mixup, CutMix).

Giao diện:
    build_criterion(kind, **kw)                 -> callable(logits, target) -> loss scalar
    class_weights(counts, beta)                 -> tensor trọng số lớp
    mix_batch(x, y, alpha, mode)                -> (x_mixed, (y_a, y_b, lam))
    mixed_loss(criterion, logits, targets)      -> loss scalar
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_criterion(kind: str = "ce", **kw):
    """Trả về hàm loss theo `kind`: "ce", "ls" (label smoothing), "focal", "ce_weighted"."""
    weight = kw.get("weight", None)
    if isinstance(weight, np.ndarray):
        weight = torch.tensor(weight, dtype=torch.float32)

    if kind == "ce":
        return nn.CrossEntropyLoss(weight=weight)
    elif kind == "ls":
        smoothing = kw.get("smoothing", 0.1)
        return LabelSmoothingCE(smoothing=smoothing, weight=weight)
    elif kind == "focal":
        gamma = kw.get("gamma", 2.0)
        alpha = kw.get("alpha", weight)
        return FocalLoss(gamma=gamma, alpha=alpha)
    elif kind == "ce_weighted":
        if weight is None:
            raise ValueError("ce_weighted yêu cầu tham số 'weight'")
        return nn.CrossEntropyLoss(weight=weight)
    else:
        return nn.CrossEntropyLoss(weight=weight)


class LabelSmoothingCE(nn.Module):
    """Cross-entropy với label smoothing: q'(k) = (1 - eps) * 1[k == y] + eps / K (slide trang 56)."""

    def __init__(self, smoothing: float = 0.1, weight: torch.Tensor | None = None):
        super().__init__()
        self.smoothing = smoothing
        self.register_buffer("weight", weight if weight is not None else None)
        self.ce = nn.CrossEntropyLoss(label_smoothing=smoothing, weight=weight)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.ce(logits, target)


class FocalLoss(nn.Module):
    """Focal loss nhiều lớp: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t) (slide trang 57).

    Khi gamma = 0 và alpha = None: chính xác bằng CrossEntropyLoss.
    """

    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        if alpha is not None and not isinstance(alpha, torch.Tensor):
            alpha = torch.tensor(alpha, dtype=torch.float32)
        if alpha is not None:
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_p = F.log_softmax(logits, dim=-1)
        p = torch.exp(log_p)

        target_expanded = target.unsqueeze(1)
        log_pt = log_p.gather(1, target_expanded).squeeze(1)
        pt = p.gather(1, target_expanded).squeeze(1)

        focal_term = (1.0 - pt) ** self.gamma

        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            alpha_t = alpha.gather(0, target)
            loss = -alpha_t * focal_term * log_pt
        else:
            loss = -focal_term * log_pt

        return loss.mean()


def class_weights(counts, beta: float = 0.0) -> torch.Tensor:
    """Trọng số theo lớp từ số ảnh mỗi lớp trong tập TRAIN.

    - beta = 0: trọng số tỉ lệ nghịch với số ảnh (1 / n_c), chuẩn hoá về mean = 1
    - beta > 0: class-balanced theo số mẫu hiệu dụng (Cui et al., slide trang 57)
    """
    counts = np.array(counts, dtype=np.float64)
    c = len(counts)

    if beta <= 0.0:
        # 1 / n_c
        weights = 1.0 / np.maximum(counts, 1.0)
        # Chuẩn hoá về mean = 1
        weights = weights / weights.mean()
    else:
        # Effective number of samples: (1 - beta^n) / (1 - beta)
        effective_num = 1.0 - np.power(beta, counts)
        weights = (1.0 - beta) / np.maximum(effective_num, 1e-8)
        # Chuẩn hoá về tổng = c (mean = 1)
        weights = weights / weights.sum() * c

    return torch.tensor(weights, dtype=torch.float32)


def rand_bbox(size, lam):
    """Tính toạ độ bounding box cho CutMix."""
    W = size[3]
    H = size[2]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # Tâm ngẫu nhiên
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def mix_batch(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0, mode: str = "cutmix"):
    """Trộn một batch ảnh và nhãn (Mixup hoặc CutMix)."""
    if alpha > 0:
        lam = float(np.random.beta(alpha, alpha))
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    y_a = y
    y_b = y[index]

    if mode == "mixup":
        x_mixed = lam * x + (1.0 - lam) * x[index]
    elif mode == "cutmix":
        bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
        x_mixed = x.clone()
        x_mixed[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
        # Điều chỉnh lại lam theo diện tích thực sự
        total_area = x.size(2) * x.size(3)
        cut_area = (bbx2 - bbx1) * (bby2 - bby1)
        lam = 1.0 - float(cut_area) / float(total_area)
    else:
        x_mixed = x
        lam = 1.0

    return x_mixed, (y_a, y_b, lam)


def mixed_loss(criterion, logits: torch.Tensor, targets: tuple) -> torch.Tensor:
    """Loss cho batch đã trộn: lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)."""
    y_a, y_b, lam = targets
    return lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)
