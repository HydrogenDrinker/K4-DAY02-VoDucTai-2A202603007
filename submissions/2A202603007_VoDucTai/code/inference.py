"""inference.py - các phương pháp suy luận (Bước 3 của GUIDE.md).

Giao diện:
    predict_logits(model, loader, device, view=None) -> (filenames, y_true, logits[N, 9])
    view_identity(x)
    view_hflip(x)
    views_multicrop(x, crop: int)
    views_multiscale(x, sizes)
    aggregate_views(list_of_logits, space)           -> probs[N, 9]
    fit_temperature(val_logits, val_labels)          -> float T
    apply_temperature(logits, T)                     -> probs
    ensemble_probs(list_of_probs)                    -> probs
    fuse_conv_bn(model)                              -> model (BN đã gộp vào conv)
"""
from __future__ import annotations

import copy
import numpy as np
import scipy.optimize
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.fusion import fuse_conv_bn_eval


def predict_logits(model: nn.Module, loader, device: str | torch.device, view=None):
    """Chạy model trên loader và gom logit theo đúng thứ tự file."""
    model.eval()
    all_filenames = []
    all_y_true = []
    all_logits = []

    device = torch.device(device)
    use_amp = torch.cuda.is_available() and device.type == "cuda"

    with torch.inference_mode():
        for batch in loader:
            images, targets, filenames = batch
            if view is not None:
                images = view(images)

            images = images.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(images)

            all_filenames.extend(filenames)
            all_y_true.append(targets.cpu().numpy())
            all_logits.append(logits.float().cpu().numpy())

    y_true = np.concatenate(all_y_true, axis=0) if all_y_true else np.array([])
    logits_arr = np.concatenate(all_logits, axis=0) if all_logits else np.empty((0, 9))

    return all_filenames, y_true, logits_arr


def view_identity(x: torch.Tensor) -> torch.Tensor:
    """Giữ nguyên tensor."""
    return x


def view_hflip(x: torch.Tensor) -> torch.Tensor:
    """Lật ngang batch (N, C, H, W) qua chiều rộng W (slide trang 75)."""
    return torch.flip(x, dims=[-1])


def views_multicrop(x: torch.Tensor, crop: int = 224) -> list[torch.Tensor]:
    """5 crop (4 góc + giữa) kích thước `crop`."""
    H, W = x.shape[-2], x.shape[-1]
    if H < crop or W < crop:
        # Nếu nhỏ hơn crop thì resize
        x = F.interpolate(x, size=(crop, crop), mode="bilinear", align_corners=False)
        return [x]

    tl = x[..., :crop, :crop]
    tr = x[..., :crop, W - crop:]
    bl = x[..., H - crop:, :crop]
    br = x[..., H - crop:, W - crop:]
    ch = (H - crop) // 2
    cw = (W - crop) // 2
    c = x[..., ch:ch + crop, cw:cw + crop]

    return [c, tl, tr, bl, br]


def views_multiscale(x: torch.Tensor, sizes: list[int] = (224, 256, 288)) -> list[torch.Tensor]:
    """Resize batch về từng kích thước trong `sizes`."""
    res = []
    for s in sizes:
        scaled = F.interpolate(x, size=(s, s), mode="bilinear", align_corners=False)
        res.append(scaled)
    return res


def softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Softmax an toàn số học cho numpy."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def aggregate_views(logits_per_view: list[np.ndarray], space: str = "prob") -> np.ndarray:
    """Gộp K lượt chạy của TTA thành một dự đoán (slide trang 62).
      - space="prob": trung bình softmax của từng view
      - space="logit": trung bình logit rồi softmax
    """
    if not logits_per_view:
        raise ValueError("logits_per_view không được rỗng")

    if space == "prob":
        probs = [softmax_np(l, axis=-1) for l in logits_per_view]
        avg_prob = np.mean(probs, axis=0)
        # Chuẩn hoá lại để tổng đúng bằng 1
        return avg_prob / np.sum(avg_prob, axis=-1, keepdims=True)
    elif space == "logit":
        avg_logit = np.mean(logits_per_view, axis=0)
        return softmax_np(avg_logit, axis=-1)
    else:
        raise ValueError(f"space phải là 'prob' hoặc 'logit', nhận được: {space}")


def ensemble_probs(list_of_probs: list[np.ndarray]) -> np.ndarray:
    """Trung bình xác suất của nhiều mô hình (khác backbone hoặc khác seed)."""
    if not list_of_probs:
        raise ValueError("list_of_probs không được rỗng")
    avg = np.mean(list_of_probs, axis=0)
    return avg / np.sum(avg, axis=-1, keepdims=True)


def fit_temperature(val_logits: np.ndarray, val_labels: np.ndarray) -> float:
    """Tìm nhiệt độ T > 0 cực tiểu NLL trên VAL: p = softmax(logit / T) (slide trang 69)."""
    logits_t = torch.as_tensor(val_logits, dtype=torch.float32)
    labels_t = torch.as_tensor(val_labels, dtype=torch.long)

    def nll(log_t):
        T = np.exp(log_t)
        scaled_logits = logits_t / T
        return F.cross_entropy(scaled_logits, labels_t).item()

    res = scipy.optimize.minimize_scalar(nll, bracket=(-2.0, 2.0), method="brent")
    best_T = float(np.exp(res.x))
    # Giới hạn trong khoảng hợp lý [0.05, 10.0]
    return float(np.clip(best_T, 0.05, 10.0))


def apply_temperature(logits: np.ndarray, T: float) -> np.ndarray:
    """Trả về softmax(logits / T)."""
    T = max(float(T), 1e-5)
    return softmax_np(logits / T, axis=-1)


def fuse_conv_bn(model: nn.Module) -> nn.Module:
    """Gộp BatchNorm vào Conv2d liền trước trong mạng CNN (slide trang 71, 75)."""
    model_copy = copy.deepcopy(model).eval()

    def _fuse_recursive(module: nn.Module):
        last_conv_name = None
        last_conv = None

        for name, child in list(module.named_children()):
            if isinstance(child, nn.Conv2d):
                last_conv_name = name
                last_conv = child
            elif isinstance(child, (nn.BatchNorm2d, nn.SyncBatchNorm)) and last_conv is not None:
                # Gộp last_conv và child BN
                try:
                    fused_conv = fuse_conv_bn_eval(last_conv, child)
                    setattr(module, last_conv_name, fused_conv)
                    setattr(module, name, nn.Identity())
                except Exception:
                    pass
                last_conv = None
                last_conv_name = None
            else:
                last_conv = None
                last_conv_name = None
                _fuse_recursive(child)

    _fuse_recursive(model_copy)
    return model_copy
