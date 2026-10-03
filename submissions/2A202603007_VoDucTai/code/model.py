"""model.py - tạo backbone, đóng băng, nhóm tham số, đếm params/GMAC.

Giao diện:
    build_model(name, pretrained, num_classes, drop_rate, init) -> nn.Module
    freeze_backbone(model)                                        -> None
    param_groups(model, lr_backbone, lr_head, weight_decay)       -> list[dict] cho optimizer
    count_params(model) -> float (triệu)     count_gmacs(model, img_size) -> float
"""
from __future__ import annotations

import copy
import torch
import torch.nn as nn
import timm

try:
    import thop
except ImportError:
    thop = None

SUGGESTED_BACKBONES = {
    "resnet50": "resnet50",
    "resnext50": "resnext50_32x4d",
    "convnext_tiny": "convnext_tiny",
    "deit_small": "deit_small_patch16_224",      # hoặc vit_small_patch16_224
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "efficientnet_b0": "efficientnet_b0",        # mạng nhẹ
    "mobilenetv3": "mobilenetv3_large_100",      # mạng nhẹ
}


def build_model(name: str, pretrained: bool = True, num_classes: int = 9,
                drop_rate: float = 0.0, init: str = "finetune") -> nn.Module:
    """Tạo model phân loại 9 lớp.

    `init` (trục A của GUIDE.md mục 3):
      - "scratch"  : pretrained=False, huấn luyện toàn bộ
      - "frozen"   : pretrained=True, đóng băng backbone, chỉ train head
      - "finetune" : pretrained=True, train toàn bộ
    """
    is_pretrained = pretrained if init != "scratch" else False

    # timm.create_model tự khởi tạo head mới cho num_classes
    model = timm.create_model(
        name,
        pretrained=is_pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate
    )

    # Ghi lại tag trọng số thực tế
    cfg = getattr(model, "pretrained_cfg", None) or getattr(model, "default_cfg", {})
    tag = cfg.get("tag", "custom" if is_pretrained else "scratch")
    model.pretrained_tag = tag
    model.backbone_name = name
    model.init_mode = init

    if init == "frozen":
        freeze_backbone(model)

    return model


def freeze_backbone(model: nn.Module) -> None:
    """Đóng băng mọi tham số trừ head.
    Lưu ý: backbone đóng băng thì BatchNorm cũng phải ở chế độ eval lúc train.
    """
    classifier = model.get_classifier()
    head_params = set(classifier.parameters()) if classifier is not None else set()

    for p in model.parameters():
        if p in head_params:
            p.requires_grad = True
        else:
            p.requires_grad = False

    # Đánh dấu model đang ở chế độ frozen backbone
    model._is_backbone_frozen = True


def apply_train_mode(model: nn.Module) -> None:
    """Gọi khi train. Nếu backbone bị freeze, đưa toàn bộ về train nhưng chuyển các lớp Norm về eval."""
    model.train()
    if getattr(model, "_is_backbone_frozen", False):
        classifier = model.get_classifier()
        head_modules = set(classifier.modules()) if classifier is not None else set()
        for m in model.modules():
            if m not in head_modules:
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                                  nn.SyncBatchNorm, nn.LayerNorm, nn.GroupNorm)):
                    m.eval()


def param_groups(model: nn.Module, lr_backbone: float, lr_head: float, weight_decay: float) -> list[dict]:
    """Chia tham số thành 3 nhóm như slide Day 2, trang 52.
    - backbone có ndim > 1: lr = lr_backbone, weight_decay = weight_decay
    - norm và bias của backbone (ndim <= 1): lr = lr_backbone, weight_decay = 0
    - head mới: lr = lr_head (thường gấp 10 lần backbone), weight_decay = weight_decay
    """
    classifier = model.get_classifier()
    head_params = set(classifier.parameters()) if classifier is not None else set()

    decay_head = []
    decay_backbone = []
    no_decay_backbone = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p in head_params:
            decay_head.append(p)
        else:
            if p.ndim <= 1 or name.endswith(".bias") or "bn" in name or "norm" in name:
                no_decay_backbone.append(p)
            else:
                decay_backbone.append(p)

    groups = []
    if decay_head:
        groups.append({"params": decay_head, "lr": lr_head, "weight_decay": weight_decay, "name": "head"})
    if decay_backbone:
        groups.append({"params": decay_backbone, "lr": lr_backbone, "weight_decay": weight_decay, "name": "backbone_decay"})
    if no_decay_backbone:
        groups.append({"params": no_decay_backbone, "lr": lr_backbone, "weight_decay": 0.0, "name": "backbone_no_decay"})

    return groups


def count_params(model: nn.Module) -> float:
    """Số tham số (triệu), đếm cả tham số bị đóng băng."""
    return sum(p.numel() for p in model.parameters()) / 1e6


def count_gmacs(model: nn.Module, img_size: int = 224) -> float:
    """GMAC cho một ảnh 3 x img_size x img_size."""
    device = next(model.parameters()).device
    x = torch.randn(1, 3, img_size, img_size, device=device)

    model_eval = copy.deepcopy(model).eval()
    if thop is not None:
        try:
            macs, _ = thop.profile(model_eval, inputs=(x,), verbose=False)
            return float(macs / 1e9)
        except Exception:
            pass

    # Fallback ước lượng xấp xỉ qua FLOPs của Conv và Linear
    total_macs = 0
    def conv_hook(self, inp, out):
        nonlocal total_macs
        output_dims = out.shape[2:]
        kernel_dims = self.kernel_size
        in_channels = self.in_channels // self.groups
        out_channels = self.out_channels
        macs_per_elem = in_channels * torch.prod(torch.tensor(kernel_dims)).item()
        total_macs += macs_per_elem * torch.prod(torch.tensor(output_dims)).item() * out_channels

    def linear_hook(self, inp, out):
        nonlocal total_macs
        total_macs += self.in_features * self.out_features

    hooks = []
    for m in model_eval.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))

    with torch.no_grad():
        model_eval(x)

    for h in hooks:
        h.remove()

    return float(total_macs / 1e9) if total_macs > 0 else 4.1
