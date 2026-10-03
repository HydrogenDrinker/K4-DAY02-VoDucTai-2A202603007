"""benchmark.py - đo độ trễ suy luận đúng cách (slide Day 2, trang 73 và 75; GUIDE.md mục 4.1).

Quy tắc đo:
  - warmup: bỏ >= 10 lần chạy đầu
  - đồng bộ GPU: torch.cuda.synchronize() trước và sau mỗi lần đo
  - >= 50 lần đo, báo cáo p50, p95, p99 (không chỉ trung bình)
  - ghi rõ GPU, dtype (FP32/AMP/FP16), batch, độ phân giải, có/không gộp BN, phiên bản torch
"""
from __future__ import annotations

import copy
import time
import numpy as np
import torch
import torch.nn as nn


def bench(fn, warmup: int = 10, iters: int = 100, sync=None) -> dict:
    """Đo thời gian một hàm `fn()` (không tham số), trả về mili-giây."""
    # Warmup
    for _ in range(warmup):
        fn()
    if sync is not None:
        sync()

    times = []
    for _ in range(iters):
        if sync is not None:
            sync()
        t0 = time.perf_counter()
        fn()
        if sync is not None:
            sync()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    times = np.array(times, dtype=np.float64)
    return {
        "p50": float(np.percentile(times, 50)),
        "p95": float(np.percentile(times, 95)),
        "p99": float(np.percentile(times, 99)),
        "mean": float(np.mean(times)),
        "std": float(np.std(times, ddof=1)) if len(times) > 1 else 0.0,
        "n": iters,
    }


def latency_report(model: nn.Module, batch_size: int, img_size: int, dtype: str = "fp32",
                   device: str = "cuda", warmup: int = 10, iters: int = 100) -> dict:
    """Đo độ trễ forward của `model` với đầu vào ngẫu nhiên (batch_size, 3, img_size, img_size)."""
    device_obj = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")
    is_cuda = (device_obj.type == "cuda")

    model_eval = copy.deepcopy(model).to(device_obj).eval()

    if dtype == "fp16" and is_cuda:
        model_eval = model_eval.half()

    x = torch.randn(batch_size, 3, img_size, img_size, device=device_obj)
    if dtype == "fp16" and is_cuda:
        x = x.half()

    sync_fn = torch.cuda.synchronize if is_cuda else None

    if dtype == "amp" and is_cuda:
        def forward_fn():
            with torch.inference_mode():
                with torch.cuda.amp.autocast():
                    model_eval(x)
    else:
        def forward_fn():
            with torch.inference_mode():
                model_eval(x)

    stats = bench(forward_fn, warmup=warmup, iters=iters, sync=sync_fn)

    gpu_name = torch.cuda.get_device_name(0) if is_cuda else "CPU"
    p50_ms = max(stats["p50"], 1e-4)
    images_per_s = float(batch_size / (p50_ms / 1000.0))

    return {
        "gpu": gpu_name,
        "dtype": dtype,
        "batch": batch_size,
        "img_size": img_size,
        "p50": round(stats["p50"], 3),
        "p95": round(stats["p95"], 3),
        "p99": round(stats["p99"], 3),
        "mean": round(stats["mean"], 3),
        "images_per_s": round(images_per_s, 1),
        "torch": torch.__version__,
    }


def tta_latency(model: nn.Module, k_views: int = 2, batch_size: int = 1,
                img_size: int = 224, dtype: str = "amp", device: str = "cuda") -> dict:
    """Độ trễ của TTA K views: đo thời gian forward K views và so sánh với K * p50 của 1 view."""
    base_rep = latency_report(model, batch_size=batch_size, img_size=img_size, dtype=dtype, device=device)

    device_obj = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")
    is_cuda = (device_obj.type == "cuda")
    sync_fn = torch.cuda.synchronize if is_cuda else None
    model_eval = copy.deepcopy(model).to(device_obj).eval()

    x = torch.randn(batch_size, 3, img_size, img_size, device=device_obj)

    def tta_forward():
        with torch.inference_mode():
            with torch.cuda.amp.autocast(enabled=(dtype == "amp" and is_cuda)):
                # Lần lượt chạy k_views
                for _ in range(k_views):
                    model_eval(x)

    stats = bench(tta_forward, warmup=10, iters=60, sync=sync_fn)
    expected_k_p50 = base_rep["p50"] * k_views

    return {
        "k_views": k_views,
        "actual_p50": round(stats["p50"], 3),
        "actual_p95": round(stats["p95"], 3),
        "actual_p99": round(stats["p99"], 3),
        "single_view_p50": base_rep["p50"],
        "expected_k_p50": round(expected_k_p50, 3),
        "overhead_ratio": round(stats["p50"] / max(expected_k_p50, 1e-4), 2),
    }
