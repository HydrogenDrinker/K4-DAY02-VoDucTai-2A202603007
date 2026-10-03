import sys
sys.path.insert(0, ".")
sys.path.insert(0, "submissions/2A202603007_VoDucTai/code")

import torch
import numpy as np
import dataset
import model
import losses
import benchmark
import train
import eval as ev

print("=== 1. Testing check_split on real data ===")
train_df, val_df, test_df = dataset.load_split("data/labels", fold=0)
stats = dataset.check_split(train_df, val_df, test_df, "data/images")
print("Split stats:", stats["n"])
print("Per-class counts summary:")
for c in range(9):
    info = stats["per_class"][c]
    print(f"  Class {c} ({info['name']}): train={info['train']}, val={info['val']}, test={info['test']}, total={info['total']}")

print("\n=== 2. Testing Focal Loss gamma=0 == CE ===")
ce = torch.nn.CrossEntropyLoss()
fl0 = losses.FocalLoss(gamma=0.0)
logits = torch.randn(16, 9)
targets = torch.randint(0, 9, (16,))
diff = abs(ce(logits, targets).item() - fl0(logits, targets).item())
print(f"Focal loss diff vs CE: {diff:.8e}")
assert diff < 1e-5, f"Focal loss diff too large: {diff}"

print("\n=== 3. Testing Initial Loss ~ 2.197 ===")
m = model.build_model("resnet50", pretrained=False, num_classes=9)
with torch.no_grad():
    x = torch.randn(16, 3, 224, 224)
    y = torch.randint(0, 9, (16,))
    init_loss = ce(m(x), y).item()
print(f"Initial loss: {init_loss:.4f} (expected ~ 2.197)")

print("\n=== 4. Testing Overfitting 1 Batch ===")
opt = torch.optim.Adam(m.parameters(), lr=1e-3)
x_batch = torch.randn(8, 3, 224, 224)
y_batch = torch.randint(0, 9, (8,))
for i in range(80):
    opt.zero_grad()
    loss = ce(m(x_batch), y_batch)
    loss.backward()
    opt.step()
print(f"Overfit final loss: {loss.item():.6f}")
assert loss.item() < 0.05, f"Could not overfit 1 batch, loss={loss.item()}"

print("\n=== 5. Testing Param Groups (3 groups) ===")
groups = model.param_groups(m, lr_backbone=1e-4, lr_head=1e-3, weight_decay=0.05)
print(f"Created {len(groups)} param groups:")
for g in groups:
    print(f"  Group '{g['name']}': lr={g['lr']}, weight_decay={g['weight_decay']}, num_params={len(g['params'])}")
assert len(groups) == 3

print("\n=== 6. Testing Freeze Backbone ===")
m_frozen = model.build_model("resnet50", pretrained=False, num_classes=9, init="frozen")
trainable = [p for p in m_frozen.parameters() if p.requires_grad]
frozen = [p for p in m_frozen.parameters() if not p.requires_grad]
print(f"Frozen model: {len(trainable)} trainable tensors, {len(frozen)} frozen tensors")
assert len(trainable) > 0 and len(frozen) > 0

print("\n=== 7. Testing Benchmark Latency ===")
rep = benchmark.latency_report(m, batch_size=1, img_size=224, dtype="fp32", device="cuda" if torch.cuda.is_available() else "cpu", warmup=5, iters=20)
print("Benchmark report:", rep)

print("\n>>> ALL PIPELINE SANITY CHECKS PASSED SUCCESSFULLY! <<<")
