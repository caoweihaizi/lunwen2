"""预测模型训练（Pinball Loss，多策略混合）。"""
from __future__ import annotations

import time
import numpy as np
import torch
from torch.utils.data import DataLoader

from .model import PredictModel, pinball_loss
from .dataset import LinkStateDataset


def train_model(cfg, train_ds, val_ds, log, epochs=20, batch_size=256, lr=1e-3,
                device="mps"):
    model = PredictModel(n_feat=12, hidden=64, n_h=2, h_max=3).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    best_val = float("inf")
    best_state = None
    history = []

    for ep in range(epochs):
        model.train()
        t0 = time.time()
        train_loss = 0.0; n = 0
        for x, future, y in train_loader:
            x, future, y = x.to(device), future.to(device), y.to(device)
            pred = model(x, future)
            loss = pinball_loss(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item() * len(x)
            n += len(x)
        train_loss /= max(n, 1)

        # 验证
        model.eval()
        val_loss = 0.0; nv = 0
        crossing = 0; crossing_tot = 0
        with torch.no_grad():
            for x, future, y in val_loader:
                x, future, y = x.to(device), future.to(device), y.to(device)
                pred = model(x, future)
                val_loss += pinball_loss(pred, y).item() * len(x)
                nv += len(x)
                # crossing rate
                q05, q50, q95 = pred[..., 0], pred[..., 1], pred[..., 2]
                crossing += int(((q05 > q50 + 1e-6) | (q50 > q95 + 1e-6)).sum())
                crossing_tot += pred[..., 0].numel()
        val_loss /= max(nv, 1)
        crossing_rate = crossing / max(crossing_tot, 1)
        dt = time.time() - t0
        log.info(f"  epoch {ep+1}/{epochs}: train {train_loss:.4f} val {val_loss:.4f} "
                 f"crossing {crossing_rate:.4f} {dt:.0f}s")
        history.append({"epoch": ep + 1, "train_loss": train_loss,
                        "val_loss": val_loss, "crossing_rate": crossing_rate})
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    return model, {"best_val_loss": best_val, "history": history,
                   "final_crossing_rate": history[-1]["crossing_rate"] if history else 0}
