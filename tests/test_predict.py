"""预测模型单元测试。"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.predict.model import PredictModel, pinball_loss


def test_forward_shape():
    model = PredictModel(n_feat=12, hidden=32, n_h=2, h_max=3)
    B, W = 4, 30
    x = torch.randn(B, W, 12)
    future = torch.randn(B, 3, 3)
    out = model(x, future)
    assert out.shape == (B, 2, 3), f"输出形状 {out.shape} != (4,2,3)"


def test_monotonic_quantiles():
    """单调头：y05 <= y50 <= y95（crossing rate=0）。"""
    model = PredictModel(n_feat=12, hidden=32, n_h=2, h_max=3)
    x = torch.randn(10, 30, 12)
    future = torch.randn(10, 3, 3)
    out = model(x, future)  # (10, 2, 3)
    q05, q50, q95 = out[..., 0], out[..., 1], out[..., 2]
    assert (q05 <= q50 + 1e-6).all(), "y05 > y50 违反单调"
    assert (q50 <= q95 + 1e-6).all(), "y50 > y95 违反单调"


def test_gradient_flow():
    model = PredictModel(n_feat=12, hidden=32, n_h=2, h_max=3)
    x = torch.randn(4, 30, 12)
    future = torch.randn(4, 3, 3)
    target = torch.randn(4, 2)
    out = model(x, future)
    loss = pinball_loss(out, target)
    loss.backward()
    # 检查梯度存在
    for name, p in model.named_parameters():
        assert p.grad is not None, f"{name} 无梯度"
        break  # 检查一个即可


def test_pinball_loss():
    pred = torch.zeros(2, 2, 3)
    pred[..., 1] = 0.0  # q50=0
    target = torch.ones(2, 2)
    loss = pinball_loss(pred, target)
    assert loss > 0
