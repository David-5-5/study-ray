import torch
import math
from ray.train.torch import TorchTrainer, TorchConfig, get_device
from ray.train import ScalingConfig

def train_loop_per_worker():
    device = get_device()

    # 多项式拟合 sin(x)，沿用你之前autograd练习代码
    x = torch.linspace(-math.pi, math.pi, 2000, device=device)
    y = torch.sin(x)

    a = torch.randn((), device=device, requires_grad=True)
    b = torch.randn((), device=device, requires_grad=True)
    c = torch.randn((), device=device, requires_grad=True)
    d = torch.randn((), device=device, requires_grad=True)

    learning_rate = 1e-8
    for t in range(2000):
        y_pred = a + b*x + c*x**2 + d*x**3
        loss = (y_pred - y).pow(2).sum()

        loss.backward()
        # 手动梯度更新
        with torch.no_grad():
            a -= learning_rate * a.grad
            b -= learning_rate * b.grad
            c -= learning_rate * c.grad
            d -= learning_rate * d.grad
            # 清空梯度
            a.grad = None
            b.grad = None
            c.grad = None
            d.grad = None

        if t % 200 == 0:
            print(f"t={t}, loss={loss.item():.4f}, a={a.item():.4f},b={b.item():.4f},c={c.item():.4f},d={d.item():.4f}")

if __name__ == "__main__":
    trainer = TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        torch_config=TorchConfig(backend="gloo"),
        scaling_config=ScalingConfig(
            num_workers=2,
            use_gpu=False
        )
    )
    result = trainer.fit()
