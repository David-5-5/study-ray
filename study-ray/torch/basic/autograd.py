# torch.autograd 是 PyTorch 的自动微分引擎，为神经网络训练提供动力。

import torch
from torchvision.models import resnet18, ResNet18_Weights
model = resnet18(weights=ResNet18_Weights.DEFAULT)
data = torch.rand(1, 3, 64, 64)
labels = torch.rand(1, 1000)

prediction = model(data) # forward pass
loss = (prediction - labels).sum()
loss.backward() # backward pass

optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
optim.step()  # gradient descent

#######################################################################
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import math

a = torch.linspace(0.0, 2.0 * math.pi, steps=25, requires_grad=True)
print(a)

b = torch.sin(a)
plt.plot(a.detach(), b.detach())

print(b)

c = 2 * b
print(c)

d = c + 1
print(d)

out = d.sum()
print(out)

print("d:")
print(d.grad_fn)
print(d.grad_fn.next_functions)
print(d.grad_fn.next_functions[0][0].next_functions)
print(d.grad_fn.next_functions[0][0].next_functions[0][0].next_functions)
print(
    d.grad_fn.next_functions[0][0]
    .next_functions[0][0]
    .next_functions[0][0]
    .next_functions
)
print("\nc:")
print(c.grad_fn)
print("\nb:")
print(b.grad_fn)
print("\na:")
print(a.grad_fn)

out.backward()
print(a.grad)
plt.plot(a.detach(), a.grad.detach())