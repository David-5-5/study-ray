import torch

dtype = torch.float
device = torch.accelerator.current_accelerator().type() if torch.accelerator.is_available() else "cpu"
print(f'Using {device} device')
torch.set_default_device(device)

# Create Tensor to hold input and outputs.
# By default, requires_grad=False, which indicates that we do not need to
# compute gradient with respect to these Tensors during the backward pass.
x = torch.linspace(-1, 1, 2000, dtype=dtype)

# A Taylor expansion would be 1 + x + (1/2) x**2 + (1/3!) x**3 + ...
y = torch.exp(x)

a = torch.randn((), dtype=dtype, requires_grad=True)
b = torch.randn((), dtype=dtype, requires_grad=True)
c = torch.randn((), dtype=dtype, requires_grad=True)
d = torch.randn((), dtype=dtype, requires_grad=True)

initial_loss = 1.
learning_rate = 1e-5

for t in range(5000):
    # Forward pass: compute predicted y using operations on Tensors
    y_pred = a + b * x + c * x ** 2 + d * x ** 3

    # Compute and print loss using operations on Tensors.
    # Now loss is a Tensor of sharp(1, )
    # loss.item() gets the scalar value held in the loss.
    loss = (y_pred - y).pow(2).sum()

    # Calculate initial loss, so we can report loss relative to it
    if t == 0:
        initial_loss = loss.item()
    
    if t % 100 == 99:
        print(
            f'Iteration t = {t:4d} loss(t)/loss(0) = {round(loss.item()/initial_loss, 6):10.6f} '
            f'a = {a.item():10.6f} b = {b.item():10.6f} c = {c.item():10.6f} d={d.item():10.6f}'
        )
    
    # Use autograd to compute the backward pass. This call will compute the
    # gradient of loss with respect to all Tensors with requires_grad=True.
    # After this call a.grad, b.grad. c.grad and d.grad will be Tensors holding
    # the gradient of the loss with respect to a, b, c, d respectively.
    loss.backward()

    # Manually update weights using gradient descent. Wrap in torch.no_grad()
    # because weights have requires_grad=True, but we don't need to track this
    # in autograd.
    with torch.no_grad():
        a -= learning_rate * a.grad
        b -= learning_rate * b.grad
        c -= learning_rate * c.grad
        d -= learning_rate * d.grad

        a.grad = None
        b.grad = None
        c.grad = None
        d.grad = None

print(f'Result: y ={a.item()} + {b.item()}x + {c.item()} x^2 + {d.item()} x^3')