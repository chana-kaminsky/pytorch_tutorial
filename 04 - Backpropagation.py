# calculating gradients with backpropagation

# chain rule: 
# if a(x) = y and b(y) = z, 
# then (dz/dx) = (dz/dy) * (dy/dx)
# (derivative of outside keeping inside the same
# times the derivative of the inside)

# computational graph
# pytorch makes a graph for each computation we do
# at each node can calculate local gradients, used
# later to compute final gradient
# we know the functions at each node, so easy to 
# compute local gradient, use chain rule
# at end calculate loss function somehow...

# three steps:
# 1. forward pass: compute loss
# 2. compute local gradients
# 3. backward pass: compute (dloss/dweights) using chain rule

# linear regression
# y^ (y hat) means y predicted
# y^ = w * x
# loss = (y^ - y)^2 = ((w*x)-y)^2 |-> the squared error

# ex. x = 1, y = 2, w = 1:
# last node: derivative of s^2 = 2s
# middle node: derivative of (y^ - y) = 1 ??
# first node: derivative of (w*x) = 1x*1 = x ??
# do backward pass and somehow get -2 for final gradient

import torch

x = torch.tensor(1.0)
y = torch.tensor(2.0)
w = torch.tensor(1.0, requires_grad=True)

# forward passs and compute the loss
y_hat = w * x
loss = (y_hat - y)**2

print(loss)

# backward pass
loss.backward() # this is the whole gradient computation
print(w.grad)

# next steps:
### update weights
### next forward and backward pass
### ^ do for a couple iterations