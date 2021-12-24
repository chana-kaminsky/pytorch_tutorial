# autograd package calculates gradients

import torch

# x = torch.randn(3, requires_grad=True)
# print(x)

# calculate gradients of func with respect to x,
# so need to do requires_grad=True

# y = x+2

# first do forward pass and calculate output y
# since requires_grad=True, pytorch will create
# and store func that is used later in backpropagation

# y has an attribute grad_fn which points to a gradient func
# in this case it calculates the gradient of x with respect to y (dx/dy)
# here grad_fn = AddBackward0 bc will be used in backpropagation

# print(y)

# z = y*y*2
# print(z)

# y's grad_fn was called ADD bc was adding, 
# z's is called MUL bc multiplying, etc

# z = z.mean()
# print(z)

# to calculate gradient of z with respect to x:
# z.backward() # dz/dx
# print(x.grad) # x.grad is now a tensor with gradients of x

# if requires_grad is not specified as True will give error

# something abt needs to be multiplied by vector of same size
# if didn't do z.mean(), wld give error bc it's not a scalar value

# so if z is not a scalar value, can do:
# v = torch.tensor([0.1, 1.0, 0.001], dtype=torch.float32)
# # then need to pass v (the vector) to backward function
# z.backward(v)
# print(x.grad)

# three ways to prevent pytorch fm tracking gradients:
# - x.requires_grad_(False)
# - x.detatch()
# - with torch.no_grad()

# x.requires_grad_(False) # the trailing underscore modifies x
# print(x) # now requires_grad is false

# y = x.detach()
# print(y)
# now y is tensor with same vals as x but doesn't require grads

# with torch.no_grad():
#     y = x + 2
#     print(y)
    # here y doesn't have grad func

# now comment out lines 5 and 6
weights = torch.ones(4, requires_grad=True)



# when call backward func, gradient for tensor
# is accumulated into .grad attribute

for epoch in range(3):
    model_output = (weights*3).sum() # dummy operation

    model_output.backward()
    print(weights.grad)

    # each iteration, the backward func 
    # accumulates the values in the .grad attribute
    # weights are getting summed, so gradients clearly incorrect

    # so in the loop need to do 
    weights.grad.zero_()
   # now weights are same each interation

# ex when using pytorch builtin optimizer
# optimizer = torch.optim.SGD(weights, lr=0.01)
# optimizer.step()
# optimizer.zero_grad() # this does same as in line 83