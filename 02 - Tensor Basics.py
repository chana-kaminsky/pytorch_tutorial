
import torch

# x = torch.empty(2, 2, 3)
# x = torch.rand(2, 2)
# x = torch.zeros(2, 2)
# y = torch.ones(2, 2, dtype=torch.int)
# print(x)
# print(y.dtype)
# print(y.size())

# x = torch.tensor([2.5, 0.1])
# print(x)

# x = torch.rand(2,2)
# y = torch.rand(2,2)
# print(x)
# print(y)

# z = x + y # elementwise addition
# z = torch.add(x,y) # ^ does same thing
# can also do torch.sub, torch.mul, torch.div
# ^ all elementwise operations

# print(z)

# y.add_(x)
# underscores mean in place operations
# this adds x to y and modifies y
# print(y)

# slicing operations

# x = torch.rand(5,3)
# print(x)
# print(x[:, 0]) # [row, col]
# print(x[1, :]) # : means all
# print(x[1, 1]) # tensor at pos 1, 1
# print(x[1,1].item()) # value of item in tensor at pos 1, 1

# reshaping a tensor

# x = torch.rand(4,4)
# print(x)
# x is a 2d tensor
# this makes y a 1d tensor with values of x
# needs to have same number of elements as x (4*4=16)
# y = x.view(16) 
# print(y)

# z = x.view(-1, 8) 
# ^ if write -1, pytorch figures out what other 
# dimension goes with 8 to fit 4*4, ie 2
# print(z)
# print(z.size()) # can check that size = [2,8]

# torch to numpy
import numpy as np

# a = torch.ones(5)
# b = a.numpy()
# print(type(a))
# print(type(b))

# however a and b share memory location, 
# so if modify one it modifies the other

# print('orig a: ', a)
# print('orig b: ', b)
# a.add_(1)
# print(a)
# print(b) # b changes also!

# numpy to torch

# a = np.ones(5)
# print(a)
# b = torch.from_numpy(a) # can specify datatype
# print(b)

# again they share memory location
# to prevent this, do operations on gpu instead of cpu
# can do this using cuda

# if torch.cuda.is_available():
#     device = torch.device('cuda')
#     x = torch.ones(5, device=device)
#     y = torch.ones(5)
#     y = y.to(device)
#     z = x + y # done on gpu!
#     #z.numpy() # gives error! bc can only do on cpu
#     # so do:
#     z = z.to('cpu')


x = torch.ones(5, requires_grad=True)
print(x)

# this tells pytorch that will need to calculate 
# gradient for this tensor later in optimization steps
# so make requires_grad=True whenever have variable
# that want to optimize
