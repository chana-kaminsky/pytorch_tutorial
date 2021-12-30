# activation functions are an important feature in nn's
# they apply a nonlinear transformation to the output and 
# basically decide whether a neuron should be activated or not

# without activation functions, all transformations would be linear,
# like a linear regression model
# but network can perform better and do more complex tasks with
# nonlinear transformations

# activation functions typically applied after each layer, so first
# have a normal linear transformation and then apply activation function

# most common activation functions:
# 1. step function
# 2. sigmoid
# 3. TanH (hyperbolic tan)
# 4. ReLU
# 5. Leaky ReLU
# 6. Softmax

# step function:
# if input >= certain number (ex 0), output 1
# if input < that number, output 0
# this is not used in practice

# sigmoid function:
# f(x) = 1 / (1 + e^(-x))
# outputs a probability between 0 and 1
# typically used in the last layer of a binary classfication

# TanH:
# f(x) = (2 / (1 + e^(-2x))) - 1
# basically a scaled sigmoid function that's a little shifted
# outputs a value between -1 and 1
# a good choice for hidden layers

# ReLU:
# most popular choice
# if input is positive, just outputs the input
# if input is negative, outputs 0
# so positive part is just a linear function, and negative part
# is just 0, but together it's a nonlinear function
# if don't know which function to use, then use ReLU for hidden layers

# Leaky ReLU:
# slightly modified/improved ReLU
# if input is positive, also just outputs the input
# but if input is negative, it multiplies it by a small negative value
# this solves the vanishing gradient problem
# in regular ReLU, since negative inputs are 0, it causes gradient
# to be 0 in backpropagation or something, which means that those
# neurons can never be updated
# if notice that weights aren't updating, can try leaky relu

# softmax:
# s(y_i) = e^(y_i) / sum(e^(y_i))
# squashes inputs to be between 0 and 1
# gives probability as output
# good choice for last layer of multiclass classification


import torch
import torch.nn as nn
import torch.nn.functional as F

# two ways to use activation funtions

# option 1: create nn modules, make the functions attributes
class NeuralNetwork1(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNetwork1, self).__init__()

        # define layers
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.sigmoid(out)
        return out

# option 2: use actuvation functions directly in forward pass
class NeuralNetwork2(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNetwork2, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        out = torch.relu(self.linear1(x))
        out = torch.sigmoid(self.linear2(out))
        return out

# both options do the same thing
# all the funcions listed above are available from the nn module
# (except maybe the step one)
nn.Sigmoid
nn.Tanh
nn.ReLU
nn.LeakyReLU
nn.Softmax

# they are also available from torch, or from torch.nn.functional
torch.sigmoid
torch.tanh
torch.relu
# leaky relu is only available from torch.nn.functional
F.leaky_relu
torch.softmax