# replacing loss computation and parameter updates
# with builtin pytorch loss and optimizer classes
# also replace model prediction by implementing pytorch model
# this video does steps three and four

# three general steps of training pipeline in pytorch
# 1. design model - input size, output size, forward pass with diff layers
# 2. construct loss and optimizer
# 3. training loop
#   forward pass: compute prediction
#   backward pass: gradients
#   update weights

# pytorch does everything except step one, we need to design the model

import torch
import torch.nn as nn # neural network module

x = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)

# don't need to manually define weights because pytorch knows parameters or s/t

# normally need to design the model ourselves, but this is a very
# simple model, it's just linear, so pytorch has one we can use
# also now need to change x and y to be 2d


numberOfSamples, numberOfFeatures = x.shape
print(numberOfSamples, numberOfFeatures)

inputSize = numberOfFeatures
outputSize = numberOfFeatures

# model = nn.Linear(inputSize, outputSize)

# if want to write a custom linear model:
class LinearRegression(nn.Module):

    def __init__(self, inputSize, outputSize):
        # call super class constructor
        super(LinearRegression, self).__init__() 

        # define layers
        self.lin = nn.Linear(inputSize, outputSize)
    
    def forward(self, x):
        return self.lin(x)

# this should do the same thing as the one on line 35
model = LinearRegression(inputSize, outputSize)

userInput = 5
# make a test tensor because need to pass 
# a tensor into the model function, not an int
x_test = torch.tensor([userInput], dtype=torch.float32)
# don't need to call .item(), it's only because know it's a
# single value and don't want to see the whole tensor I think
print(f'Prediction before training: f({userInput}) = {model(x_test).item():.3f}')

# Training
learning_rate = 0.01
numberOfIterations = 200

# the MSE, which is what we were doing before manually
# it's a callable funtion, so can call it like in line 59
loss = nn.MSELoss() 
# SGD stands for stochastic gradient descent
# takes a parameter to optimize, and a learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(numberOfIterations):
    # prediction = forward pass
    y_predicted = model(x)

    # loss
    l = loss(y, y_predicted)

    # gradients = backward pass
    l.backward() # calculates dl/dw

    # update weights
    optimizer.step() # does an optimization step

    # zero gradients, because backward() accumulates gradients
    optimizer.zero_grad() # to empty the gradients

    if epoch % 10 == 0:
        # b is an optional bias
        [w, b] = model.parameters() # unpacking w and b
        print(f'epoch {epoch+1}: w = {w[0][0].item():.3f}, loss = {l:.8f}')

print(f'Prediction after training: f({userInput}) = {model(x_test).item():.3f}')