# four steps:
# 1. prediction
# 2. gradients computation
# 3. loss computation
# 4. parameter updates

# first we can do all these steps manually, 
# then can learn to use builtin pytorch stuff

# we have to know how algorithms work so we can design
# the model, but then pytorch does most of the work

# this video does step one and two
# we shld know how linear regression and gradient descent works

import numpy as np

# using numpy because doing it fm scratch
# training it to output 2 * input ???

# f = w * x
# f = 2 * x

x = np.array([1, 2, 3, 4], dtype=np.float32)
y = np.array([2, 4, 6, 8], dtype=np.float32)

# f = 3 * x
# y = np.array([3, 6, 9, 12])

# f = 7 + x -- # doesn't work!
# y = np.array([8, 9, 10, 11])

w = 0.0

# calculate model prediction
def forward(x):
    return w * x

# calculate loss
# loss = mean squared error (MSE)
def loss(y, y_predicted):
    return ((y_predicted - y)**2).mean()

# calculate gradient
# MSE = 1/N * (w*x - y)**2
# (dJ/dw) = 1/N * 2x * (w*x - y)
def gradient(x, y, y_predicted):
    # dot product in np is like matrix multiplication ?
    ## [2,4,6,8] * [-2,-4,-6,-8]
    # print('\t', 2*x, (y_predicted-y))
    # print('\t', np.dot(2*x, y_predicted-y))
    return np.dot(2*x, y_predicted-y).mean()

userInput = 5
print(f'Prediction before training: f({userInput}) = {forward(userInput):.3f}')

# Training
learning_rate = 0.01
n_iterations = 30

for epoch in range(n_iterations):
    # prediction = forward pass
    y_predicted = forward(x)
    
    # loss
    l = loss(y, y_predicted)

    # gradients
    dw = gradient(x, y, y_predicted)
    
    # update weights
    w -= learning_rate * dw # update formula

    if epoch % 3 == 0:
        # print('y_predicted:', y_predicted)
        # print('dw:', dw)
        print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')


print(f'Prediction after training: f({userInput}) = {forward(userInput):.3f}')

'''
### using pytorch
import torch


x = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)

w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

def forwardTorch(x):
    return w * x


def lossTorch(y, y_predicted):
    return ((y_predicted - y) ** 2).mean()

userInput = 5
print(f'Prediction before training: f({userInput}) = {forward(userInput):.3f}')

learning_rate = 0.01
n_iterations = 100

for epoch in range(n_iterations):
    # prediction = forward pass
    y_predicted = forwardTorch(x)

    # loss
    l = loss(y, y_predicted)

    # gradients = backward pass
    l.backward() # calculates dl/dw

    # update weights
    with torch.no_grad():
        w -= learning_rate * w.grad
    
    # zero gradients, because backward() accumulates gradients
    w.grad.zero_()

    if epoch % 10 == 0:
        print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')

print(f'Prediction after training: f({userInput}) = {forward(userInput):.3f}')
'''