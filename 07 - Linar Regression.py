# implementing linear regression

# three general steps of training pipeline in pytorch
# 1. design model - input size, output size, forward pass with diff layers
# 2. construct loss and optimizer
# 3. training loop
#   forward pass: compute prediction
#   backward pass: gradients
#   update weights

# to install sklearn, use anaconda shell and do:
# conda install -n <env_name> <package>
# conda install -n pytorch scikit-learn
# and for matplotlib do:
# conda install -n pytorch matplotlib 

import torch
import torch.nn as nn
import numpy as np # for data transformations
from sklearn import datasets # to generate regression dataset
import matplotlib.pyplot as plt # to plot data

# print('hello world')

# step 0: prepare data
# generate a regression dataset
# (not sure what the errors here are but seems to work anyway)
x_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1)

# now it's a double, want it to be a float to avoid problems later
x = torch.from_numpy(x_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))

# reshape y, now it's one row, want to make it a column vector so one column
y = y.view(y.shape[0], 1)

n_samples, n_features = x.shape

# three steps:
# 1. design model
# 2. define loss and optimizer
# 3. training loop

# step one:
# in linear regression case, model has one
# layer so can use builtin linear model

inputSize = n_features
outputSize = 1 # only want one value for each sample we put in
model = nn.Linear(inputSize, outputSize)

# step two:
learning_rate = 0.01
criterion = nn.MSELoss() # calculates MSE for linear regression
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# step three:
num_iterations = 100

for i in range(num_iterations):
    # forward pass and loss
    y_predicted = model(x)
    loss = criterion(y_predicted, y)

    # backward pass
    # does backpropagation and calculates gradients
    loss.backward()

    # update weights
    optimizer.step()
    optimizer.zero_grad()

    if (i+1) % 10 == 0:
        print(f'iteration: {i+1}, loss = {loss.item():.4f}')

# plot
# get all predicted values

# call the final model, and use detach to prevent the tensor
# from being tracked in the graph

# ie requires_grad is true, so detach makes a copy 
# with requires_grad set to false
predicted = model(x).detach().numpy()
plt.plot(x_numpy, y_numpy, 'ro') # plot as red dots
# plot generated/approximated function
plt.plot(x_numpy, predicted, 'b')
plt.show()