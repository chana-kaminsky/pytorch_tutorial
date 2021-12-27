# three general steps of training pipeline in pytorch
# 1. design model - input size, output size, forward pass with diff layers
# 2. construct loss and optimizer
# 3. training loop
#   forward pass: compute prediction
#   backward pass: gradients
#   update weights

# code for logistic regression is similar to linear regression, just
# need to make slight adjustments for model and loss function so add
# one more layer to the model, and use a diff builtin loss function

import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets # to load binary classification dataset
from sklearn.preprocessing import StandardScaler # to scale features
from sklearn.model_selection import train_test_split
# ^to sepatate training and testing data

# step 0: prepare data

# binary classification problem where can 
# predict cancer based on input features
bc = datasets.load_breast_cancer()
x = bc.data
y = bc.target

n_samples, n_features = x.shape
# print('samples:', n_samples)
# print('features:', n_features)
# ^shows 569 samples and 30 features

# split data:
# x_train and y_train are training samples
x_train, x_test, y_train, y_test = (train_test_split(x, y, test_size=0.2, random_state=1234))

# scale features
# set up standard scalar
# makes zero mean and unit variance or s/t ..?
# recommended for logistic regression
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

x_train = torch.from_numpy(x_train.astype(np.float32))
x_test = torch.from_numpy(x_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

# reshape y tensors from rows to col vectors
y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)

# three steps:
# 1. design model
# 2. define loss and optimizer
# 3. training loop

# model
# model is linear combination of weights and bias
# f = wx + b
# in logistic regression use sigmoid function at the end

class LogisticRegression(nn.Module):

    def __init__(self, n_input_features):
        super(LogisticRegression, self).__init__()

        # define layer, has one layer
        # output size is 1
        self.linear = nn.Linear(n_input_features, 1)

    # forward pass
    # first apply linear layer, then sigmoid function
    def forward(self, x):
        y_predicted = torch.sigmoid(self.linear(x))
        return y_predicted
    

# layer is size 30 by 1, because input is n_features
# which in this case is 30, and output is always 1
model = LogisticRegression(n_features)

# loss and optimizer
# diff than with linear regression
# BCE is binary cross entropy
criterion = nn.BCELoss()

learning_rate = 0.01
# optimizer is same as with linear regression
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

num_iterations = 100
for i in range(num_iterations):
    # forward pass
    y_predicted = model(x_train)
    loss = criterion(y_predicted, y_train)

    # backward pass
    loss.backward()

    # update weights
    optimizer.step()
    optimizer.zero_grad()

    if (i+1) % 10 == 0:
        print(f'iteration: {i+1}, loss = {loss.item():.4f}')

# evaluate model:
with torch.no_grad():
    # get all predictions from test samples
    y_predicted = model(x_test)
    # sigmoid function returns a value between 0 and 1
    # this splits y_predicted into two "classes",
    # above and below 0.5
    y_predicted_classes = y_predicted.round()
    # use torch.no_grad() so that the .round() computation
    # is not tracked as part of the gradients

    # sum up the correct predictions and divide by number of test samples
    accuracy = y_predicted_classes.eq(y_test).sum() / float(y_test.shape[0])
    print(f'accuracy = {accuracy:.4f}')

# to get better accuracy can play around with number of iterations
# or learning rate, or can try a different optimizer
