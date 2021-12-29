# Softmax function and Cross Entropy loss are some of the most
# common functions used in neural networks

# softmax:
# s(y_i) = (e^(y_i)/sum(e^(y_i)))
# basically it raises each element to the power of e, and then
# normalizes everything by dividing by the sum of all the elements
# raised to the power of e
# it basically squashes the output to be between 0 and 1, 
# which can act as a probability

import torch
import torch.nn as nn
import numpy as np

def softmax(x):
    # np.exp(x) calculates e^x
    # axis=0 means that it sums along the first axis
    return np.exp(x) / np.sum(np.exp(x), axis=0)

x = np.array([2.0, 1.0, 0.1])
outputs = softmax(x)
print('softmax numpy:', outputs)
# ^it returns an array with these numbers squashed between
# 0 and 1, and they are like probabilities relative to the
# numbers in the original array, ex the largest nuumber has
# the highest probability and v/v, etc.

# can also do it in pytorch
x = torch.tensor([2.0, 1.0, 0.1])
# dim means dimensions, so it computes it along the first axis
outputs = torch.softmax(x, dim=0)
print('softmax pytorch:', outputs)
# ^results with numpy and pytorch are basically the same

# softmax function is often combined with cross entropy loss,
# which measures teh performance of classification model whose
# output is a probability between 0 and 1
# can be used in multiclass problems

# loss increases as predicted gets farther from actual
# so better prediction means lower cross entropy loss
# and worse prediction means higher cross entropy loss

# Y must be one hot encoded, meaning that for ex., if there are 
# 4 classes, then Y is an array of length 4, and the index of
# the correct class is a 1, while the indexes of the incorrect
# classes are 0.
# ex. the correct class is 2, so Y = [0,0,1,0], ie Y[2] = 1.

# Y hat (predicted) is encoded with probabilities. So use softmax.

def cross_entropy(actual, predicted):
    # negative of sum of actual labels times log of predicted labels
    loss = -np.sum(actual * np.log(predicted))
    # could normalize by dividing by number of samples
    return loss # / float(predicted.shape[0])

# y is one hot encoded:
# here correct class is 0
y = np.array([1,0,0])

# y_predicted has probabilities:
# get them from softmax that was printed earlier

# this is a good prediction because the probabilities
# corrolate to the original array of [2.0, 1.0, 0.1]
y_predicted_good = np.array([0.7, 0.2, 0.1])

# make up a bad prediction 
# it's bad because the probabilities don't really corrolate
y_predicted_bad = np.array([0.1, 0.3, 0.6])

# compute cross entropy
loss1 = cross_entropy(y, y_predicted_good)
loss2 = cross_entropy(y, y_predicted_bad)
print(f'Loss1 numpy: {loss1:.4f}')
print(f'Loss2 numpy: {loss2:.4f}')
# see that first prediction has a low loss
# and second prediction has a high loss

# doing it in pytorch
loss = nn.CrossEntropyLoss()

# need to be careful because pytorch builtin CrossEntropyLoss() already
# applies nn.LogSoftmax and nn.NLLLoss (negative log likelihood loss)
# so we should not implement out own softmax

# ALSO Y is not one hot encoded, rather should 
# put the actual correct class labels
# also Y hat has raw scores, so no softmax

# put 0 because 0 is the correct class label
y = torch.tensor([0])

# be careful about size
# size here is num_samples x num_classes = 1 x 3
# also use raw values here, not softmax
y_predicted_good = torch.tensor([[2.0, 1.0, 0.1]])
y_predicted_bad = torch.tensor([[0.5, 2.0, 0.3]])

loss1 = loss(y_predicted_good, y)
loss2 = loss(y_predicted_bad, y)
print(loss1.item())
print(loss2.item())
# ^the good prediction has a lower cross entropy loss

# get the actual predictions

# _ means don't actually need it or something..?
# 1 means along the first dimension
_, predictions1 = torch.max(y_predicted_good, 1)
_, predictions2 = torch.max(y_predicted_bad, 1)
print('good:', predictions1)
print('bad:', predictions2)
# this is choosing a class
# the good prediction chooses class 0, which is correct
# but the bad prediction chooses class 1, which is incorrect

# loss in pytorch allows for multiple samples
# ex. try with 3 samples
# ie there are three possible classes that can be correct

# so 2, 0, and 1 are all correct classes
y = torch.tensor([2, 0, 1])

# so size of y hat is num_samples x num_sizes = 3 x 3

# in the good prediction, the first array needs its last value
# to be a high number, because index 2 is correct
# whereas second array needs its first value to be a high
# number, because index 0 is correct
# and third array needs its middle number to be a high value
# because index 1 is correct
y_predicted_good = torch.tensor([[0.1, 1.0, 2.1], # correct label = 2
                                [2.0, 1.0, 0.1],  # correct label = 0
                                [0.1, 3.0, 0.1]]) # correct label = 1

# here want to make bad data so do the opposite as the good prediction
y_predicted_bad = torch.tensor([[2.1, 1.0, 0.1],  # correct label = 2
                                [0.1, 1.0, 2.1],  # correct label = 0
                                [0.1, 3.0, 0.1]]) # correct label = 1


# compute cross entropy loss the same way
loss1 = loss(y_predicted_good, y)
loss2 = loss(y_predicted_bad, y)
print(loss1.item())
print(loss2.item())
# ^good prediction has a low loss, bad prediction has a high loss

_, predictions1 = torch.max(y_predicted_good, 1)
_, predictions2 = torch.max(y_predicted_bad, 1)
print('good:', predictions1)
print('bad:', predictions2)
# ^good prediction is correct, matches actual Y
# but bad prediction is incorrect

# typical neural network in multiclass classification problem
# have an image and want to decide what animal it shows
# have input layer, some hidden layers, maybe some activation 
# functions in between

# at the end of the nn there's a linear layer with one output
# for each class, apply softmax (builtin with pytorch CrossEntropyLoss)
# on each output to get probabilities of each one being correct

# multiclass problem
class MulticlassNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MulticlassNN, self).__init__()
        # define layers
        # first there's a linear layer
        # (nn.Linear takes input_size and output_size)
        self.linear1 = nn.Linear(input_size, hidden_size)
        # then an activation function
        self.relu = nn.ReLU()
        # then a second linear layer
        # at the end there is one output per class
        self.linear2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # apply the layers
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        # no softmax
        return out

# create model
model = MulticlassNN(input_size=28*28, hidden_size=5, num_classes=3)
# calculate cross entropy loss
criterion = nn.CrossEntropyLoss() # includes softmax

# this type of nn can work for multiple classes, like if want it 
# to chose whether an image is a dog or a cat, or whether it's a
# dog, a cat, or a bird, etc.

# however, it's differeant than a binary classification problem,
# ex to determine if the image is a dog and the output is yes/no
# here, the end layer only has one output, and softmax is NOT used
# rather use sigmoid function, which also gives a probability,
# and if it's higher than 0.5 say yes, and if it's lower say no
# in pytorch do nn.BCELoss() which does sigmoid at the end

# binary classification
class BinaryNN(nn.Module):
    # does not take in num_classes, because it's just binary
    def __init__(self, input_size, hidden_size):
        super(BinaryNN, self).__init__()
        
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        # final output is always 1
        self.linear2 = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        # apply the layers
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        # do sigmoid at the end
        y_predicted = torch.sigmoid(out)
        return y_predicted

# create model
model = BinaryNN(input_size=28*28, hidden_size=5)
# use binary cross entropy
criterion = nn.BCELoss()

# we should be aware of the differences between binary and 
# multiclass classification problems, and when to use which