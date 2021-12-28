# pytorch dataset and dataloader classes
# so far we got our data fm a file or s/t, and 
# looped over that, but it's not efficient, takes time

# for large datasets, better to divide training data
# into smaller batches, so in the training loop
# have another loop that loops over the training batches
# then do optimization based on the batches
# dataset and dataloader classes make this easy to do

# some terms:
# epoch = 1 forward and backward pass of ALL training samples
# batch_size = number of training samples in one forward and backward pass
# number_of_iterations = number of passes, each pass using batch_size number of samples
# ex 100 samples with batch_size = 20 --> 100/20 = 5 iterations for 1 epoch

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

# using wine.csv (from pythonengineer's github)
# trying to predict whether a wine is in category 1, 2, or 3
# first row of file is the header
# in the rest of the rows, the first column is the category,
# and the rest of the columns are features

# implement a custom dataset
class WineDataset(Dataset):
    def __init__(self):
        # data loading
        # delimiter means that the data is separated by commas
        # and skiprows=1 means to skip the first row, since it's the header
        xy = np.loadtxt('wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
        # split dataset into x and y
        # use slicing to get all the samples, except the first column (col 0)
        self.x = torch.from_numpy(xy[:, 1:])
        # also want all the samples, but here only want the first column
        self.y = torch.from_numpy(xy[:, [0]])
        # ^also convert them both to tensors, though could also do that later
        self.n_samples = xy.shape[0]

    # this allows for indexing the dataset
    def __getitem__(self, index):
        # ex. dataset[0]
        # this returns a tuple
        return self.x[index], self.y[index]
    
    # allows to call len on the dataset
    def __len__(self):
        # ex. len(dataset)
        return self.n_samples

dataset = WineDataset()

## get first sample
# first_data = dataset[0]
# features, labels = first_data
## features is a single row vector
## labels is just one number, it's the category (I think...)
# print(features, labels)

# shuffle is good for training, it shuffles the data
# num_workers is not necessary but might make training
# faseter because it uses multiple subprocesses
# (I set it to 0 though instead of 2 because it was giving errors.)
dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, num_workers=0)

# data_iterater = iter(dataloader)
# data = data_iterater.next()
# features, labels = data
# print(features, labels)
## batch_size is 4, that's why there are 4 feature vectors and 4 labels

# can iterate over whole dataloader instead of just getting next item
# training loop
num_epochs = 2
total_samples = len(dataset)
num_iterations = math.ceil(total_samples/4)
print(total_samples, num_iterations)
# there are 178 samples and 45 iterations

for epoch in range(num_epochs):
    # enumerate gives index, and inputs and labels
    for i, (inputs, labels) in enumerate(dataloader):
        # forward pass, backward pass, update weights
        # this is a dummy example
        if (i+1) % 5 == 0:
            print(f'epoch {epoch+1}/{num_epochs}, step {i+1}/{num_iterations}, inputs {inputs.shape}')
        # see that there are two epochs, and each epoch has 45 steps
        # and this print statement is executing every 5th step,
        # and tensor is 4 by 13, meaning there are 4 batches
        # (batch_size=4) and each batch has 13 different features


# pytorch already has some builtin datasets
# ex can get mnist dataset -> torchvision.datasets.MNIST()
# can also get fashion-mnist, cifar, and coco datasets