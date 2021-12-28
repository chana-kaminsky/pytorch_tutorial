# about transforms to datasets
# if use builtin dataset, can pass in transform argument
# pytorch has a lot of transforms already implemented
# can see at https://pytorch.org/vision/stable/transforms.html
# ex. transforms that can be applied to images or tensors, ones 
# that do conversions, generic transforms, etc.
# can also compose a list transforms to do multiple transforms in a row

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math


# extending WineDataset class to support transforms
class WineDataset(Dataset):

    # add optional transform parameter whose default it None
    def __init__(self, transform=None):
        xy = np.loadtxt('wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
        
        # note that we do not convert to a tensor here
        self.x = xy[:, 1:]
        self.y = xy[:, [0]]
        self.n_samples = xy.shape[0]

        # add self.transform
        self.transform = transform
    
    def __getitem__(self, index):
        # apply transform if it's available
        sample = self.x[index], self.y[index]

        # if self.transform != None, then transform the sample
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
    def __len__(self):
        return self.n_samples


# create custom transform classes
# write our own ToTensor class
class ToTensor:
    # makes it a callable object, so can call ToTensor()
    def __call__(self, sample):
        # first unpack samples
        inputs, targets = sample
        # returns a tuple
        return torch.from_numpy(inputs), torch.from_numpy(targets)

    
dataset = WineDataset(transform=ToTensor())
first_data = dataset[0]
features, labels = first_data
print(features)
print(type(features), type(labels))
# ^see that features and labels are now of type tensor,
# whereas if pass None as transform argument then they
# are of type numpy.ndarray

# make a multiplication transform
class MultiplicationTransform:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, sample):
        inputs, target = sample
        inputs *= self.factor
        return inputs, target # as tuple

# apply multiplication transform using a composed transform
# Compose takes a list of transforms
composed = torchvision.transforms.Compose([ToTensor(), MultiplicationTransform(4)])
dataset = WineDataset(transform=composed)
first_data = dataset[0]
features, labels = first_data
print(features)
print(type(features), type(labels))
# ^here see that each item got multiplied by the
# factor passed into the multiplication transform