# making CNN to classigy CIFAR-10 dataset, which is an
# image dataset with ten classes that's available in pytorch

# CNNs are similar to regular NNs, however CNNs work mainly
# on image data, and apply convolutional filters
# they have
# - convolutional layers
# - optional activation funtions
# - pooling layers
# - at least one fully connected (FC) layer at the end,
#   which does the actual classification

# convolutional filters apply a filter kernal to the image
# first the filter is put at the first position on the image,
# and an output is calculated by multiplying and summing all
# the values or something
# then slide the image over to the next position and do the
# same thing, and do this for each position in the image

# in the end get an output image, need to be careful to
# get the right size for the output image

# pooling layers
# for example max pooling, which downsamples an image by 
# applying a maximum filter to subregions, so take the 
# maximum value in each subregion as the output
# this reduces the size of the image, which reduces cost
# so it reduces parameters that model has to learn, 
# and it avoids overfitting by providing an abstracted form
# of input or something ...?

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data

# GPU support
# device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyperparameters are values set before training,
# so they are not derived from training

# define hyperparameters
num_epochs = 10
batch_size = 4
learning_rate = 0.001

# load dataset
# dataset has PILImage images of range [0, 1]
# transform them to tensors with normalized range [-1, 1]
transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                    download=True, transform=transform)

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                    download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                           shuffle=False)

# hardcode classes
classes = ('plane', 'car', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck')


# structure of the CNN
# feature learning part:
# - convolution layer with relu
# - max pooling
# - second convolution layer with relu
# - second max pooling
# classification part:
# - three fully connected layers
# - softmax and cross entropy

# implement CNN
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Conv2d takes input, output, and kernal sizes
        # input size is 3 because inputs have 3 color channels
        # can choose output and kernal size
        self.conv1 = nn.Conv2d(3, 6, 5)

        # MaxPool2d takes kernal size and stride
        # stride is like how many pixels to shift the max filter
        # when shifting to the next position
        self.pool = nn.MaxPool2d(2, 2)

        # input size should equal the output size of last conv layer
        self.conv2 = nn.Conv2d(6, 16, 5)

        # fully connected layers
        # also takes input and output size

        # input of fc1 and output of fc3 must have
        # certain values, but the rest of the inputs and
        # outputs can be changed

        # very important to get the right input size for fc1
        # ** see bottom for explanation
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)

        # output size of this last layer must be
        # 10 because there are 10 classes
        self.fc3 = nn.Linear(84, 10)


    def forward(self, x):
        # apply layers with activation functions
        # activation function does not change the size
        # also apply pooling layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        # flatten the tensor
        # when do -1, pytorch calculates correct size
        # which in this case is 4, which is the batch_size
        x = x.view(-1, 16*5*5)

        # apply fc1 and fc2 with activation functions
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # fc3 does not have an activation functions
        x = self.fc3(x)

        # no softmax here because it's 
        # included in nn's cross entropy
        return x


model = CNN().to(device)

# multiclasss problem so use cross entropy and SGD
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# training loop
total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # original shape: [4, 3, 32, 32] = 4, 3, 1024
        # input_layer: 3 input channels, 6 output channels
        # 5 kernel size
        images = images.to(device)
        labels = labels.to(device)

        # forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # backpropagation and optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 2000 == 0:
            print(f'epoch [{epoch+1}/{num_epochs}], step [{i+1}/{total_steps}], loss: {loss.item():.4f}')
        
print('Finished Training')
PATH = './cnn.pth'
torch.save(model.state_dict(), PATH)

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)

        # max returns (value, index)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

        for i in range(batch_size):
            label = labels[i]
            pred = predicted[i]
            if label == pred:
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    accuracy = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {accuracy} $')

    for i in range(10):
        accuracy = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of {classes[i]}: {accuracy} %')


# if print images.shape, the original image size is [4, 3, 32, 32]
# this is because there are 4 batches, 3 color channels, and 32x32
# pixels in each image

# after applying conv1, the image size is [4, 6, 28, 28]
# 4 for 4 batches, 6 because we specified 6 output channels, 
# and the actual image has fewer pixels because the filter 
# doesn't fit in the corners

# formula to calculate output size from a convolutional layer:
# (W - F + 2P) / S + 1
# ie (width - filter_size + 2*paddings) / stride + 1
# in this case we don't have paddings

# so in our case
# width = 32
# filter_size = 5
# paddings = 0
# stride = 1

# so according to the formula
# (32 - 5 + 0) / 1 + 1 = 28
# the output size from the first layer is 28x28

# then do the pooling layer, and then the image size is [4, 6, 14, 14]
# this is because a pooling layer with kernal size 2x2 and stride 2
# divides the pixel size in half

# then do the second convolutional layer
# the image size is [4, 16, 10, 10]
# because we specified the next channel output size to be 16
# and according to the formula the pixel size is 10x10
# (14 - 5 + 0) / 1 + 1 = 10

# do the second pooling layer, and image size is [4, 16, 5, 5]
# this again divides the pixel size in half

# then want to put it into the fc layers, but first want to 
# flatten the size from a 3D tensor into a 1D tensor
# in order to do this, multiply the channel size by the pixel sizes,
# so do 16*5*5 to turn it from 3D to 1D
# this is why the input layer to fc1 is 16*5*5