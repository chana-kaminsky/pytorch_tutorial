# implementing a multilayer NN to do digit classification 
# - use MNIST dataset
# - use DataLoader to load it
# - apply transformations to data set
# - make the NN with input, hidden, and output layers, 
#   as well as activation functions
# - set up loss and optimizer 
# - implement training loop with batch training
# - evaluate model and calculate accuracy
# - ensure that code runs on GPU if have GPU support

import torch
import torch.nn as nn
import torchvision # for datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt # to show data
import torch.utils.data


# device configuration
# guarentees that it runs on GPU if there is GPU support
device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# define hyper parameters
input_size = 784 # because images are 28x28, and flatten to be 1D
hidden_size = 100 # can try out different hidden sizes
num_classes = 10 # because there are 10 digits (0-9)
num_epochs = 2 # can also change this
batch_size = 100
learning_rate = 0.001

# import MNIST data

# first is the training data
# root is where the dataset should be stored
# train=True means that this is the training dataset
# apply a transform to convert it to a tensor
# and it should download the data if it's not already downloaded
train_dataset = torchvision.datasets.MNIST(root='./data', train=True,
        transform=transforms.ToTensor(), download=True)

# then the testing data
# here train=False, and don't need to download
test_dataset = torchvision.datasets.MNIST(root='./data', train=False,
        transform=transforms.ToTensor())

# next create dataloaders

# shuffle is good for training
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
        batch_size=batch_size, shuffle=True)

# shuffle doesn't matter for testing/evaluation
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
        batch_size=batch_size, shuffle=False)

# look at one batch of data
examples = iter(train_loader)
# unpack
samples, labels = examples.next()
print(samples.shape, labels.shape)
# ^see 100 because there are 100 samples in the batch,
# 1 because there is 1 channel, and two 28's because
# the image is 28x28

# plot some data
for i in range(6):
    # 2 rows, 3 columns, index i+1
    plt.subplot(2, 3, i+1)
    # cmpa is colormap
    plt.imshow(samples[i][0], cmap='gray')
plt.show()
# see some examples of handwritten digets, want to classify them


# make an NN with one hidden layer
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        # create layers and activation function
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, num_classes)

    # x is a single sample
    def forward(self, x):
        out = self.layer1(x)
        out = self.relu(out)
        out = self.layer2(out)
        # like with most multiclass classifications, don't use
        # activation function here because cross entropy does it
        return out

# create model
model = NeuralNetwork(input_size, hidden_size, num_classes)

# loss and optimizer
criterion = nn.CrossEntropyLoss()

# use Adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# training loop
total_steps = len(train_loader)

# loop over all the epochs
for epoch in range(num_epochs):
    # loop over all the batches in current epoch
    # unpack i into images and labels
    # enumerate over train_loader, which gives index and data
    # where data is the images and labels or something
    for i, (images, labels) in enumerate(train_loader):
        # reshape images 
        # now each image is [100, 1, 28, 28],
        # but input_size = 784, 
        # so images tensor should have size [100, 784]
        # where 100 is the number of batches
        images = images.reshape(-1, 28*28).to(device) 
        lables = labels.to(device)
        # ^to(device) pushes to GPU if it's available

        # forward pass
        outputs = model(images)

        # calculate loss
        loss = criterion(outputs, labels)

        # backward pass
        optimizer.zero_grad() # empty gradients
        loss.backward() # backpropagation
        optimizer.step() # updatate weights

        # print loss
        if (i+1) % 100 == 0:
            print(f'epoch {epoch+1} / {num_epochs}, step {i+1}/{total_steps}, loss = {loss.item():.4f}')

# test
with torch.no_grad(): # done training so don't want to update gradient
    n_correct = 0
    n_samples = 0

    # loop over batches in test samples
    for images, labels in test_loader:
        # reshape like with training data
        # and push to device
        images = images.reshape(-1, 28*28).to(device) 
        lables = labels.to(device)

        # calculate predictions
        outputs = model(images) # using trained model

        # get predictions
        # torch.max returns value and index
        # only want index so put an underscore
        _, predictions = torch.max(outputs, 1)

        # get number of samples in current batch
        n_samples += labels.shape[0]

        # get number of correct predictions
        # basically, if predictions == labels, then n_correct += 1
        n_correct += (predictions == labels).sum().item()

# calculate total accuracy
accuracy = 100.0 * n_correct / n_samples # as a percent
print(f'accuracy = {accuracy}')