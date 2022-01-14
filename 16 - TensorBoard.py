# TensorBoard is a visualization toolkit that can be used with pytorch,
# it can help visualize and analyze model and training pipeline.

# can see different things to do on the website:
# https://www.tensorflow.org/tensorboard/

# can also use it in code

# to make a tensorboard:
# - open anaconda shell
# - activate pytorch environment
# - GO TO DIRECTORY that code is in !!!
# - do command 'tensorboard --logdir=runs'
# - it might say that tensorflow is not installed but that's ok
# - copy the url and paste in browser
# - import the tensorboard like in line 27 
#     (needs to be from tensorboard.writer)
# - to run the program, open another anaconda shell 
#     and run from command line

# copy code from module 13

import torch
import torch.nn as nn
import torchvision # for datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt # to show data
import torch.utils.data
import sys
import torch.nn.functional as F

# import tensorboard
from torch.utils.tensorboard.writer import SummaryWriter

# create a writer, takes a directory to save files
# writer = SummaryWriter('runs/mnist')

# changed learning rate:
writer = SummaryWriter('runs/mnist2')

# device configuration
device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# define hyper parameters
input_size = 784
hidden_size = 100
num_classes = 10
num_epochs = 2
batch_size = 100
# learning_rate = 0.001
# tensorboard stuff -> optimize learning rate
learning_rate = 0.01

# import MNIST data
train_dataset = torchvision.datasets.MNIST(root='./data', train=True,
        transform=transforms.ToTensor(), download=True)

test_dataset = torchvision.datasets.MNIST(root='./data', train=False,
        transform=transforms.ToTensor())

# next create dataloaders
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
        batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
        batch_size=batch_size, shuffle=False)

# look at one batch of data
examples = iter(test_loader)
example_data, example_targets = examples.next()
# print(example_data.shape, example_targets.shape)

# plot some data
for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(example_data[i][0], cmap='gray')
# plt.show()

# ------------- TensorBoard stuff -------------
# instead of plotting, add images to tensorboard
# create an image grid, takes data
image_grid = torchvision.utils.make_grid(example_data)

# add the image grid to the writer and give it a label
writer.add_image('mnist_images', image_grid)

# makes sure all the outputs are being written
writer.close()

# want to exit here
# sys.exit() # comment this out to run the graph

# ^see an 8 x 8 grid on tensorboard since batch size is 64

# make an NN with one hidden layer
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.relu(out)
        out = self.layer2(out)
        return out

# create model, loss and optimizer
model = NeuralNetwork(input_size, hidden_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# ------------- TensorBoard stuff -------------
# can add graph to help analyze the data
# give it model and input
# need to reshape input the same as in the loops
writer.add_graph(model, example_data.reshape(-1, 28*28))

# again close the writer and exit the system
writer.close()
#sys.exit()

# ^now see a graph tab in addition to an images tab
# if click on parts of the graph, can see more details of the model

# can also add loss and accuracy to tensorboard
running_loss = 0.0
running_correct = 0

# training loop
total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, 28*28).to(device) 
        lables = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ------------- TensorBoard stuff -------------
        # adding loss and accuracy to tensorboard

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        running_correct += (predicted == labels).sum()

        if (i+1) % 100 == 0:
            print(f'epoch {epoch+1} / {num_epochs}, step {i+1}/{total_steps}, loss = {loss.item():.4f}')
            
            # add loss
            # give it a label, data, and the current global step
            writer.add_scalar('training loss', running_loss / 100, epoch * total_steps + i)
            
            # add accuracy
            writer.add_scalar('accuracy', running_correct / 100, epoch * total_steps + i)

            # reset to 0
            running_loss = 0.0
            running_correct = 0

# ^now also see a scalar tab
# it should have two plots, for loss and for accuracy
# see that accuracy is increasing and loss is decreasing
# can modify how much tensorboard smooths the line on the graph

# can use the graphs to analyze and see if there's somewhere where the
# accuracy doesn't increase or something, know where to look to fix
# things, and one of the first things to optimize is the learning rate

# now let's optimize learning rate and compare
# when reload, that each of the graphs have a second line for mnist2
# so can clearly see the differeces between the learning rates

# can also add precision recall (pr) curves, which helps with
# understanding model performance and threshold settings,
# makes more sense with binary classification problems
# wikipedia -> precision measures quality, recall measures quantity

# should check out https://www.pytorch.org/docs/stable/tensorboard.html
# it talks about the add_pr_curve method

# adding a precision recall curve for each class
labelsList = []
preds = []

# test
with torch.no_grad():
    n_correct = 0
    n_samples = 0

    # loop over batches in test samples
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device) 
        lables = labels.to(device)

        outputs = model(images)
        _, predictions = torch.max(outputs, 1)

        n_samples += labels.shape[0]
        n_correct += (predictions == labels).sum().item()

        # --- TensorBoard Stuff ---
        # use softmax to get probabilities between 0 and 1
        # use list comprehension, dim is the dimension
        class_predictions = [F.softmax(output, dim=0) for output in outputs]
        preds.append(class_predictions)
        labelsList.append(predictions)
    
    # stack the predictions and concatonate 
    # into 2D tensor using list comprehension
    # shape is 10000 x 10, because 10000 samples 
    # and 10 classes, and it's stacked for each class
    preds = torch.cat([torch.stack(batch) for batch in preds])

    # concatonate all elements in labels list into a 1D tensor
    # shape is 10000 x 1
    labelsList = torch.cat(labelsList)

# calculate total accuracy
accuracy = 100.0 * n_correct / n_samples
print(f'accuracy = {accuracy}')

# add the pr curve
classes = range(10)
for i in classes:
    labels_i = labelsList == i

    # preds_i = all samples in current class
    preds_i = preds[:, i]

    # from pytorch website, add_pr_curve takes:
    # add_pr_curve(tag, labels, predictions, global_step=None, 
    # num_thresholds=127, weights=None, walltime=None)
    writer.add_pr_curve(str(i), labels_i, preds_i, global_step=0)
    writer.close()

# now should see the pr curves for each class label
# precision on y axis, recall on x axis
# can analyze for different thresholds