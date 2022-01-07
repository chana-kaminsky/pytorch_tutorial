# transfer learning is an ML method where a model developed for one
# task is reused as the starting point for a model for a second task

# ex. train a model to recognize birds vs cats, and then use the
# same model, just modify the last layer, to recognize bees and dogs

# transfer learning is popular because it saves a lot of time 
# since it can take a long time to train a whole new model, and
# transfer learning has pretty good results

# with transfer learning, typically only the last layer is changed
# ex. with CNN, can retrain just the last fully connected layer

# doing transfer learning using pretrained resnet 18? CNN,
# which is an NN trained on more than 1 million images and 
# classifies them into over 1000 different categories
# our goal is to classify images as bees or ants

# a scheduler can be used to change the learning rate

# before we used builtin datasets
# now using data that's saved in a folder, using ImageFolder
# the data folder must be divided in a certain way:
# it has 'train' and 'val' folders, 
# each of which have a folder for each class, 
# ex. they would each have an 'ants' folder and a 'bees' folder
# in each of the class folders are images in that class

# then can get image_datasets and class_names

# first model -> classifies ants and bees

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import torch.utils.data

mean = np.array([0.5, 0.5, 0.5])
std = np.array([0.25, 0.25, 0.25])

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
}

# here is where datasets.ImageFolder is used
# it takes a path and transforms
data_dir = 'data/hymenoptera_data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                    data_transforms[x]) for x in ['train', 'val']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], 
                batch_size=4, shuffle=True, num_workers=0)
                for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(class_names)


def showImage(inp, title):
    """image show for tensor"""
    inp = inp.numpy().transpose((1, 2, 0))
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    plt.title(title)
    plt.show()

# get a batch of training data
inputs, classes = next(iter(dataloaders['train']))

# make a grid from batch
out = torchvision.utils.make_grid(inputs)

# showImage(out, title=[class_names[x] for x in classes])

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_weights = copy.deepcopy(model.state_dict())
    best_accuracy = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs-1))
        print('-' * 10)

        # each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train() # set model to training mode
            else:
                model.eval() # set model to evaluate mode
        
            running_loss = 0.0
            running_corrects = 0

            # iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # forward

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backpropagation and optimize if in training phase
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_accuracy = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Accuracy: {:.4f}'.format(
                    phase, epoch_loss, epoch_accuracy))
            
            # deep copy the model
            if phase == 'val' and epoch_accuracy > best_accuracy:
                best_accuracy = epoch_accuracy
                best_model_weights = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
    print('Best val accuracy: {:4f}'.format(best_accuracy))

    # load best model weights
    model.load_state_dict(best_model_weights)
    return model

    
# transfer learning:
# two options
# 1. fine tuning -> retrain the whole model a little bit, fine
#   tune all the weights based on new data and new last layer
# 2. freeze all the layers at the beginning and only retrain
#   the last layer; this option is faster but not as accurate

# option one -> fine tuning
# import pretrained model
# available in torchvision.models module
model = models.resnet18(pretrained=True)

# want to exchange the last fully connected layer
# get number of input features in last fc layer
num_features = model.fc.in_features

# create new layer and assign it to last fc layer
# input_size is num_features, output_size is 2
# because there are 2 classes
model.fc = nn.Linear(num_features, 2)
model.to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# scheduler to update learning rate
# lr scheduler is available in torch.optim module
# give it the optimzer
# step size=7 and gamma=0.1 means that every 7 epochs
# the learning rate is multiplied by 0.1
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
# ^typically scheduler.step() is used like on line 141

model = train_model(model, criterion, optimizer, scheduler, num_epochs=20)

# option two -> only retrain last layer
# first copy all the code from option one

model = models.resnet18(pretrained=True)

# loop over parameters:
for param in model.parameters():
    # set requires_grad to False, which
    # freezes all the beginning layers
    param.requires_grad = False

num_features = model.fc.in_features

# this new last layer by default has requires_grad=True
model.fc = nn.Linear(num_features, 2)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

model = train_model(model, criterion, optimizer, scheduler, num_epochs=20)