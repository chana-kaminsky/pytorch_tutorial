# different methods to save and load models
# and options for using a gpu

# three torch methods used for saving and loading:
# 1. torch.save(arg, PATH)
#    - can use tensors, models, or any dictionary 
#      as parameter for saving
#    - uses pythons pickel module to serialize objects
#      so results are serialized and not readable by people
# 2. torch.load(PATH)
# 3. model.load_state_dict(arg)

# two options to save model:
# 1. torch.save(model, PATH) -> lazy method
#    - need to specify path
#    - then to load the model do:
#      model = torch.load(PATH)
#      model.eval()
#    - this is lazy because the serialized data is bound
#      to the specific classes and directory structure used
#      when the model is saved
# 2. torch.save(model.state_dict(), PATH) -> recommended method
#    - this only saves the parameters, state_dict holds parameters
#    - then to load the model do:
#      model = Model(*args, **kwargs) -> create model object
#      model.load_state_dict(torch.load(PATH)) -> takes loaded dict
#      model.eval()


import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, n_input_features):
        super(Model, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)
    
    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred

model = Model(n_input_features=6)

# train model...

# ---------------------------
# lazy method:

# define filename, common practice to make it a .pth file for pytorch
FILE = 'model.pth'

# save whole model
torch.save(model, FILE)

# run this, and see that it creates a model.pth file
# if open model.pth, see that it's not human readable

# load the model
model = torch.load(FILE)

# set to evaluation method
model.eval()

# now can use the model ex.
for param in model.parameters():
    print(param)

# run this again and see that it prints the parameters

# recommended method:

# save state_dicts of model
torch.save(model.state_dict(), FILE)

# run this and see that it creates model.pth
# but it only saves the state_dicts

# load model, first create model object
loaded_model = Model(n_input_features=6)
loaded_model.load_state_dict(torch.load(FILE))
loaded_model.eval()

# print parameters of loaded model
for param in loaded_model.parameters():
    print(param)

# since the model isn't trained, the values are random
# see that the parameters are the same for both methods

# ---------------------------
# see what state_dict looks like
print(model.state_dict())
# ^see linear.weight, which is a tensor with the weights
# and have linear.bias, which is a tensor with the bias

# ---------------------------
# can also save a whole checkpoint during training
# make learning_rate and optimizer
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# can print the state_dict of the optimizer
print(optimizer.state_dict())
# ^see state_dict of the optimizer, which
# includes learning rate, momentum, etc

# create a checkpoint, must be a dictionary
checkpoint = {
    'epoch': 90,
    'model_state': model.state_dict(),
    'optim_state': optimizer.state_dict()
}

# then call torch.save() to save the checkpoint
torch.save(checkpoint, 'checkpoint.pth')
# ^after running see that checkpoint.pth is created

# load the checkpoint
loaded_checkpoint = torch.load('checkpoint.pth')

# get epoch
epoch = loaded_checkpoint['epoch']

# reset model and optimizer
model = Model(n_input_features=6)

# can pass in different learning rate but correct one is loaded
optimizer = torch.optim.SGD(model.parameters(), lr=0)

model.load_state_dict(checkpoint['model_state'])
optimizer.load_state_dict(checkpoint['optim_state'])

# once loaded all this from the checkpoint can continue training

print(optimizer.state_dict())
# ^run and see that learning rate is the 

# ---------------------------
# using a GPU (before everything was on CPU)
# how to save a model on GPU and then load it on CPU
PATH = ''

# saving on GPU:
# during training set up cuda device
device = torch.device('cuda')

# send model to device
model.to(device)

# save model
torch.save(model.state_dict(), PATH)

# loading to CPU:
# set up cpu device
device = torch.device('cpu')

# recreate model
model = Model(n_input_features=6)

# load the model
# specify map_location as the CPU device
model.load_state_dict(torch.load(PATH, map_location=device))

# how to both save and load a model on the GPU

# saving model:
device = torch.device('cuda')
model.to(device)
torch.save(model.state_dict(), PATH)

# loading model:
model = Model(n_input_features=6)
model.load_state_dict(torch.load(PATH))
model.to(device)

# how to save model on CPU and load on GPU

# saving model:
torch.save(model.state_dict(), PATH)

# loading model:
device = torch.device('cuda')
model = Model(n_input_features=6)
# specify the map_location as 'cuda:device_number'
model.load_state_dict(torch.load(PATH, map_location='cuda:0'))
model.to(device)