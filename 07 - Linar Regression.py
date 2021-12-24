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

print('hello world')