import torch
import torch.nn as nn
from lenet import LeNet

model = LeNet() 
total_params = sum(p.numel() for p in model.parameters())
print("Total number of parameters: " + str(total_params))
