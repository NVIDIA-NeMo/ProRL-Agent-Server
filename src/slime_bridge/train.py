import torch
import torch.nn as nn
import torch.optim as optim
from polar.train_loop import train_loop

# ...

def train_slime(model, device, train_loader, test_loader, optimizer, epochs):
    train_loop(model, device, train_loader, test_loader, optimizer, epochs)

# ...