import torch
import torch.nn as nn
import torch.optim as optim

# ...

def train_loop(model, device, train_loader, test_loader, optimizer, epochs):
    for epoch in range(epochs):
        train(model, device, train_loader, optimizer, epoch)
        # Evaluate on test set every 5 epochs
        if epoch % 5 == 0:
            test_accuracy = evaluate(model, device, test_loader)
            print(f'Epoch {epoch+1}, Test Accuracy: {test_accuracy:.4f}')

# ...