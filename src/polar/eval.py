import torch
import torch.nn as nn

# ...

def evaluate(model, device, loader):
    model.eval()
    total_correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            total_correct += (predicted == target).sum().item()
    accuracy = total_correct / len(loader.dataset)
    return accuracy

# ...