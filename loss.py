import torch.nn as nn
import torch.nn.functional as f

class Function(nn.Module):
    def __init__(self):
        super(Function, self).__init__()


# Loss function

# Mean squared error
class MSE(Function):
    def forward(self, inputs, targets):
        return f.mse_loss(inputs, targets)