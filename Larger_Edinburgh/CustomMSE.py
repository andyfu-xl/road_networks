import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CustomMSE(nn.Module):
    """
    Custom loss function for the model
    We are using a combination of MSE and directional difference
    """
    def __init__(self):
        super(CustomMSE, self).__init__()

    def forward(self, output, target):
        mask = (target != 0).float().to(device)
        mse = torch.mean(((output - target) ** 2) * mask)
        directional_diff = torch.mean(torch.abs(torch.atan2(output[:, 1], output[:, 0]) - torch.atan2(target[:, 1], target[:, 0])))
        return mse + directional_diff
