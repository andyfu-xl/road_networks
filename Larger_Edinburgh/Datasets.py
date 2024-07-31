from torch.utils.data import Dataset
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class myDataset(Dataset):
    def __init__(self, tensor, input_mask):
        self.tensor = tensor.float().to(device)
        self.input_mask = input_mask.float().to(device)
    def __len__(self):
        return len(self.tensor)
    def __getitem__(self, idx):
        return self.tensor[idx], self.input_mask[idx]

class DatasetWithPlans(Dataset):
    def __init__(self, tensor, input_mask, checkpoints):
        self.tensor = tensor.float().to(device)
        self.input_mask = input_mask.float().to(device)
        self.checkpoints = checkpoints.float().to(device)
    def __len__(self):
        return len(self.tensor)
    def __getitem__(self, idx):
        return self.tensor[idx], self.input_mask[idx], self.checkpoints[idx]