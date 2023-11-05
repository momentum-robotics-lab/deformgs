import torch

class MDNerfDataset(torch.utils.data.Dataset):
    def __init__(self, viewpoints):
        self.viewpoints = viewpoints
        

    def __len__(self):
        return len(self.viewpoints)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        data["gaussians"] = self.gaussians
        data["args"] = self.args
        return data
