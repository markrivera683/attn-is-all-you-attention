import torch 
import numpy as np
from torch.utils.data import Dataset, random_split, DataLoader

class SynDataModule:
    def __init__(self, batch_size=32, val_split=0.2):
        data, labels = synthetic_data()
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.val_split = val_split
    
    def setup(self):
        dataset = SynDataset(self.data, self.labels)
        val_size = int(len(dataset) * self.val_split)
        train_size = len(dataset) - val_size
        self.train_dataset, self.val_dataset = random_split(dataset, [train_size, val_size])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)


class SynDataset(Dataset):
    def __init__(self, data: torch.Tensor, labels: torch.Tensor) -> None:
        self.data = data
        self.labels = labels

    def __len__(self) -> int:
        return self.data.shape[0]
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.labels[idx]


def synthetic_data() -> tuple[torch.Tensor, torch.Tensor]:
    def make_covariance(dim: int, rho: float) -> torch.Tensor:
        if not 0 <= rho < 1:
            raise ValueError("rho must be in the range [0, 1)")
        sigma = np.zeros((dim, dim))
        for i in range(dim):
            for j in range(dim):
                sigma[i, j] = rho ** abs(i - j)
        return torch.tensor(sigma, dtype=torch.float32)
    
    sigma = make_covariance(20, 0.9)
    L = torch.linalg.cholesky(sigma)
    z = torch.randn(1000, 20)
    X = z @ L.t()

    true_w = torch.zeros(20)
    true_w[5] = 1.0

    y = X @ true_w + 0.1 * torch.randn(X.shape[0])

    return X, y

