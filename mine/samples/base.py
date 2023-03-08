from typing import Tuple
from torch import Tensor
from torch.utils.data import Dataset


class Sample(Dataset):
    def __init__(self, samples: Tensor, dim: int, device: str = 'cpu'):
        self.samples: Tensor = samples.to(device)
        self.number_of_samples:int = samples.size(0)
        self.dim: int = dim

    def __len__(self) -> int:
        return self.number_of_samples

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        i: int = idx % self.number_of_samples
        d: int = self.dim
        sample: Tensor = self.samples[i, :]
        xy: Tensor = sample[:d]
        x_y: Tensor = sample[d:]
        return xy, x_y
