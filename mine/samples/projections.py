from typing import Optional
import torch
from torch import Tensor
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Tensor
from dataclasses import dataclass
from mine.samples.base import Sample

@dataclass
class ProjectionSampleInput():
    dim_x: int
    dim_y: int
    c0: float = 1.
    c1: float = .0
    number_of_samples: int = 1000000
    device:str = 'cpu'



class ProjectionSample(Sample):
    def __init__(self, ps_input:ProjectionSampleInput):
        dim_x:int = ps_input.dim_x
        dim_y:int = ps_input.dim_y
        assert dim_y <= dim_x
        dim:int = dim_x + dim_y
        x: np.ndarray = np.random.uniform(
                size=(ps_input.number_of_samples, 
                    dim_x)
                )
        c0: float = abs(ps_input.c0 )
        c1: float = abs(ps_input.c1 )
        self.k0: float = c0 / (c0 + c1)
        self.k1: float = c1 / (c0 + c1)
        y: np.ndarray = self.k0 * x[:, :dim_y] + self.k1 * np.random.uniform(
                size=(ps_input.number_of_samples, dim_y))
        joint : np.ndarray = np.concatenate((x, y), axis=1)
        x_: np.ndarray = np.array(x, copy=True)
        np.random.shuffle(x_)
        prod: np.ndarray = np.concatenate((x_, y), axis=1)
        assert joint.shape == (
                ps_input.number_of_samples, dim)
        assert prod.shape == (
                ps_input.number_of_samples, dim)
        samples: Tensor = torch.from_numpy(
                np.concatenate(
                    (joint, prod), axis=1
        ))
        self.samples: Tensor = samples.to(ps_input.device)
        self.number_of_samples = ps_input.number_of_samples
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim = dim_x + dim_y
        assert self.samples.size() == (
                self.number_of_samples, 2 * self.dim)

    @staticmethod
    def from_(ps):
        ps_input: ProjectionSampleInput = ProjectionSampleInput(
            dim_x=ps.dim_x,
            dim_y=ps.dim_y,
            c0=ps.k0,
            c1=ps.k1,
            number_of_samples=ps.number_of_samples,
        )
        return ProjectionSample(ps_input)


def dataloader(
        projection: Optional[ProjectionSample] = None,
        ps_input: Optional[ProjectionSampleInput] = None,
        batch_size: Optional[int] = 1,
        shuffle: bool = True,
) -> DataLoader:
    if projection is None:
        assert ps_input is not None
        projection = Projection(ps_input)
    return DataLoader(projection,
            batch_size=batch_size,
            shuffle=shuffle
            )




