from typing import Dict, Callable, Optional
import numpy as np
from dataclasses import dataclass
from enum import Enum
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from mine.samples.base import Sample


class Transformation(Enum):
    ID = 0
    SIN = 1
    COS = 2
    X2 = 3

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return _transformation_to_numpy_callable[self](x)


_transformation_to_numpy_callable:  Dict[Transformation, Callable[[np.ndarray], np.ndarray]] = {
    Transformation.ID: lambda x: x,
    Transformation.SIN: lambda x: np.sin(np.pi * x / 2.),
    Transformation.COS: lambda x: np.cos(np.pi * x / 2.),
    Transformation.X2: np.square,
}


@dataclass
class SignalInput:
    dim_x: int
    transformation: Transformation
    noise_variance: float
    number_of_samples: int


class Signal(Sample):
    def __init__(self,
                 sig_input: SignalInput
                 ):
        dim_x: int = sig_input.dim_x
        transformation: Transformation = sig_input.transformation
        noise_variance: float = sig_input.noise_variance
        number_of_samples: int = sig_input.number_of_samples

        assert dim_x > 0
        dim_y: int = dim_x
        dim: int = dim_x + dim_y
        noise_std: float = np.sqrt(noise_variance)
        l: np.ndarray = np.zeros((dim_x,), dtype=float)
        h: np.ndarray = np.ones((dim_x,), dtype=float)
        x0: np.ndarray = np.random.uniform(
            low=l,
            high=h,
            size=(number_of_samples, dim_x),
        )
        y0: np.ndarray = (
            transformation(x0) +
            np.random.normal(
                scale=noise_std,
                size=(number_of_samples, dim_y),
            )
        )
        xy: np.ndarray = np.concatenate([x0, y0], axis=1)
        x1: np.ndarray = np.random.uniform(
            low=l,
            high=h,
            size=(number_of_samples, dim_x),
        )
        y1: np.ndarray = (
            transformation(np.random.uniform(
                low=l,
                high=h,
                size=(number_of_samples, dim_x),
            )) +
            np.random.normal(
                scale=noise_std,
                size=(number_of_samples, dim_y)
            )
        )
        samples: Tensor = torch.from_numpy(
            np.concatenate([xy, x1, y1], axis=1),
        )
        assert samples.size() == (number_of_samples, 2 * dim)
        self.samples: Tensor = samples
        self.dim: int = dim
        self.dim_x: int = dim_x
        self.dim_y: int = dim_y
        self.noise_variance: float = noise_variance
        self.number_of_samples: int = number_of_samples

    @staticmethod
    def from_(sig):
        dim_x: int = sig.dim_x
        transformation: Transformation = sig.transformation
        noise_variance: float = sig.noise_variance
        number_of_samples: int = sig.number_of_samples
        sig_input: SignalInput = SignalInput(
            dim_x=dim_x,
            transformation=transformation,
            noise_variance=noise_variance,
            number_of_samples=number_of_samples,
        )
        return Signal(sig_input)


def dataloader(
        sig: Optional[Signal] = None,
        sig_input: Optional[SignalInput] = None,
        batch_size: Optional[int] = 1,
        shuffle: bool = True,
) -> DataLoader:
    if sig is None:
        assert sig_input is not None
        sig = Signal(sig_input)
    return DataLoader(sig, batch_size=batch_size, shuffle=shuffle)
