from typing import Optional
import torch
from torch import Tensor
import numpy as np
from scipy.stats import multivariate_normal
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Tensor
from dataclasses import dataclass
from mine.samples.base import Sample


@dataclass
class NormalSampleInput():
    sigma: np.ndarray
    dim_x: Optional[int] = None
    dim_y: Optional[int] = None
    number_of_samples: int = 1000000


class NormalSample(Sample):
    def __init__(self,
                 ns_input: NormalSampleInput,
                 ):
        sigma: np.ndarray = ns_input.sigma
        dim_x: Optional[int] = ns_input.dim_x
        dim_y: Optional[int] = ns_input.dim_y
        number_of_samples: int = ns_input.number_of_samples
        assert sigma.ndim == 2
        dim: int = len(sigma)
        assert sigma.shape == (dim, dim)
        self.dim: int = dim
        dim_x = dim_x or dim//2
        dim_y = dim_y or dim - dim_x
        assert dim_x + dim_y == dim
        self.dim_x: int = dim_x
        self.dim_y: int = dim_y
        cov_xy: np.ndarray = sigma
        cov_x: np.ndarray = sigma[:dim_x, :dim_x]
        cov_y: np.ndarray = sigma[dim_x:, dim_x:]

        xy: np.ndarray = multivariate_normal.rvs(
            cov=cov_xy,  # type: ignore
            size=number_of_samples,
        )
        x: np.ndarray = np.atleast_2d(
            multivariate_normal.rvs(
                cov=cov_x,  # type: ignore
                size=number_of_samples,
            ))
        y: np.ndarray = np.atleast_2d(
            multivariate_normal.rvs(
                cov=cov_y,  # type: ignore
                size=number_of_samples,
            ))
        if x.shape == (1, number_of_samples):
            x = x.T
        assert x.shape == (number_of_samples, dim_x)
        if y.shape == (1, number_of_samples):
            y = y.T
        assert y.shape == (number_of_samples, dim_y)
        self.samples: Tensor = torch.from_numpy(
            np.concatenate(
                [xy, x, y], axis=1,
            )
        )
        assert self.samples.size() == torch.Size((number_of_samples, 2*self.dim))
        self.number_of_samples: int = number_of_samples
        self.exact_cov: np.ndarray = cov_xy
        self.empirical_cov_xy: np.ndarray = np.cov(xy, rowvar=False)
        self.empirical_cov_x: np.ndarray = np.atleast_2d(
            np.cov(x, rowvar=False)
        )
        self.empirical_cov_y: np.ndarray = np.atleast_2d(
            np.cov(y, rowvar=False)
        )

    @staticmethod
    def from_(ns):
        sigma: np.ndarray = ns.exact_cov
        dim_x: int = ns.dim_x
        dim_y: int = ns.dim_y
        number_of_samples: int = ns.number_of_samples
        ns_input: NormalSampleInput = NormalSampleInput(
            sigma=sigma,
            dim_x=dim_x,
            dim_y=dim_y,
            number_of_samples=number_of_samples,
        )
        return NormalSample(ns_input)

    def mutual_information(self, empirical: bool = False) -> float:
        Q: np.ndarray
        QX: np.ndarray
        QY: np.ndarray
        dx: int = self.dim_x
        if empirical:
            Q = self.empirical_cov_xy
            QX = self.empirical_cov_x
            QY = self.empirical_cov_y
        else:
            Q = self.exact_cov
            QX = Q[:dx, :dx]
            QY = Q[dx:, dx:]
        det_Q: float = np.linalg.det(Q)
        if det_Q == 0.:
            return np.infty
        else:
            return 0.5 * np.log(
                np.linalg.det(QX) * np.linalg.det(QY) / det_Q
            )


def dataloader(
        ns: Optional[NormalSample] = None,
        ns_input: Optional[NormalSampleInput] = None,
        dim: Optional[int] = None,
        batch_size: Optional[int] = 1,
        shuffle: bool = True,
) -> DataLoader:
    if ns is None:
        if ns_input is None:
            dim = dim or 2
            sigma_sqrt: np.ndarray = np.random.normal(size=(dim, dim))
            sigma: np.ndarray = sigma_sqrt @ sigma_sqrt.T
            ns_input = NormalSampleInput(sigma)
        ns = NormalSample(ns_input)
    return DataLoader(ns, batch_size=batch_size, shuffle=shuffle)
