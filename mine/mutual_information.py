from typing import Optional, Iterator
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, RandomSampler
from mine import donsker_varadhan as dv


class MutualInformation:
    def __init__(self,
                 dataset: Dataset,
                 test_function: Optional[dv.TestFunction] = None,
                 dim: int = 1,
                 dtype=torch.float64,
                 empirical_sample_size: int = 1,
                 ):
        self.dataset: Dataset = dataset
        self.test_function: dv.TestFunction = (
            test_function or dv.TestFunction(
                dim,
                dtype=dtype,
            )
        )
        assert empirical_sample_size > 0
        self.sampler: DataLoader = DataLoader(
            self.dataset,
            batch_size=empirical_sample_size,
            sampler=RandomSampler(
                self.dataset,  # type: ignore
                replacement=True,
            ),
        )
        self.sample: Iterator = iter(self.sampler)

    def train(self,
              optimizer=None,
              learning_rate: float = 1e-3,
              batch_size: int = 1000,
              shuffle: bool = True,
              num_of_epochs: int = 1,
              verbose: bool = True,
              ):
        for epoch in range(num_of_epochs):
            print(f'\n\nEPOCH: {epoch}\n')
            dl: DataLoader = DataLoader(
                self.dataset,
                batch_size=batch_size,
                shuffle=shuffle,
            )
            f_star: dv.TestFunction = dv.train(
                dataloader=dl,
                f=self.test_function,
                optimizer=optimizer,
                learning_rate=learning_rate,
                verbose=verbose,
            )
            self.test_function = f_star

    def __call__(self,
                 xy: Optional[Tensor] = None,
                 x_y: Optional[Tensor] = None,
                 ):
        xy_, x_y_ = next(self.sample)
        joint: Tensor
        product: Tensor
        if xy is None:
            joint = xy_
        else:
            joint = xy
        if x_y is None:
            product = x_y_
        else:
            product = x_y
        f: dv.TestFunction = self.test_function
        v: Tensor = dv.V(joint, product, f)
        return v.item()
