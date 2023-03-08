
from typing import Tuple, Optional, List, Dict
import numpy as np
from mine.mutual_information import MutualInformation
from mine.tests import base
from mine.samples import projections


def experiment(
        ps_input: projections.ProjectionSampleInput,
        empirical_sample_size: int = 2000,
        optimizer=None,
        learning_rate: float = 1e-3,
        batch_size: int = 2000,
        num_of_epochs: int = 5,
        verbose: bool = False,
        device:str = 'cpu',
) -> Tuple[projections.ProjectionSample, MutualInformation]:
    ps: projections.ProjectionSample = projections.ProjectionSample(ps_input)
    mi: MutualInformation = base.experiment(
        dataset=ps,
        dim=ps.dim,
        empirical_sample_size=empirical_sample_size,
        optimizer=optimizer,
        learning_rate=learning_rate,
        batch_size=batch_size,
        num_of_epochs=num_of_epochs,
        verbose=verbose,
        device=device,
    )
    learnt_mi: float = mi()
    print('\n\n\n')
    print(f'estimated mutual information (single-point): {learnt_mi}')
    return ps, mi


def all_noise(
        dim_x:int = 3,
        dim_y:int = 2,
        number_of_samples: int = 1000000,
        empirical_sample_size: int = 2000,
        optimizer=None,
        learning_rate: float = 1e-3,
        batch_size: int = 2000,
        num_of_epochs: int = 5,
        verbose: bool = False,
        device:str = 'cpu',
) -> Tuple[projections.ProjectionSample, MutualInformation]:
    assert dim_x > 0
    assert dim_y > 0
    assert dim_y <= dim_x
    ps_input = projections.ProjectionSampleInput(
            dim_x=dim_x,
            dim_y=dim_y,
            c0=.0,
            c1=1.,
            number_of_samples=number_of_samples,
            device=device,
    )
    return experiment(
            ps_input,
        empirical_sample_size=empirical_sample_size,
        optimizer=optimizer,
        learning_rate=learning_rate,
        batch_size=batch_size,
        num_of_epochs=num_of_epochs,
        verbose=verbose,
        device=device,
        )


def no_noise(
        dim_x:int = 3,
        dim_y:int = 2,
        number_of_samples: int = 1000000,
        empirical_sample_size: int = 2000,
        optimizer=None,
        learning_rate: float = 1e-3,
        batch_size: int = 2000,
        num_of_epochs: int = 5,
        verbose: bool = False,
        device:str = 'cpu',
) -> Tuple[projections.ProjectionSample, MutualInformation]:
    assert dim_x > 0
    assert dim_y > 0
    assert dim_y <= dim_x
    ps_input = projections.ProjectionSampleInput(
            dim_x=dim_x,
            dim_y=dim_y,
            c0=1.,
            c1=.0,
            number_of_samples=number_of_samples,
            device=device,
    )
    return experiment(
            ps_input,
        empirical_sample_size=empirical_sample_size,
        optimizer=optimizer,
        learning_rate=learning_rate,
        batch_size=batch_size,
        num_of_epochs=num_of_epochs,
        verbose=verbose,
        device=device,
        )

