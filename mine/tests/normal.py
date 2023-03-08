from typing import Tuple, Optional, List, Dict
import numpy as np
from mine.mutual_information import MutualInformation
from mine.tests import base
from mine.samples import normal


def experiment(
        ns_input: normal.NormalSampleInput,
        empirical_sample_size: int = 2000,
        optimizer=None,
        learning_rate: float = 1e-3,
        batch_size: int = 2000,
        num_of_epochs: int = 5,
        verbose: bool = False,
        device:str = 'cpu',
) -> Tuple[normal.NormalSample, MutualInformation]:
    ns: normal.NormalSample = normal.NormalSample(ns_input)
    mi: MutualInformation = base.experiment(
        dataset=ns,
        dim=ns.dim,
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
    print(f'mutual information: {ns.mutual_information()}')
    print(
        f'empirical mutual information: {ns.mutual_information(empirical=True)}')
    print(f'estimated mutual information (single-point): {learnt_mi}')
    return ns, mi


def independent_univariates(
        v_0: float = 1.,
        v_1: float = 1.,
        number_of_samples: int = 1000000,
        empirical_sample_size: int = 2000,
        optimizer=None,
        learning_rate: float = 1e-3,
        batch_size: int = 2000,
        num_of_epochs: int = 5,
        verbose: bool = False,
        device:str = 'cpu',
) -> Tuple[normal.NormalSample, MutualInformation]:
    assert v_0 > 0
    assert v_1 > 0
    sigma: np.ndarray = np.diag([v_0, v_1])
    ns_input = normal.NormalSampleInput(
        sigma=sigma,
        number_of_samples=number_of_samples,
        device=device,
    )
    return experiment(
        ns_input=ns_input,
        empirical_sample_size=empirical_sample_size,
        optimizer=optimizer,
        learning_rate=learning_rate,
        batch_size=batch_size,
        num_of_epochs=num_of_epochs,
        verbose=verbose,
        device=device,
    )


def multivariates(
        sigma: Optional[np.ndarray] = None,
        dim_x: Optional[int] = None,
        dim_y: Optional[int] = None,
        number_of_samples: int = 1000000,
        empirical_sample_size: int = 2000,
        optimizer=None,
        learning_rate: float = 1e-3,
        batch_size: int = 2000,
        num_of_epochs: int = 5,
        verbose: bool = False,
        device:str = 'cpu',
) -> Tuple[normal.NormalSample, MutualInformation]:
    if sigma is None:
        assert dim_x is not None
        assert dim_y is not None
        dim = dim_x + dim_y
        A: np.ndarray = np.random.normal(size=(dim, dim))
        sigma = A @ A.T

    ns_input = normal.NormalSampleInput(
        sigma=sigma,
        dim_x=dim_x,
        dim_y=dim_y,
        number_of_samples=number_of_samples,
        device=device,
    )
    return experiment(
        ns_input=ns_input,
        empirical_sample_size=empirical_sample_size,
        optimizer=optimizer,
        learning_rate=learning_rate,
        batch_size=batch_size,
        num_of_epochs=num_of_epochs,
        verbose=verbose,
        device=device,
    )


def smile(
        grid_size: int = 10,
        number_of_samples: int = 1000000,
        insample_empirical_sample_size: int = 2000,
        outsample_empirical_sample_size: int = 7500,
        optimizer=None,
        learning_rate: float = 1e-3,
        batch_size: int = 2000,
        num_of_epochs: int = 5,
        num_evaluations: int = 30,
        verbose: bool = False,
) -> List[Dict[str, float]]:
    rho = np.linspace(-.95, .95, num=grid_size)
    _smile: List[Dict[str, float]] = []
    for rho_i in rho:
        print(f'\n\n\nCorrelation: {rho_i}\n')
        sigma = np.array(
            [[1., rho_i],
             [rho_i, 1.]]
        )
        training_sample, insample_mi = multivariates(
            sigma=sigma,
            number_of_samples=number_of_samples,
            empirical_sample_size=insample_empirical_sample_size,
            optimizer=optimizer,
            learning_rate=learning_rate,
            batch_size=batch_size,
            num_of_epochs=num_of_epochs,
            verbose=verbose,
        )
        test_sample = normal.NormalSample.from_(training_sample)
        outsample_mi = MutualInformation(
            dataset=test_sample,
            test_function=insample_mi.test_function,
            empirical_sample_size=outsample_empirical_sample_size,
        )
        exact_training_mi = training_sample.mutual_information()
        empirical_training_mi = training_sample.mutual_information(
            empirical=True)
        exact_test_mi = test_sample.mutual_information()
        empirical_test_mi = test_sample.mutual_information(empirical=True)
        insample_estimated_mi = np.mean(
            [insample_mi() for _ in range(num_evaluations)])
        outsample_estimated_mi = np.mean(
            [outsample_mi() for _ in range(num_evaluations)])
        res: Dict[str, float] = dict(
            correlation=rho_i,
            exact_training_mi=exact_training_mi,
            empirical_training_mi=empirical_training_mi,
            exact_test_mi=exact_test_mi,
            empirical_test_mi=empirical_test_mi,
            insample_estimated_mi=insample_estimated_mi,
            outsample_estimated_mi=outsample_estimated_mi,
        )
        for k, v in res.items():
            print(f'{k}: {v}')
        print('\n')
        _smile.append(res)
    return _smile


def totally_correlated_univariates(
        v_0: float = 1.,
        v_1: float = 1.,
        number_of_samples: int = 1000000,
        empirical_sample_size: int = 2000,
        optimizer=None,
        learning_rate: float = 5e-3,
        batch_size: int = 2000,
        num_of_epochs: int = 6,
        verbose: bool = False,
) -> Tuple[normal.NormalSample, MutualInformation]:
    assert v_0 > 0
    assert v_1 > 0
    sigma: np.ndarray = np.diag([v_0, v_1])
    sigma[0, 1] = np.sqrt(v_0 * v_1)
    sigma[1, 0] = sigma[0, 1]
    return multivariates(
        sigma=sigma,
        dim_x=1,
        dim_y=1,
        number_of_samples=number_of_samples,
        empirical_sample_size=empirical_sample_size,
        optimizer=optimizer,
        learning_rate=learning_rate,
        batch_size=batch_size,
        num_of_epochs=num_of_epochs,
        verbose=verbose,
    )
