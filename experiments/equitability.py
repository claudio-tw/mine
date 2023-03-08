from typing import Tuple,  Dict
from mine.mutual_information import MutualInformation
from experiments import base
from mine.samples import signal


def experiment(
        sig_input: signal.SignalInput,
        empirical_sample_size: int = 2000,
        optimizer=None,
        learning_rate: float = 1e-3,
        batch_size: int = 2000,
        num_of_epochs: int = 5,
        verbose: bool = False,
) -> Tuple[signal.Signal, MutualInformation]:
    sig: signal.Signal = signal.Signal(sig_input)
    mi: MutualInformation = base.experiment(
        dataset=sig,
        dim=sig.dim,
        empirical_sample_size=empirical_sample_size,
        optimizer=optimizer,
        learning_rate=learning_rate,
        batch_size=batch_size,
        num_of_epochs=num_of_epochs,
        verbose=verbose,
    )
    return sig, mi


def univariates(
        noise_variance: float = 1.,
        number_of_samples: int = 1000000,
        empirical_sample_size: int = 2000,
        optimizer=None,
        learning_rate: float = 1e-3,
        batch_size: int = 2000,
        num_of_epochs: int = 2,
        verbose: bool = False,
) -> Dict[signal.Transformation, Tuple[signal.Signal, MutualInformation]]:
    table: Dict[signal.Transformation,
                Tuple[signal.Signal, MutualInformation]] = {}
    for transformation in signal.Transformation:
        print(f'\n\n{transformation}')
        sig_input: signal.SignalInput = signal.SignalInput(
            dim_x=1,
            transformation=transformation,
            noise_variance=noise_variance,
            number_of_samples=number_of_samples,
        )
        sig, mi = experiment(
            sig_input=sig_input,
            empirical_sample_size=empirical_sample_size,
            optimizer=optimizer,
            learning_rate=learning_rate,
            batch_size=batch_size,
            num_of_epochs=num_of_epochs,
            verbose=verbose,
        )
        table.update({
            transformation: (sig, mi)})
    return table
