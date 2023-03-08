from torch.utils.data import Dataset
from mine.mutual_information import MutualInformation


def experiment(
        dataset: Dataset,
        dim: int,
        empirical_sample_size: int = 1000,
        optimizer=None,
        learning_rate: float = 1e-3,
        batch_size: int = 1000,
        num_of_epochs: int = 5,
        verbose: bool = False,
        device:str = 'cpu',
) -> MutualInformation:
    mi: MutualInformation = MutualInformation(
        dataset=dataset,
        dim=dim,
        empirical_sample_size=empirical_sample_size,
        device=device,
    )
    mi.train(
        optimizer=optimizer,
        learning_rate=learning_rate,
        batch_size=batch_size,
        num_of_epochs=num_of_epochs,
        verbose=verbose,
    )
    return mi
