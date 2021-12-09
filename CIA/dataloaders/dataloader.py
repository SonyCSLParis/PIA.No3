class DataloaderGenerator:
    """
    Base abstract class for data loader generators
    dataloaders
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def dataloaders(
        self, batch_size, num_workers, shuffle_train=True, shuffle_val=False
    ):
        raise NotImplementedError
