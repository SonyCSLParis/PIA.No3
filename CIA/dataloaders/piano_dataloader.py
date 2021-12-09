import torch
from CIA.dataset_managers.piano_helper import PianoIteratorGenerator
from CIA.dataset_managers.piano_midi_dataset import PianoMidiDataset


class PianoDataloaderGenerator:
    def __init__(
        self,
        sequences_size,
        transformations,
        offset_beginning,
        offset_end,
        num_elements,
        *args,
        **kwargs,
    ):
        corpus_it_gen = PianoIteratorGenerator(subsets=[""], num_elements=num_elements)
        dataset: PianoMidiDataset = PianoMidiDataset(
            corpus_it_gen=corpus_it_gen,
            sequence_size=sequences_size,
            smallest_time_shift=0.02,
            max_transposition=6,
            time_dilation_factor=0.1,
            velocity_shift=20,
            transformations=transformations,
            different_time_table_ts_duration=False,
            offset_beginning=offset_beginning,
            offset_end=offset_end,
        )
        self.dataset = dataset
        self.features = ["pitch", "velocity", "duration", "time_shift"]
        self.num_channels = 4

    @property
    def sequences_size(self):
        return self.dataset.sequence_size

    def dataloaders(self, batch_size, shuffle_train=True, shuffle_val=False):
        dataloaders = self.dataset.data_loaders(
            batch_size, shuffle_train=shuffle_train, shuffle_val=shuffle_val
        )

        def _build_dataloader(dataloader):
            for data in dataloader:
                x = torch.stack([data[e] for e in self.features], dim=-1)
                ret = {"x": x}
                yield ret

        dataloaders = [
            _build_dataloader(dataloaders[split]) for split in ["train", "val", "test"]
        ]
        return dataloaders

    def write(self, x, path):
        """
        :param x: (batch_size, num_events, num_channels)
        :return: list of music21.Score
        """
        # TODO add back when fixing signatures for write
        # xs = self.dataset.interleave_silences_batch(x, self.features)
        xs = x
        # values
        sequences = {
            feature: xs[:, feature_index]
            for feature_index, feature in enumerate(self.features)
        }
        score = self.dataset.tensor_to_score(sequences, fill_features=None)
        score.write(f"{path}.mid")
        print(f"File {path}.mid written")

    def get_elapsed_time(self, x):
        """
        This function only returns the aggregated sum,
        it's not properly said the elapsed time
        x is (batch_size, num_events, num_channels)
        """
        assert "time_shift" in self.features
        assert x.shape[2] == 4

        timeshift_indices = x[:, :, self.features.index("time_shift")]
        # convert timeshift indices to their actual duration:
        y = self.dataset.timeshift_indices_to_elapsed_time(
            timeshift_indices, smallest_time_shift=0.02
        )
        cumsum_y = y.cumsum(dim=-1)
        assert torch.all(cumsum_y[:, 1:] >= cumsum_y[:, :-1] - 1e-3)
        return cumsum_y

    def get_feature_index(self, feature_name):
        return self.features.index(feature_name)
