import copy
import glob
import itertools
import os
import pickle
import random
import re
import shutil

import numpy as np
import pretty_midi
import torch
from CIA.dataset_managers.piano_helper import (
    MaestroIteratorGenerator,
    extract_cc,
    find_nearest_value,
    get_time_table_duration,
    get_time_table_ts,
)
from torch.utils import data
from tqdm import tqdm

"""
Typical piano sequence:
p0 p1 TS p0 p1 p2 TS p0 STOP X X X X

If beginning:
START p0 p1 TS p0 p1 p2 TS p0 STOP X X X

If end:
p0 p1 TS p0 p1 p2 TS p0 END STOP X X X

"""

START_SYMBOL = "START"
END_SYMBOL = "END"
PAD_SYMBOL = "XX"


class PianoMidiDataset(data.Dataset):
    """
    Class for all arrangement dataset
    It is highly recommended to run arrangement_statistics before building the database
    """

    def __init__(
        self,
        corpus_it_gen,
        sequence_size,
        smallest_time_shift,
        max_transposition,
        time_dilation_factor,
        velocity_shift,
        transformations,
        different_time_table_ts_duration,
        offset_beginning,
        offset_end,
    ):
        """
        All transformations
        {
            'time_shift': True,
            'time_dilation': True,
            'transposition': True
        }

        :param corpus_it_gen: calling this function returns an iterator
        over chorales (as music21 scores)
        :param name:
        :param metadatas: list[Metadata], the list of used metadatas
        :param subdivision: number of sixteenth notes per beat
        """
        super().__init__()
        self.split = None
        pad_before = -offset_beginning
        if type(pad_before) == int:
            assert (pad_before > 0) and (
                pad_before < sequence_size
            ), "wrong pad_before size"
            self.pad_before = pad_before
        else:
            if pad_before:
                self.pad_before = sequence_size
            else:
                self.pad_before = 1

        pad_after = offset_end  # poor notation... pad_after is negativeto shift start of sequence to the left
        if type(pad_after) == int:
            # can be negative to force sequences to be longer than a certain size
            assert pad_after < sequence_size, "wrong pad_after size"
            self.pad_after = pad_after
        else:
            self.pad_after = 0

        self.list_ids = {"train": [], "validation": [], "test": []}

        self.corpus_it_gen = corpus_it_gen
        self.sequence_size = sequence_size
        self.hop_size = min(sequence_size // 4, 10)

        #  features
        self.smallest_time_shift = smallest_time_shift
        if different_time_table_ts_duration:
            # Legacy... REMOVE IT one day
            self.time_table_duration = get_time_table_duration(self.smallest_time_shift)
            self.time_table_time_shift = get_time_table_ts(self.smallest_time_shift)
        else:
            self.time_table_duration = get_time_table_duration(self.smallest_time_shift)
            self.time_table_time_shift = get_time_table_duration(
                self.smallest_time_shift
            )
        self.pitch_range = range(21, 109)
        self.velocity_range = range(128)
        self.programs = range(128)

        # Index 2 value
        self.index2value = {}
        self.value2index = {}
        self.default_value = {
            "pitch": 60,
            "duration": 0.2,
            "time_shift": 0.1,
            "velocity": 80,
        }
        self.silence_value = {
            "pitch": 60,
            "duration": 0.5,
            "time_shift": 0.5,
            "velocity": 0,
        }

        #  Building/loading the dataset
        if os.path.isfile(self.dataset_file):
            self.load()
        else:
            print(f"Building dataset {str(self)}")
            self.make_tensor_dataset()

        print("Loading index dictionnary")
        # Can be different for every instance, so compute after loading
        self.compute_index_dicts()

        #  data augmentations have to be initialised after loading
        self.max_transposition = max_transposition
        self.time_dilation_factor = time_dilation_factor
        self.velocity_shift = velocity_shift
        self.transformations = transformations

    def __str__(self):
        prefix = str(self.corpus_it_gen)
        name = (
            f"PianoMidi-"
            f"{prefix}-"
            f"{self.sequence_size}_"
            f"{self.smallest_time_shift}"
        )
        if self.pad_before == self.sequence_size:
            name += "_padbefore"
        elif self.pad_before > 1:
            name += f"_padbefore{self.pad_before}"
        if self.pad_after != 0:
            name += f"_padafter{self.pad_after}"
        return name

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.list_ids[self.split])

    @property
    def data_folder_name(self):
        # Same as __str__ but without the sequence_len
        name = f"PianoMidi-{self.corpus_it_gen}"
        return name

    @property
    def cache_dir(self):
        cache_dir = f'{os.path.expanduser("~")}/Data/dataset_cache/PianoMidi'
        if not os.path.exists(cache_dir):
            os.mkdir(cache_dir)
        return cache_dir

    @property
    def dataset_file(self):
        dataset_dir = f"{self.cache_dir}/{str(self)}"
        return dataset_dir

    def save(self):
        # Only save list_ids
        with open(self.dataset_file, "wb") as ff:
            pickle.dump(self.list_ids, ff, 2)

    def load(self):
        """
        Load a dataset while avoiding local parameters specific to the machine used
        :return:
        """
        with open(self.dataset_file, "rb") as ff:
            list_ids = pickle.load(ff)
        self.list_ids = list_ids

    def __getitem__(self, index):
        """
        Generates one sample of data
        """
        # Select sample
        """ttt = time.time()"""
        id = self.list_ids[self.split][index]
        """ttt = time.time() - ttt
        print(f'Get indices: {ttt}')
        ttt = time.time()"""

        # Load data and extract subsequence
        sequence = {}
        with open(
            f'{self.cache_dir}/{self.data_folder_name}/{self.split}/{id["score_name"]}/length.txt'
        ) as ff:
            sequence_length = int(ff.read())

        # start_time can be negative, used for padding
        start_time = id["start_time"]
        sequence_start_time = max(start_time, 0)
        end_time = min(id["start_time"] + self.sequence_size, sequence_length)

        fpr_pitch = np.memmap(
            f'{self.cache_dir}/{self.data_folder_name}/{self.split}/{id["score_name"]}/pitch',
            dtype=int,
            mode="r",
            shape=(sequence_length),
        )
        sequence["pitch"] = fpr_pitch[sequence_start_time:end_time]
        del fpr_pitch
        fpr_velocity = np.memmap(
            f'{self.cache_dir}/{self.data_folder_name}/{self.split}/{id["score_name"]}/velocity',
            dtype=int,
            mode="r",
            shape=(sequence_length),
        )
        sequence["velocity"] = fpr_velocity[sequence_start_time:end_time]
        del fpr_velocity
        fpr_duration = np.memmap(
            f'{self.cache_dir}/{self.data_folder_name}/{self.split}/{id["score_name"]}/duration',
            dtype="float32",
            mode="r",
            shape=(sequence_length),
        )
        sequence["duration"] = fpr_duration[sequence_start_time:end_time]
        del fpr_duration
        fpr_time_shift = np.memmap(
            f'{self.cache_dir}/{self.data_folder_name}/{self.split}/{id["score_name"]}/time_shift',
            dtype="float32",
            mode="r",
            shape=(sequence_length),
        )
        sequence["time_shift"] = fpr_time_shift[sequence_start_time:end_time]
        del fpr_time_shift
        """ttt = time.time() - ttt
        print(f'Loading text files: {ttt}')
        ttt = time.time()"""

        # Perform data augmentations (only for train split)
        if (self.transformations["velocity_shift"]) and (self.split == "train"):
            velocity_shift = int(self.velocity_shift * (2 * random.random() - 1))
            sequence["velocity"] = np.maximum(
                0, np.minimum(127, sequence["velocity"] + velocity_shift)
            )
        else:
            velocity_shift = 0
        if (self.transformations["time_dilation"]) and (self.split == "train"):
            time_dilation_factor = (
                1
                - self.time_dilation_factor
                + 2 * self.time_dilation_factor * random.random()
            )
            sequence["duration"] = sequence["duration"] * time_dilation_factor
            sequence["time_shift"] = sequence["time_shift"] * time_dilation_factor
        else:
            time_dilation_factor = 1
        if (self.transformations["transposition"]) and (self.split == "train"):
            transposition = int(
                random.uniform(-self.max_transposition, self.max_transposition)
            )
            sequence["pitch"] = sequence["pitch"] + transposition
            sequence["pitch"] = np.where(
                sequence["pitch"] > self.pitch_range.stop - 1,
                sequence["pitch"] - 12,
                sequence["pitch"],
            )  # lower one octave for sequence['pitch'] too high
            sequence["pitch"] = np.where(
                sequence["pitch"] < self.pitch_range.start,
                sequence["pitch"] + 12,
                sequence["pitch"],
            )  # raise one octave for pitch too low
        else:
            transposition = 0
        """ttt = time.time() - ttt
        print(f'Data augmentation: {ttt}')
        ttt = time.time()"""

        # Add pad, start and end symbols
        sequence = self.add_start_end_symbols(
            sequence, start_time=start_time, sequence_size=self.sequence_size
        )
        """ttt = time.time() - ttt
        print(f'Adding meta symbols: {ttt}')
        ttt = time.time()"""

        # Tokenize
        sequence = self.tokenize(sequence)
        """ttt = time.time() - ttt
        print(f'Tokenizing: {ttt}')

        print(f'###################################')"""

        return {
            "pitch": torch.tensor(sequence["pitch"]).long(),
            "velocity": torch.tensor(sequence["velocity"]).long(),
            "duration": torch.tensor(sequence["duration"]).long(),
            "time_shift": torch.tensor(sequence["time_shift"]).long(),
            "index": index,
            "data_augmentations": {
                "time_dilation": time_dilation_factor,
                "velocity_shift": velocity_shift,
                "transposition": transposition,
            },
        }

    def add_start_end_symbols(
        self, sequence, start_time, sequence_size, no_end=False, no_start=False
    ):
        sequence = {k: list(v) for k, v in sequence.items()}
        if start_time < 0:
            before_padding_length = -start_time
            if no_start:
                sequence = {
                    k: [PAD_SYMBOL] * before_padding_length + v
                    for k, v in sequence.items()
                }
            else:
                sequence = {
                    k: [PAD_SYMBOL] * (before_padding_length - 1) + [START_SYMBOL] + v
                    for k, v in sequence.items()
                }

        end_padding_length = sequence_size - len(sequence["pitch"])
        if end_padding_length > 0:
            if no_end:
                sequence = {
                    k: v + [PAD_SYMBOL] * end_padding_length
                    for k, v in sequence.items()
                }
            else:
                sequence = {
                    k: v + [END_SYMBOL] + [PAD_SYMBOL] * (end_padding_length - 1)
                    for k, v in sequence.items()
                }

        # assert all sequences have the correct size
        sequence = {k: v[:sequence_size] for k, v in sequence.items()}

        return sequence

    def tokenize(self, sequence):
        sequence["pitch"] = [self.value2index["pitch"][e] for e in sequence["pitch"]]
        sequence["velocity"] = [
            self.value2index["velocity"][e] for e in sequence["velocity"]
        ]
        # legacy...
        # TODO use only one table?!
        # This if state is always True
        if hasattr(self, "time_table_duration"):
            sequence["duration"] = [
                self.value2index["duration"][
                    find_nearest_value(self.time_table_duration, e)
                ]
                if e not in [PAD_SYMBOL, END_SYMBOL, START_SYMBOL]
                else self.value2index["duration"][e]
                for e in sequence["duration"]
            ]
            sequence["time_shift"] = [
                self.value2index["time_shift"][
                    find_nearest_value(self.time_table_time_shift, e)
                ]
                if e not in [PAD_SYMBOL, END_SYMBOL, START_SYMBOL]
                else self.value2index["time_shift"][e]
                for e in sequence["time_shift"]
            ]
        else:
            sequence["duration"] = [
                self.value2index["duration"][find_nearest_value(self.time_table, e)]
                if e not in [PAD_SYMBOL, END_SYMBOL, START_SYMBOL]
                else self.value2index["duration"][e]
                for e in sequence["duration"]
            ]
            sequence["time_shift"] = [
                self.value2index["time_shift"][find_nearest_value(self.time_table, e)]
                if e not in [PAD_SYMBOL, END_SYMBOL, START_SYMBOL]
                else self.value2index["time_shift"][e]
                for e in sequence["time_shift"]
            ]
        return sequence

    def iterator_gen(self):
        return (elem for elem in self.corpus_it_gen())

    def split_datasets(self, split=None, indexed_datasets=None):
        train_dataset = copy.copy(self)
        train_dataset.split = "train"
        val_dataset = copy.copy(self)
        val_dataset.split = "validation"
        test_dataset = copy.copy(self)
        test_dataset.split = "test"
        return {"train": train_dataset, "val": val_dataset, "test": test_dataset}

    def data_loaders(self, batch_size, shuffle_train=True, shuffle_val=False):
        """
        Returns three data loaders obtained by splitting
        self.tensor_dataset according to split
        :param shuffle_val:
        :param shuffle_train:
        :param batch_size:
        :param split:
        :return:
        """

        datasets = self.split_datasets()

        train_dl = data.DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=shuffle_train,
            pin_memory=True,
            drop_last=True,
        )

        val_dl = data.DataLoader(
            datasets["val"],
            batch_size=batch_size,
            shuffle=shuffle_val,
            pin_memory=True,
            drop_last=True,
        )

        test_dl = data.DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
        )
        return {"train": train_dl, "val": val_dl, "test": test_dl}

    def compute_index_dicts(self):
        ######################################################################
        #  Index 2 value
        for feat_name in ["pitch", "velocity", "duration", "time_shift"]:
            index2value = {}
            value2index = {}
            index = 0

            if feat_name == "time_shift":
                values = self.time_table_time_shift
            elif feat_name == "duration":
                values = self.time_table_duration[1:]
            elif feat_name == "pitch":
                values = self.pitch_range
            elif feat_name == "velocity":
                values = self.velocity_range
            else:
                raise Exception

            for value in values:
                index2value[index] = value
                value2index[value] = index
                index2value[index] = value
                value2index[value] = index
                index += 1
            # Pad
            index2value[index] = PAD_SYMBOL
            value2index[PAD_SYMBOL] = index
            index += 1
            # Start
            index2value[index] = START_SYMBOL
            value2index[START_SYMBOL] = index
            index += 1
            # End
            index2value[index] = END_SYMBOL
            value2index[END_SYMBOL] = index
            index += 1

            self.index2value[feat_name] = index2value
            self.value2index[feat_name] = value2index

    def make_tensor_dataset(self):
        """
        Implementation of the make_tensor_dataset abstract base class
        """
        print("Making tensor dataset")

        chunk_counter = {
            "train": 0,
            "validation": 0,
            "test": 0,
        }

        # Build x folder if not existing
        if not os.path.isfile(f"{self.cache_dir}/{self.data_folder_name}/xbuilt"):
            if os.path.isdir(f"{self.cache_dir}/{self.data_folder_name}"):
                shutil.rmtree(f"{self.cache_dir}/{self.data_folder_name}")
            os.mkdir(f"{self.cache_dir}/{self.data_folder_name}")
            os.mkdir(f"{self.cache_dir}/{self.data_folder_name}/train")
            os.mkdir(f"{self.cache_dir}/{self.data_folder_name}/validation")
            os.mkdir(f"{self.cache_dir}/{self.data_folder_name}/test")
            # Iterate over files
            for midi_file, split in tqdm(self.iterator_gen()):
                # midi to sequence
                sequences = self.process_score(midi_file)
                midi_name = os.path.splitext(re.split("/", midi_file)[-1])[0]
                folder_name = (
                    f"{self.cache_dir}/{self.data_folder_name}/{split}/{midi_name}"
                )
                if os.path.exists(folder_name):
                    print(f"Skipped {folder_name}")
                    continue

                os.mkdir(folder_name)

                # np.savetxt(f'{folder_name}/pitch.txt', np.asarray(sequences['pitch']).astype(int), fmt='%d')
                # np.savetxt(f'{folder_name}/velocity.txt', np.asarray(sequences['velocity']).astype(int), fmt='%d')
                # np.savetxt(f'{folder_name}/duration.txt', np.asarray(sequences['duration']).astype(np.float32),
                #            fmt='%.3f')
                # np.savetxt(f'{folder_name}/time_shift.txt', np.asarray(sequences['time_shift']).astype(np.float32),
                #            fmt='%.3f')

                # test mmap
                sequence_length = len(sequences["pitch"])
                with open(f"{folder_name}/length.txt", "w") as ff:
                    ff.write(f"{sequence_length:d}")
                fp_pitch = np.memmap(
                    f"{folder_name}/pitch",
                    dtype=int,
                    mode="w+",
                    shape=(sequence_length),
                )
                fp_pitch[:] = np.asarray(sequences["pitch"]).astype(int)
                del fp_pitch
                fp_velocity = np.memmap(
                    f"{folder_name}/velocity",
                    dtype=int,
                    mode="w+",
                    shape=(sequence_length),
                )
                fp_velocity[:] = np.asarray(sequences["velocity"]).astype(int)
                del fp_velocity
                fp_duration = np.memmap(
                    f"{folder_name}/duration",
                    dtype="float32",
                    mode="w+",
                    shape=(sequence_length),
                )
                fp_duration[:] = np.asarray(sequences["duration"]).astype("float32")
                del fp_duration
                fp_time_shift = np.memmap(
                    f"{folder_name}/time_shift",
                    dtype="float32",
                    mode="w+",
                    shape=(sequence_length),
                )
                fp_time_shift[:] = np.asarray(sequences["time_shift"]).astype("float32")
                del fp_time_shift
            open(f"{self.cache_dir}/{self.data_folder_name}/xbuilt", "w").close()

        # Build index of files
        for split in ["train", "validation", "test"]:
            paths = glob.glob(f"{self.cache_dir}/{self.data_folder_name}/{split}/*")
            for path in paths:
                # read file
                with open(f"{path}/length.txt", "r") as ff:
                    sequence_length = int(ff.read())
                score_name = path.split("/")[-1]

                # split in chunks
                # WARNING difference between self.sequence_size (size of the returned sequences) and sequence_length
                # (actual size of the file)
                start_at = -self.pad_before
                end_at = sequence_length + self.pad_after
                for start_time in range(start_at, end_at, self.hop_size):
                    chunk_counter[split] += 1
                    self.list_ids[split].append(
                        {
                            "score_name": score_name,
                            "start_time": start_time,
                        }
                    )

        print(f"Chunks: {chunk_counter}\n")

        # Save class (actually only serve for self.list_ids, helps with reproducibility)
        self.save()
        return

    def process_score(self, midi_file):
        #  Preprocess midi
        midi = pretty_midi.PrettyMIDI(midi_file)
        raw_sequence = list(
            itertools.chain(
                *[
                    inst.notes
                    for inst in midi.instruments
                    if inst.program in self.programs and not inst.is_drum
                ]
            )
        )
        control_changes = list(
            itertools.chain(
                *[
                    inst.control_changes
                    for inst in midi.instruments
                    if inst.program in self.programs and not inst.is_drum
                ]
            )
        )
        # sort by starting time
        raw_sequence.sort(key=lambda x: x.start)
        control_changes.sort(key=lambda x: x.time)

        #  pedal, cc = 64
        sustain_pedal_time, sustain_pedal_value = extract_cc(
            control_changes=control_changes, channel=64, binarize=True
        )

        # sostenuto pedal, cc = 66
        sostenuto_pedal_time, sostenuto_pedal_value = extract_cc(
            control_changes=control_changes, channel=66, binarize=True
        )

        # soft pedal, cc = 67
        soft_pedal_time, soft_pedal_value = extract_cc(
            control_changes=control_changes, channel=67, binarize=True
        )

        seq_len = len(raw_sequence)

        pitch_sequence = []
        velocity_sequence = []
        duration_sequence = []
        time_shift_sequence = []
        for event_ind in range(seq_len):
            # Get values
            event = raw_sequence[event_ind]
            event_values = {}

            # Compute duration taking sustain
            sustained_index_start = (
                np.searchsorted(sustain_pedal_time, event.start, side="left") - 1
            )
            if sustain_pedal_value[sustained_index_start] == 1:
                if (sustained_index_start + 1) >= len(sustain_pedal_time):
                    event_end_sustained = 0
                else:
                    event_end_sustained = sustain_pedal_time[sustained_index_start + 1]
                event_end = max(event.end, event_end_sustained)
            else:
                event_end = event.end

            #  also check if pedal is pushed before the end of the note !!
            sustained_index_end = (
                np.searchsorted(sustain_pedal_time, event.end, side="left") - 1
            )
            if sustain_pedal_value[sustained_index_end] == 1:
                if (sustained_index_end + 1) >= len(sustain_pedal_time):
                    # notes: that's a problem, means a sustain pedal is not switched off....
                    event_end_sustained = 0
                else:
                    event_end_sustained = sustain_pedal_time[sustained_index_end + 1]
                event_end = max(event.end, event_end_sustained)

            duration_value = find_nearest_value(
                self.time_table_duration[1:], event_end - event.start
            )

            event_values["duration"] = duration_value
            if event.pitch in self.pitch_range:
                event_values["pitch"] = event.pitch
            else:
                continue
            event_values["velocity"] = event.velocity
            if event_ind != seq_len - 1:
                next_event = raw_sequence[event_ind + 1]
                event_values["time_shift"] = find_nearest_value(
                    self.time_table_time_shift, next_event.start - event.start
                )
            else:
                event_values["time_shift"] = duration_value

            #  Convert to str
            pitch_sequence.append(event_values["pitch"])
            velocity_sequence.append(event_values["velocity"])
            duration_sequence.append(event_values["duration"])
            time_shift_sequence.append(event_values["time_shift"])
        return {
            "pitch": pitch_sequence,
            "velocity": velocity_sequence,
            "duration": duration_sequence,
            "time_shift": time_shift_sequence,
        }

    def init_generation_filepath(
        self,
        batch_size,
        context_length,
        filepath,
        banned_instruments=[],
        unknown_instruments=[],
        subdivision=None,
    ):
        raise NotImplementedError

    def interleave_silences_batch(self, sequences, index_order):
        ret = []
        silence_frame = torch.tensor(
            [
                self.value2index[feat_name][self.silence_value[feat_name]]
                for feat_name in index_order
            ]
        )
        for e in sequences:
            ret.extend(e)
            ret.append(silence_frame)
            ret.append(silence_frame)
            ret.append(silence_frame)
        ret_stack = torch.stack(ret, dim=0)
        return ret_stack

    def fill_missing_features(self, sequence, selected_features_indices):
        # Fill in missing features with default values
        default_frame = [
            self.value2index[feat_name][self.default_value[feat_name]]
            for feat_name in self.index_order
        ]
        sequence_filled = torch.tensor([default_frame] * len(sequence))
        sequence_filled[:, selected_features_indices] = sequence
        return sequence_filled

    def tensor_to_score(self, sequences, fill_features):
        # Create score
        score = pretty_midi.PrettyMIDI()
        # 'Acoustic Grand Piano', 'Bright Acoustic Piano',
        #                   'Electric Grand Piano', 'Honky-tonk Piano',
        #                   'Electric Piano 1', 'Electric Piano 2', 'Harpsichord',
        piano_program = pretty_midi.instrument_name_to_program("Acoustic Grand Piano")
        piano = pretty_midi.Instrument(program=piano_program)

        # Fill in missing features with default values
        a_key = list(sequences.keys())[0]
        sequence_length = len(sequences[a_key])
        if fill_features is not None:
            for feature in fill_features:
                sequences[feature] = [0] * sequence_length
        else:
            fill_features = []

        start_time = 0.0
        for t in range(sequence_length):
            pitch_ind = int(sequences["pitch"][t])
            duration_ind = int(sequences["duration"][t])
            velocity_ind = int(sequences["velocity"][t])
            time_shift_ind = int(sequences["time_shift"][t])

            if "pitch" in fill_features:
                pitch_value = self.default_value["pitch"]
            else:
                pitch_value = self.index2value["pitch"][pitch_ind]
            if "duration" in fill_features:
                duration_value = self.default_value["duration"]
            else:
                duration_value = self.index2value["duration"][duration_ind]
            if "velocity" in fill_features:
                velocity_value = self.default_value["velocity"]
            else:
                velocity_value = self.index2value["velocity"][velocity_ind]
            if "time_shift" in fill_features:
                time_shift_value = self.default_value["time_shift"]
            else:
                time_shift_value = self.index2value["time_shift"][time_shift_ind]

            if (
                pitch_value in [PAD_SYMBOL, START_SYMBOL, END_SYMBOL]
                or duration_value in [PAD_SYMBOL, START_SYMBOL, END_SYMBOL]
                or velocity_value in [PAD_SYMBOL, START_SYMBOL, END_SYMBOL]
                or time_shift_value in [PAD_SYMBOL, START_SYMBOL, END_SYMBOL]
            ):
                continue

            note = pretty_midi.Note(
                velocity=velocity_value,
                pitch=pitch_value,
                start=start_time,
                end=start_time + duration_value,
            )

            piano.notes.append(note)

            start_time += time_shift_value

        score.instruments.append(piano)
        return score

    def timeshift_indices_to_elapsed_time(self, timeshift_indices, smallest_time_shift):
        """
        Reverse operation than tokenize using get_time_table_duration
        """
        # TODO write proper test
        # WARNING any change here must be done in
        # get_time_table_duration for consistency
        y = torch.zeros_like(timeshift_indices).float()
        x = timeshift_indices

        # short time shifts
        num_short_time_shifts = int(1 / smallest_time_shift)
        y[x < num_short_time_shifts] = (
            x[x < num_short_time_shifts].float() * smallest_time_shift
        )

        # medium time shifts
        num_medium_time_shifts = int((5.0 - 1.0) / (5.0 * smallest_time_shift))

        medium_mask = torch.logical_and(
            num_short_time_shifts <= x,
            x < num_short_time_shifts + num_medium_time_shifts,
        )
        y[medium_mask] = (
            1.0
            + (x[medium_mask] - num_short_time_shifts).float()
            * 5.0
            * smallest_time_shift
        )

        num_long_time_shifts = int((20.0 - 5.0) / (50.0 * smallest_time_shift))

        long_mask = torch.logical_and(
            num_short_time_shifts + num_medium_time_shifts <= x,
            x < num_short_time_shifts + num_medium_time_shifts + num_long_time_shifts,
        )
        y[long_mask] = (
            5.0
            + (x[long_mask] - num_short_time_shifts - num_medium_time_shifts).float()
            * 50
            * smallest_time_shift
        )
        # if not (x <= (num_short_time_shifts + num_medium_time_shifts +
        #              num_long_time_shifts)).byte().all():
        #     print(x.data)
        # assert (x <= (num_short_time_shifts + num_medium_time_shifts +
        #              num_long_time_shifts)).byte().all()
        assert torch.all(y >= 0)
        return y

    def visualise_batch(self, piano_sequences, writing_dir, filepath):
        # data is a matrix (batch, ...)
        # Visualise a few examples
        if len(piano_sequences.size()) == 1:
            piano_sequences = torch.unsqueeze(piano_sequences, dim=0)

        num_batches = len(piano_sequences)

        for batch_ind in range(num_batches):
            midipath = f"{writing_dir}/{filepath}_{batch_ind}.mid"
            score = self.tensor_to_score(
                sequence=piano_sequences[batch_ind], selected_features=None
            )
            score.write(midipath)


if __name__ == "__main__":
    corpus_it_gen = MaestroIteratorGenerator(composers_filter=[], num_elements=None)
    sequence_size = 120
    smallest_time_shift = 0.02
    max_transposition = 6
    time_dilation_factor = 0.1
    velocity_shift = 10
    transformations = {
        "time_shift": True,
        "time_dilation": True,
        "velocity_shift": True,
        "transposition": True,
    }
    dataset = PianoMidiDataset(
        corpus_it_gen,
        sequence_size,
        smallest_time_shift,
        max_transposition,
        time_dilation_factor,
        velocity_shift,
        transformations,
    )

    dataloaders = dataset.data_loaders(
        batch_size=32, shuffle_train=True, shuffle_val=True
    )

    for x in dataloaders["train"]:
        # Write back to midi
        print("yoyo")
