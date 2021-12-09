import csv
import glob
import os
import numpy as np


class PianoIteratorGenerator:
    """
    Object that returns a iterator over midi files when called
    :return:
    """

    def __init__(self, subsets, num_elements=None):
        # TODO hard coded: create multiple IteratorGenerators
        # self.path = f'{os.path.expanduser("~")}/Data/databases/Piano'
        # trains on transcriptions only:
        self.path = (
            f'{os.path.expanduser("~")}/Data/databases/Piano/transcriptions/midi'
        )
        # trains on piano relax
        # self.path = f'{os.path.expanduser("~")}/Data/databases/Piano/transcriptions/relax_piano'
        # trains on dirk
        # self.path = f'{os.path.expanduser("~")}/Data/databases/Piano/dirk'
        self.subsets = subsets
        self.num_elements = num_elements

    def __call__(self, *args, **kwargs):
        it = (xml_file for xml_file in self.generator())
        return it

    def __str__(self) -> str:
        # TODO take into account subsets?
        ret = "PianoIterator"
        # ret = 'PianoRelax'
        # ret = 'Dirk3'
        if self.num_elements is not None:
            ret += f"_{self.num_elements}"
        return ret

    def generator(self):
        midi_files = []
        for subset in self.subsets:
            # Should return pairs of files
            midi_files += glob.glob(
                os.path.join(self.path, subset, "**", "*.mid"), recursive=True
            )
            midi_files += glob.glob(
                os.path.join(self.path, subset, "*.midi"), recursive=True
            )
            midi_files += glob.glob(
                os.path.join(self.path, subset, "*.MID"), recursive=True
            )

        if self.num_elements is not None:
            midi_files = midi_files[: self.num_elements]

        split_csv_path = os.path.join(self.path, f"split_{str(self)}.csv")
        if not os.path.exists(split_csv_path):
            self._create_split_csv(midi_files, split_csv_path)
        with open(split_csv_path, "r") as csv_file:
            split_csv = csv.DictReader(csv_file, delimiter="\t")
            # create dict so that we can close the file
            d = {}
            for row in split_csv:
                midi_file = row["midi_filename"]
                split = row["split"]
                d[midi_file] = split
        for midi_file, split in d.items():
            print(midi_file)
            yield midi_file, split

    def _create_split_csv(self, midi_files, split_csv_path):
        print("Creating CSV split")
        with open(split_csv_path, "w") as file:
            # header
            header = "midi_filename\tsplit\n"
            file.write(header)

            for k, midi_file_path in enumerate(midi_files):
                # 90/10/0 split
                if k % 10 == 0:
                    split = "validation"
                else:
                    split = "train"
                entry = f"{midi_file_path}\t{split}\n"
                file.write(entry)


class MaestroIteratorGenerator:
    """
    Object that returns a iterator over xml files when called
    :return:
    """

    def __init__(self, composers_filter=[], num_elements=None):
        self.path = f'{os.path.expanduser("~")}/Data/databases/Piano/maestro-v2.0.0'
        self.composers_filter = composers_filter
        self.num_elements = num_elements

    def __str__(self):
        ret = "Maestro"
        if self.num_elements is not None:
            ret += f"_{self.num_elements}"
        return ret

    def __call__(self, *args, **kwargs):
        it = (elem for elem in self.generator())
        return it

    def generator(self):
        midi_files = []
        splits = []
        master_csv_path = f"{self.path}/maestro-v2.0.0.csv"
        with open(master_csv_path, "r") as csv_file:
            csv_reader = csv.DictReader(csv_file, delimiter=",")
            for row in csv_reader:
                if row["canonical_composer"] in self.composers_filter:
                    continue
                else:
                    midi_name = row["midi_filename"]
                    midi_files.append(f"{self.path}/{midi_name}")
                    splits.append(row["split"])
        if self.num_elements is not None:
            midi_files = midi_files[: self.num_elements]
            splits = splits[: self.num_elements]
        for split, midi_file in zip(splits, midi_files):
            print(f"{split}: {midi_file}")
            yield midi_file, split


def extract_cc(control_changes, channel, binarize):
    ret_time = []
    ret_value = []
    previous_value = -1
    for cc in control_changes:
        if cc.number == channel:
            if binarize:
                value = 1 if cc.value > 0 else 0
                if value == previous_value:
                    continue
            else:
                value = cc.value
            ret_time.append(cc.time)
            ret_value.append(value)
            previous_value = value

    if len(ret_time) == 0:
        ret_time = [0]
        ret_value = [0]
    elif ret_time[0] > 0:
        ret_time.insert(0, 0.0)
        ret_value.insert(0, 0)

    return np.array(ret_time), np.array(ret_value)


def get_midi_type(midi, midi_ranges):
    for feat_name, feat_range in midi_ranges.items():
        if midi in feat_range:
            midi_type = feat_name
            return midi_type


def find_nearest_value(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def get_time_table_ts(smallest_time_shift):
    short_time_shifts = np.arange(0, 0.5, smallest_time_shift)
    medium_time_shifts = np.arange(0.5, 5.0, 5.0 * smallest_time_shift)
    time_shift_bins = np.concatenate((short_time_shifts, medium_time_shifts))
    return time_shift_bins


def get_time_table_duration(smallest_time_shift):
    # WARNING any change here must be done in
    # PianoMidiDataset.timeshift_indices_to_elapsed_time for consistency
    short_time_shifts = np.arange(0, 1.0, smallest_time_shift)
    medium_time_shifts = np.arange(1.0, 5.0, 5.0 * smallest_time_shift)
    long_time_shifts = np.arange(5.0, 20.0, 50 * smallest_time_shift)
    time_shift_bins = np.concatenate(
        (short_time_shifts, medium_time_shifts, long_time_shifts)
    )
    return time_shift_bins
