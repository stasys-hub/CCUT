from typing import Optional, List, Callable
from torch.utils.data import Dataset
import cooler
import numpy as np
import pandas as pd
import h5py
from .datahelper import fetch_signal
from .metrics import compare_images
import json
import matplotlib.pyplot as plt

# Constants
AXIS = 0
LOW_RES_KEY = "lr"
HIGH_RES_KEY = "hr"
CHROM_PREFIX = (
    "chr"  # You can now use this constant to add or remove the prefix dynamically
)


# Funtion to translate genomic coordinates to numpy coordinates
def translate_coor(start1, stop1, start2, stop2, resolution=10_000):
    return (
        int(start1 / resolution),
        int(stop1 / resolution),
        int(start2 / resolution),
        int(stop2 / resolution),
    )


class DatasetConfig:
    """
    Configuration manager for dataset parameters.

    This class loads a JSON configuration file, providing access methods
    to retrieve various data points needed for the creation of datasets, such as
    paths to cooler files and sample sheets.

    Attributes:
        configs (list of dict): Loaded configuration data.

    Methods:
        get_cooler_pair(index): Retrieves the low and high resolution cooler file paths.
        get_sample_sheet_path(index): Gets the file path for the sample sheet.
        get_prefix(index): Retrieves the chromosome prefix if any.
        get_sample_sheet(index): Reads and returns the sample sheet as a pandas DataFrame.
        get_total_datasets(): Returns the number of datasets in the configuration.
        get_sample_list_length(index): Returns the length of the sample list for a dataset.
        get_cooler_files(index): Loads and returns the cooler files as Cooler objects.
    """

    def __init__(self, config_path):
        with open(config_path, "r") as f:
            self.configs = json.load(f)

    def get_cooler_pair(self, index):
        return {
            "lr": self.configs[index]["cooler_pair"]["low_res"],
            "hr": self.configs[index]["cooler_pair"]["high_res"],
        }

    def get_sample_sheet_path(self, index):
        return self.configs[index]["coordinates"]

    def get_prefix(self, index):
        return self.configs[index]["prefix"]

    def get_sample_sheet(self, index):
        sample_sheet_path = self.get_sample_sheet_path(index)
        return pd.read_csv(sample_sheet_path)

    def get_total_datasets(self):
        return len(self.configs)

    def get_sample_list_length(self, index):
        sample_sheet_path = self.get_sample_sheet(index)
        sample_list = pd.read_csv(sample_sheet_path)
        return len(sample_list)

    def get_cooler_files(self, index):
        coolers = self.get_cooler_pair(index)
        low_res_cooler = cooler.Cooler(coolers["lr"])
        high_res_cooler = cooler.Cooler(coolers["hr"])
        return low_res_cooler, high_res_cooler


class CC_Dataset(Dataset):
    """
    A PyTorch dataset class for loading genomic data from cooler files.

    This dataset supports on-the-fly data transformations and fetching of genomic
    data at various resolutions.

    Attributes:
        config (DatasetConfig): Configuration object containing paths and options.
        balance (bool): Flag indicating whether to balance the data.
        transform_x (list of Callable): List of transformation functions for the input data.
        transform_y (list of Callable): List of transformation functions for the target data.

    Methods:
        __len__(): Returns the total number of samples in the dataset.
        __getitem__(idx): Retrieves the sample corresponding to the index `idx`.
    """

    def __init__(
        self,
        config: DatasetConfig,
        balance: bool = False,
        transform_x: Optional[List[Callable]] = None,
        transform_y: Optional[List[Callable]] = None,
    ):
        self.config = config
        self.balance = balance
        self.transform_x = transform_x
        self.transform_y = transform_y
        # get number of datasets
        self.total_configs = self.config.get_total_datasets()
        # Pre-load all sample sheets
        self.sample_sheets = [
            self.config.get_sample_sheet(i) for i in range(self.total_configs)
        ]

        # Pre-calculate lengths
        self.sample_lengths = [len(sheet) for sheet in self.sample_sheets]
        self.total_length = np.sum(self.sample_lengths)

        # Pre-load all cooler files
        self.cooler_files = [
            self.config.get_cooler_files(i) for i in range(self.total_configs)
        ]

    def get_config_idx(self, idx):
        """Map global index to a config index and local index within the config"""
        config_idx = 0
        while idx >= self.sample_lengths[config_idx]:
            idx -= self.sample_lengths[config_idx]
            config_idx += 1
        return config_idx, idx

    def get_coor(self, idx):
        config_idx, local_idx = self.get_config_idx(idx)

        sample_sheet = self.sample_sheets[config_idx]
        chr, start, stop = sample_sheet.iloc[local_idx]

        return {"CHR": chr, "START": start, "STOP": stop}

    def load_data(self, config_idx, local_idx):
        low_res_cooler, high_res_cooler = self.cooler_files[config_idx]

        # Get the sample_sheet DataFrame for this config
        sample_sheet = self.sample_sheets[config_idx]
        chr, start, stop = sample_sheet.iloc[local_idx]
        prefix = self.config.get_prefix(config_idx)
        # Load the data based on chr, start, stop from low_res and high_res cooler files
        # ...

        x = low_res_cooler.matrix(balance=self.balance).fetch(
            f"{prefix}{chr}:{start}-{stop}"
        )

        y = high_res_cooler.matrix(balance=self.balance).fetch(
            f"{prefix}{chr}:{start}-{stop}"
        )

        # #  replace nans with 0, take log10 where != 0
        y = np.log2(np.nan_to_num(y, nan=0) + 1)
        # replace nans with 0, take log10 where != 0
        x = np.log2(np.nan_to_num(x, nan=0) + 1)

        # check if data altering functions were provided
        if self.transform_x:
            # check if function is on a list
            if type(self.transform_x) is not list:
                self.transform_x = [self.transform_x]
            try:
                for func in self.transform_x:
                    x = func(x)

            except RuntimeWarning:
                print(f"idx: {config_idx}:{local_idx}")
                print(chr, start, stop)
            #       breakpoint()

        if self.transform_y:
            # check if function is on a list
            if type(self.transform_y) is not list:
                self.transform_y = [self.transform_y]
            try:
                for func in self.transform_y:
                    y = func(y)

            except RuntimeWarning:
                print(f"{chr}, {start}, {stop}")

        sample = {
            LOW_RES_KEY: np.expand_dims(x, axis=AXIS).astype(np.float32),
            HIGH_RES_KEY: np.expand_dims(y, axis=AXIS).astype(np.float32),
            "chrom": chr,
            "start": start,
            "stop": stop,
        }

        return sample

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        config_idx, local_idx = self.get_config_idx(idx)
        return self.load_data(config_idx, local_idx)

    def plot_sample(
        self, idx: int, transform: Optional[Callable] = None, compare: bool = False
    ) -> None:
        """
        Plots the low-resolution and high-resolution matrices for a given index.

        Args:
            idx (int): The index of the sample to plot.
            transform (Optional[Callable]): An optional transformation to apply to the matrices before plotting.
            compare (bool): If True, compare the two matrices using the compare_images function.
        """
        sample = self[idx]

        # If a transformation is provided, apply it to both matrices
        if transform:
            # if only one function were passsed -> turn it to list
            if type(transform) is not list:
                transform = [transform]
            # apply iterativly to ccmat
            for func in transform:
                ccmat = func(ccmat)

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        axes[0].set_title("Low Resolution")
        axes[0].imshow(
            sample["lr"][0], cmap="afmhot", interpolation="nearest"
        )  # Assuming the matrix is the first channel

        axes[1].set_title("High Resolution")
        axes[1].imshow(
            sample["hr"][0], cmap="afmhot", interpolation="nearest"
        )  # Assuming the matrix is the first channel

        if compare:
            # Use the compare_images function if needed. Ensure that compare_images returns the metrics you want to show.
            metrics = compare_images(sample["lr"][0], sample["hr"][0])
            axes[1].set_xlabel(
                f"PNSR: {metrics[0]:.3f}, MSE: {metrics[1]:.3f}, MAE: {metrics[2]:.3f}"
            )

        plt.show()


# -------------------------- EXPERIMENTAL --------------------------#


class SRCC_multi_modal(Dataset):
    def __init__(
        self,
        sample_list_file,
        cooler_file_name_full,
        cooler_file_name_down,
        annot_file_name,
        transform_x: Optional[List[Callable]] = None,
        transform_y: Optional[List[Callable]] = None,
        transform_a: Optional[List[Callable]] = None,
        resolution: int = 10_000,
        balance: bool = False,
    ):
        """
        Args:
            sample_list_file: name of csv file containing genomic coordinates of the samples (CHR, START, STOP)
            cooler_file_name_*: Path to cooler file
            transform_*: Transforms that will be applied to data (Functions)
        """

        self.sample_list = pd.read_csv(sample_list_file)
        self.cooler_file_x = cooler.Cooler(f"{cooler_file_name_down}")
        self.cooler_file_y = cooler.Cooler(f"{cooler_file_name_full}")
        self.annot_file = h5py.File(annot_file_name, "r")
        self.transform_x = transform_x
        self.transform_y = transform_y
        self.transform_a = transform_a
        self.balance = balance

    def __len__(self):
        return len(self.sample_list)

    def get_posittion(self, idx):
        chr, start, stop = self.sample_list.iloc[idx, :]
        return (chr, start, stop)

    def __getitem__(self, idx):
        chr, start, stop = self.sample_list.iloc[idx, :]

        # np.array(128,128)
        x = (
            self.cooler_file_x.matrix(balance=self.balance, sparse=True)
            .fetch(f"chr{chr}:{start}-{stop}")
            .toarray()
        )
        # np.array(128,128)
        y = (
            self.cooler_file_y.matrix(balance=self.balance, sparse=True)
            .fetch(f"chr{chr}:{start}-{stop}")
            .toarray()
        )

        # np.array(128,)
        x_annot = fetch_signal(
            self.annot_file, f"chr{chr}:{start}:{stop - 1}", bin_size=10000
        )

        # #  replace nans with 0, take log10 where != 0
        y = np.log2(np.nan_to_num(y, nan=0) + 1)
        # replace nans with 0, take log10 where != 0
        x = np.log2(np.nan_to_num(x, nan=0) + 1)

        # check if data altering functions were provided
        if self.transform_x:
            # check if function is on a list
            if type(self.transform_x) is not list:
                self.transform_x = [self.transform_x]
            try:
                for func in self.transform_x:
                    x = func(x)

            except RuntimeWarning:
                print(idx)
                print(chr, start, stop)
            #       breakpoint()

        if self.transform_y:
            # check if function is on a list
            if type(self.transform_y) is not list:
                self.transform_y = [self.transform_y]
            try:
                for func in self.transform_y:
                    y = func(y)

            except RuntimeWarning:
                print(f"{chr}, {start}, {stop}")
        if self.transform_a:
            # check if function is on a list
            if type(self.transform_a) is not list:
                self.transform_a = [self.transform_a]
            try:
                for func in self.transform_a:
                    x_annot = func(x_annot)

            except RuntimeWarning:
                print(f"{chr}, {start}, {stop}")
        sample = {
            "low_res": np.expand_dims(x, axis=0).astype(np.float32),
            "high_res": np.expand_dims(y, axis=0).astype(np.float32),
            "annot": x_annot.astype(np.float32),
            "chrom": chr,
            "start": start,
            "stop": stop,
        }

        return sample


class NumpyDataset(Dataset):
    def __init__(
        self,
        sample_cordinates_file,
        highres_path,
        lowres_path,
        transform_x=None,
        transform_y=None,
        resolution=10_000,
    ) -> None:
        self.samples = pd.read_csv(sample_cordinates_file)
        self.lr_path = np.load(lowres_path)
        self.hr_path = np.load(highres_path)
        self.lr = dict([(f"{i}", self.lr_path[f"{i}"]) for i in self.lr_path])
        self.hr = dict([(f"{i}", self.hr_path[f"{i}"]) for i in self.hr_path])
        self.trfm_x = transform_x
        self.trfm_y = transform_y
        self.resolution = resolution

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        chr, start1, stop1, start2, stop2 = self.samples.iloc[index]
        idx1, idx2, idy1, idy2 = translate_coor(
            start1, stop1, start2, stop2, self.resolution
        )

        x = self.lr[chr][idx1:idx2, idy1:idy2]
        y = self.hr[chr][idx1:idx2, idy1:idy2]

        # check if data altering functions were provided
        if self.trfm_x:
            # check if function is on a list
            if not isinstance(self.trfm_x, list):
                self.trfm_x = [self.trfm_x]
            try:
                for func in self.trfm_x:
                    x = func(x)

            except RuntimeWarning:
                print(
                    f"Error occured during loading the following sample:\n{chr}:{start1}-{stop1}, {start2}-{stop2}"
                )

        # check if data altering functions were provided
        if self.trfm_y:
            # check if function is on a list
            if not isinstance(self.trfm_y, list):
                self.trfm_y = [self.trfm_y]
            try:
                for func in self.trfm_y:
                    y = func(y)

            except RuntimeWarning:
                print(
                    f"Error occured during loading the following sample:\n{chr}:{start1}-{stop1}, {start2}-{stop2}"
                )

        sample = {
            LOW_RES_KEY: np.expand_dims(x, axis=AXIS).astype(np.float32),
            HIGH_RES_KEY: np.expand_dims(y, axis=AXIS).astype(np.float32),
            "coor": [chr, start1, stop1, start2, stop2],
        }
        return sample

    def plot_sample(
        self, idx: int, transform: Optional[Callable] = None, cmap="hot"
    ) -> None:
        """
        Plots the low-resolution and high-resolution matrices for a given index.

        Args:
            idx (int): The index of the sample to plot.
            transform (Optional[Callable]): An optional transformation to apply to the matrices before plotting.
        """
        sample = self[idx]

        # If a transformation is provided, apply it to both matrices
        if transform:
            # if only one function were passsed -> turn it to list
            if type(transform) is not list:  # noqa: E721
                transform = [transform]
            # apply iterativly to ccmat
            for func in transform:
                sample["lr"][0] = func(sample["lr"][0])
                sample["hr"][0] = func(sample["hr"][0])

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        axes[0].set_title("Low Resolution")
        axes[0].imshow(
            sample["lr"][0], cmap=cmap, interpolation="nearest"
        )  # Assuming the matrix is the first channel

        axes[1].set_title("High Resolution")
        axes[1].imshow(
            sample["hr"][0], cmap=cmap, interpolation="nearest"
        )  # Assuming the matrix is the first channel

        plt.show()
