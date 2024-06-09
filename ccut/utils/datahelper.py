from typing import Literal, Sequence
import pandas as pd
import pysam as ps
import numpy as np
from tqdm import tqdm
from typing import Union
import os
import h5py
import re

# TODO Write Comments!

# Funtion to translate genomic coordinates to numpy coordinates
def translate_coor(start1, stop1, start2, stop2, resolution=10_000):
    return (
        int(start1 / resolution),
        int(stop1 / resolution),
        int(start2 / resolution),
        int(stop2 / resolution),
    )

def check_regex_genom_coor(input_text: str) -> Union[re.Match[str], None]:
    pattern = re.compile(r"[A-Za-z0-9]+:[0-9]+:[0-9]+", re.IGNORECASE)
    return pattern.match(input_text)


# utility function to convert genomic coordinates in form of chr[1-22]:start:stop
def get_coor_from_str(genomic_position: str) -> tuple[str, int, int]:
    chr, start, stop = genomic_position.split(":")
    return chr, int(start), int(stop)


# get coordinates rounded by bin size
def get_bin(gcoordinates: str, bin_size: Union[str, int]) -> tuple[str, int, int]:
    if type(bin_size) is not int:
        bin_size = int(bin_size)
    chr, start, stop = get_coor_from_str(gcoordinates)
    start_bin = start - (start % bin_size)
    stop_bin = stop + bin_size - ((stop + bin_size) % bin_size)

    return chr, start_bin, stop_bin


# Get numpy array from hdf5 file corresponding to certain genomic interval
def fetch_signal(df, genomic_position: str, bin_size: Union[str, int]) -> np.ndarray:
    chrom, start, stop = get_bin(gcoordinates=genomic_position, bin_size=bin_size)
    if type(bin_size) is not str:
        bin_size = str(bin_size)
    start_idx = np.searchsorted(df[bin_size]["coor"][chrom][:], start)
    stop_idx = np.searchsorted(df[bin_size]["coor"][chrom][:], stop)
    return df[bin_size]["signal"][chrom][start_idx:stop_idx]


def read_bed_file(
    path_to_bed: Union[int, Sequence[int], Literal["infer"], None],
    sep: str = "\t",
    header: Union[int, Sequence[int], Literal["infer"], None] = None,
    column_name_list: Union[list[str], None] = None,
) -> pd.DataFrame:
    df = pd.read_csv(
        filepath_or_buffer=path_to_bed, sep=sep, header=header, names=column_name_list
    )
    return df


def read_fasta(path_to_fasta: str) -> ps.FastaFile:
    return ps.FastaFile(path_to_fasta)


def get_chrom_len_dict(fasta: ps.FastaFile) -> dict[str, int]:
    chrom_segments = {}

    for i in range(len(fasta.references)):
        chrom_segments[fasta.references[i]] = fasta.lengths[i]
    fasta.close()
    return chrom_segments


def create_unbinned_chrom_array(chrom_len: int, dtype=np.float64) -> np.ndarray:
    return np.zeros(chrom_len, dtype=dtype)


def annotate_chrom_array(
    chrom_len: int, annot_pd_df: pd.DataFrame, dtype=np.float64
) -> np.ndarray:
    annot_array = create_unbinned_chrom_array(chrom_len=chrom_len, dtype=dtype)
    # for idx, row in tqdm(
    #     annot_pd_df.iterrows(), total=annot_pd_df.shape[0], desc="Array annotation"
    # ):
    #     start, stop, val = row["START"], row["STOP"], row["VALUE"]
    #     scaled_val = val / (stop - start)
    #     annot_array[start:stop] += scaled_val
    for idx, row in annot_pd_df.iterrows():
        start, stop, val = row["START"], row["STOP"], row["VALUE"]
        scaled_val = val / (stop - start)
        annot_array[start:stop] += scaled_val

    return annot_array


def create_binned_chrom_array(
    unbinned_array: np.ndarray, chrom_len: int, bin_size: int
) -> tuple[np.ndarray, list[int]]:
    steps = chrom_len // bin_size
    binned_array = np.zeros(steps + 1)
    coor = []
    for idx, step in enumerate(range(0, chrom_len, bin_size)):
        coor.append(step)
        binned_array[idx] = unbinned_array[step : (step + bin_size)].sum()
    # for idx, step in enumerate(
    #     tqdm(range(0, chrom_len, bin_size), desc="Binning annotation data")
    # ):
    #     coor.append(step)
    #     binned_array[idx] = unbinned_array[step : (step + bin_size)].sum()
    return binned_array, coor


def create_h5_dataset_high_low(
    path_to_fasta: str,
    path_to_bed_annot_df: Union[int, Sequence[int], Literal["infer"], None],
    chrom_list: list[str],
    h5_file_path: Union[str, os.PathLike],
    bin_size: tuple[int, int],
    df_name: str = "Bed_annotations",
    bed_header_names: Union[list[str], None] = ["CHR", "START", "STOP", "ID", "VALUE"],
    sep: str = "\t",
    bed_header: Union[str, int, list[int], None] = None,
) -> None:
    # get reference genome
    fasta = read_fasta(path_to_fasta=path_to_fasta)
    # get chrom sizes
    chrom_len = get_chrom_len_dict(fasta)
    # get bed annotations
    bed_annot_df = read_bed_file(
        path_to_bed=path_to_bed_annot_df,
        sep=sep,
        header=bed_header,
        column_name_list=bed_header_names,
    )

    # Create a hdf5 dataset
    h5df = h5py.File(h5_file_path, "w")
    grp1 = h5df.create_group(str(bin_size[0]))
    grp2 = h5df.create_group(str(bin_size[1]))

    # create a dataset for every chrom with binned numpy arrays
    for idx, chrom in enumerate(tqdm(chrom_list)):
        # sub df for respective chrom
        sub_chrom_df = bed_annot_df[bed_annot_df["CHR"] == chrom]
        print(f"Annotating: {chrom}")
        # create and annotate numpy array for respective chrom
        full_array = annotate_chrom_array(
            chrom_len=chrom_len[chrom], annot_pd_df=sub_chrom_df
        )
        # create arrays for both bin sizes
        binned_array_1 = create_binned_chrom_array(
            full_array, chrom_len=chrom_len[chrom], bin_size=bin_size[0]
        )
        binned_array_2 = create_binned_chrom_array(
            full_array, chrom_len=chrom_len[chrom], bin_size=bin_size[1]
        )
        # create datasets for respective bin sizes
        grp1.create_dataset(chrom, data=binned_array_1)
        grp2.create_dataset(chrom, data=binned_array_2)


def create_h5_dataset(
    path_to_fasta: str,
    path_to_bed_annot_df: Union[int, Sequence[int], Literal["infer"], None],
    chrom_list: list[str],
    h5_file_path: Union[str, os.PathLike],
    bin_size: int,
    df_name: str = "Bed_annotations",
    bed_header_names: Union[list[str], None] = ["CHR", "START", "STOP", "ID", "VALUE"],
    sep: str = "\t",
    bed_header: Union[str, int, list[int], None] = None,
) -> None:
    # get reference genome
    fasta = read_fasta(path_to_fasta=path_to_fasta)
    # get chrom sizes
    chrom_len = get_chrom_len_dict(fasta)
    # get bed annotations
    bed_annot_df = read_bed_file(
        path_to_bed=path_to_bed_annot_df,
        sep=sep,
        header=bed_header,
        column_name_list=bed_header_names,
    )

    # Create a hdf5 dataset
    h5df = h5py.File(h5_file_path, "w")
    grp = h5df.create_group(str(bin_size))

    # create a dataset for every chrom with binned numpy arrays
    for idx, chrom in enumerate(tqdm(chrom_list)):
        # sub df for respective chrom
        sub_chrom_df = bed_annot_df[bed_annot_df["CHR"] == chrom]
        print(f"Annotating: {chrom}")
        # create and annotate numpy array for respective chrom
        full_array = annotate_chrom_array(
            chrom_len=chrom_len[chrom], annot_pd_df=sub_chrom_df
        )
        # create arrays for both bin sizes
        binned_array = create_binned_chrom_array(
            full_array, chrom_len=chrom_len[chrom], bin_size=bin_size
        )
        # create datasets for respective bin sizes
        grp.create_dataset(chrom, data=binned_array)


# AKTUELLE VERSION
def create_h5_dataset_coor(
    path_to_fasta: str,
    path_to_bed_annot_df: Union[int, Sequence[int], Literal["infer"], None],
    chrom_list: list[str],
    h5_file_path: Union[str, os.PathLike],
    bin_size: int,
    df_name: str = "Bed_annotations",
    bed_header_names: Union[list[str], None] = ["CHR", "START", "STOP", "ID", "VALUE"],
    sep: str = "\t",
    bed_header: Union[str, int, list[int], None] = None,
) -> None:
    # get reference genome
    fasta = read_fasta(path_to_fasta=path_to_fasta)
    # get chrom sizes
    chrom_len = get_chrom_len_dict(fasta)
    # get bed annotations
    bed_annot_df = read_bed_file(
        path_to_bed=path_to_bed_annot_df,
        sep=sep,
        header=bed_header,
        column_name_list=bed_header_names,
    )

    # Create a hdf5 dataset
    h5df = h5py.File(h5_file_path, "w")
    grp = h5df.create_group(str(bin_size))
    bins_grp = grp.create_group("signal")
    coor_grp = grp.create_group("coor")

    # create a dataset for every chrom with binned numpy arrays
    for idx, chrom in enumerate(tqdm(chrom_list)):
        # sub df for respective chrom
        sub_chrom_df = bed_annot_df[bed_annot_df["CHR"] == chrom]
        print(f"Annotating: {chrom}")
        # create and annotate numpy array for respective chrom
        full_array = annotate_chrom_array(
            chrom_len=chrom_len[chrom], annot_pd_df=sub_chrom_df
        )
        # create arrays for both bin sizes
        binned_array, coor = create_binned_chrom_array(
            full_array, chrom_len=chrom_len[chrom], bin_size=bin_size
        )
        # create datasets for respective bin sizes
        bins_grp.create_dataset(chrom, data=binned_array)
        coor_grp.create_dataset(chrom, data=coor)


import pandas as pd


# create a list of genomic regions of interest
def get_bin_list(
    first_bin_pos: int, window_size: int, chrom_length: int, step: float = 0.25
) -> int:
    """
    INPUT:
      first_bin_pos: position where to start to slide over genome
      window_size: size of genomic interval one wants to look at
      chrom_length: length of chromosome
      step: this defines the overlap. Default is 0.5 -> meaning: e.g. if bin 1 starts at window_size = 10_000
      next bin will start at bin_1 + window_size*step = 15_000 leading to an 50 percent overlap
    OUTPUT:
      Returns a list of genomic coordinates for the window
    """

    return [
        [bin, bin + window_size]
        for bin in range(
            first_bin_pos, (chrom_length - window_size), (int(window_size * step))
        )
    ]


def create_df_from_chroms(
    chr_to_process: list[int],
    chr_len: dict,
    step_size: int = 1_600_000,
    step: float = 0.25,
) -> pd.DataFrame:
    """
    INPUT:
        chr_to_process: List of integers corresponding to chrom number
        chr_len: Dict with chrom names as key and length of chrom as value e.g.  {"chr1": 248956422, 'chr2': 242193529}
    OUTPUT:
      Pandas DF containing CHR, Start and Stop positions of bins
    """
    df_list = []
    for idx, chr in enumerate(tqdm(chr_to_process, desc="Creating sample List")):
        # get bins for chromosome
        if isinstance(chr, int):
            df = pd.DataFrame(
                get_bin_list(0, step_size, chr_len[f"chr{chr}"], step=step)
            )
        else:
            if "chr" in chr:
                df = pd.DataFrame(
                    get_bin_list(0, step_size, chr_len[f"{chr}"], step=step)
                )
            else:
                df = pd.DataFrame(
                    get_bin_list(0, step_size, chr_len[f"chr{chr}"], step=step)
                )
        # column names
        chr_col = [chr for i in range(0, df.shape[0])]
        df.columns = ["START", "STOP"]
        df.insert(0, "CHR", chr_col)

        df_list.append(df)

        # df.to_csv(f"../Data/hg19_{chr}_coor.csv", index=False)

    # merge to full df
    final_df = pd.concat(df_list)
    return final_df
