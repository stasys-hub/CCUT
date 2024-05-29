import pandas as pd
from tqdm import tqdm
import argparse
import os


def downsample_hic(
    input_file: str,
    output_file: str,
    downsampling_factor: int,
    chunksize: int,
    column_names: list,
) -> None:
    """
    Downsamples Hi-C data from a pairs file.

    This function reads a large pairs file in chunks, downsamples each chunk by a specified factor,
    and writes the downsampled data to a new file.

    Args:
    - input_file: Path to the input pairs file.
    - output_file: Path to the output downsampled pairs file.
    - downsampling_factor: The factor by which to downsample the data.
    - chunksize: Number of rows per chunk to process at a time.
    - column_names: List of column names for the input data.

    Returns:
    - None
    """
    print("Starting downsampling")

    # Open the output file once
    with open(output_file, "w") as f_out:
        for idx, chunk in enumerate(
            tqdm(
                pd.read_csv(
                    input_file,
                    header=None,
                    names=column_names,
                    comment="#",
                    sep="\t",
                    chunksize=chunksize,
                )
            )
        ):
            # Calculate the number of rows to keep for downsampling
            frac_n = len(chunk) // downsampling_factor
            # Downsample the chunk
            downsampled_chunk = chunk.sample(n=frac_n)
            # Write the downsampled chunk to the output file
            downsampled_chunk.to_csv(
                f_out, index=False, header=False, sep="\t", mode="a"
            )

    print("Finished downsampling")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Downsample Hi-C data from a pairs file."
    )
    parser.add_argument("input_file", type=str, help="Input pairs file")
    parser.add_argument(
        "-o",
        "--output_file",
        type=str,
        help="Output pairs file (default: current working directory)",
    )
    parser.add_argument(
        "-f", "--factor", type=int, default=16, help="Downsampling factor (default: 16)"
    )
    parser.add_argument(
        "-c",
        "--chunksize",
        type=int,
        default=10**6,
        help="Chunk size for processing (default: 1,000,000)",
    )
    parser.add_argument(
        "--column_names",
        type=str,
        nargs="+",
        default=[
            "readID",
            "chrom1",
            "pos1",
            "chrom2",
            "pos2",
            "strand1",
            "strand2",
            "pair_type",
            "mapq1",
            "mapq2",
        ],
        help="Custom column names for the pairs file",
    )

    args = parser.parse_args()

    if args.output_file is None:
        # Use the working directory as the default output path
        args.output_file = os.path.basename(args.input_file).replace(
            ".pairs", f"_downsampled_{args.factor}x.pairs"
        )

    downsample_hic(
        args.input_file,
        args.output_file,
        args.factor,
        args.chunksize,
        args.column_names,
    )
