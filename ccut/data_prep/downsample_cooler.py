#!/usr/bin/env python3
import argparse
import cooler
import os
from typing import Optional


def downsample_cooler(
    input_cooler_path: str, output_cooler_path: str, sampling_ratio: float
) -> None:
    """
    Downsamples a Hi-C dataset in cooler format.

    Args:
    - input_cooler_path (str): Path to the input .cool file.
    - output_cooler_path (str): Path for the output downsampled .cool file.
    - sampling_ratio (float): Fraction of contacts to retain in the downsampled dataset.

    Returns:
    None: This function writes the downsampled .cool file to the output_cooler_path.
    """

    # Load the original cooler file
    c = cooler.Cooler(input_cooler_path)

    # Extract bins and pixels
    bins = c.bins()[:]  # DataFrame of the bins
    pixels = c.pixels()[:]  # DataFrame of the pixels (contacts)

    # Calculate the number of pixels to sample based on the sampling ratio
    num_pixels = int(len(pixels) * sampling_ratio)

    # Downsample pixels DataFrame
    downsampled_pixels = pixels.sample(n=num_pixels)

    # Create a new cooler file with the downsampled pixels
    cooler.create_cooler(
        cool_uri=output_cooler_path, bins=bins, pixels=downsampled_pixels
    )


def main():
    """Command line interface for the downsampling script."""
    parser = argparse.ArgumentParser(
        description="Downsample a Hi-C dataset in cooler format."
    )
    parser.add_argument("input_cooler", help="Path to the input .cool file")
    parser.add_argument(
        "-o",
        "--output",
        help="Output file path (default: current working directory)",
        default=os.getcwd(),
    )
    parser.add_argument(
        "-r", "--ratio", help="Sampling ratio (default: 0.16)", type=float, default=0.16
    )

    args = parser.parse_args()

    # Check if the output path is just a directory
    if os.path.isdir(args.output):
        output_cooler_path = os.path.join(
            args.output, os.path.basename(args.input_cooler)
        )
    else:
        output_cooler_path = args.output

    downsample_cooler(args.input_cooler, output_cooler_path, args.ratio)


if __name__ == "__main__":
    main()
