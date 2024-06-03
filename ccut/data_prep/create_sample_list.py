import argparse
from ccut.utils.datahelper import read_fasta, get_chrom_len_dict, create_df_from_chroms


def parse_chromosome_list(value):
    """
    Parses a string representing a list of chromosomes into an actual list of integers.

    The input is expected to be in the form of "1-5" or "1,2,5".
    For ranges, both the start and end are inclusive.

    Args:
    - value (str): A string containing a range or list of chromosome numbers.

    Returns:
    - list: A list of integers representing chromosome numbers.

    Raises:
    - argparse.ArgumentTypeError: If the input string is not properly formatted.
    """
    try:
        if "-" in value:
            start, end = map(int, value.split("-"))
            return list(range(start, end + 1))
        else:
            return [int(x) for x in value.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError("Invalid chromosome specification")


def main(args):
    """
    Main function to create a DataFrame with genomic coordinates.

    This function reads the genome from a fasta file, gets the length of each chromosome,
    creates a DataFrame of genomic coordinates, and writes it to a CSV file.

    Args:
    - args: Command line arguments parsed by argparse.

    Returns:
    - None
    """
    print("File:", args.file)
    print("Step Size:", args.step_size)
    print("Step:", args.step)
    print("Chromosomes:", args.chromosomes)

    # Write the DataFrame to a CSV file
    genome = read_fasta(args.file)
    genome_len = get_chrom_len_dict(genome)
    # Create a DataFrame with genomic coordinates based on the provided chromosomes and step size
    df = create_df_from_chroms(
        args.chromosomes, genome_len, step_size=args.step_size, step=args.step
    )

    if args.output_path:
        print("Output Path:", args.output_path)
        df.to_csv(f"{args.output_path}", index=False)
    else:
        print("No output path specified, writing 'sample_list.csv' to cwd!")
        df.to_csv("sample_list.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a sample list of genomic coordinates for Training"
    )

    # Mandatory argument
    parser.add_argument("file", type=str, help="Path to the input Fasta file")

    # Optional arguments
    parser.add_argument(
        "--step_size",
        type=int,
        default=2_000_000,
        help="Step size (default: 2_000_000)",
    )
    parser.add_argument(
        "--step", type=float, default=0.25, help="Step value (default: 0.25)"
    )
    parser.add_argument("-o", "--output_path", type=str, help="Path to the output file")
    parser.add_argument(
        "-chr",
        "--chromosomes",
        type=parse_chromosome_list,
        required=True,
        help="Chromosomes to process",
    )
    args = parser.parse_args()
    main(args)
