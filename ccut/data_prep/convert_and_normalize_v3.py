import cooler
import numpy as np
import argparse
from tqdm import tqdm
from multiprocessing import Pool

def get_chrom(clr, chrom: str) -> np.ndarray:
    return np.nan_to_num(clr.matrix(balance=args.balance).fetch(chrom), nan=0)

def clamp_and_norm(matrix, cutoff=None) -> np.ndarray:
    if cutoff is None:
        # Clean from outliers based on the specified percentile
        percentile_cutoff = args.percentile
        percentile_value = np.percentile(matrix, percentile_cutoff)
        cutoff = percentile_value

    # Cap the matrix using np.clip
    matrix_capped = np.clip(matrix, 0, cutoff)

    if args.normalize:
        # Min-max normalization
        if matrix_capped.max() > 0:  # Avoid division by zero
            matrix_normalized = (matrix_capped - matrix_capped.min()) / (matrix_capped.max() - matrix_capped.min())
        else:
            matrix_normalized = matrix_capped

        return matrix_normalized.astype(np.float32)
    else:
        return matrix_capped.astype(np.float32)

def parse_chromosomes_arg(chromosomes_str, available_chromosomes):
    # Parse the input, which could be a single chromosome or a range, e.g., "19" or "19-22"
    if '-' in chromosomes_str:
        start, end = map(int, chromosomes_str.split('-'))
        selected_chromosomes = [f'chr{i}' for i in range(start, end + 1) if f'chr{i}' in available_chromosomes]
    else:
        selected_chromosomes = [f'chr{chromosomes_str}'] if f'chr{chromosomes_str}' in available_chromosomes else []
    return selected_chromosomes

def process_chromosome(data):
    clr, chrom, output_path, prefix, cutoff = data
    processed_data = clamp_and_norm(get_chrom(clr, chrom), cutoff)
    return chrom, processed_data

def main(cooler_path: str, output_path: str, prefix: str, chromosomes: str, processes: int, percentile: float, cutoff: float = None):
    clr = cooler.Cooler(cooler_path)
    available_chromosomes = get_chromosome_names(clr)
    selected_chromosomes = parse_chromosomes_arg(chromosomes, available_chromosomes)
    print(f"Processing cooler file: {cooler_path} for chromosomes: {selected_chromosomes} with percentile cutoff: {percentile} and direct cutoff: {cutoff}")

    # Setup data for multiprocessing
    task_data = [(clr, chrom, output_path, prefix, cutoff) for chrom in selected_chromosomes]

    # Use multiprocessing to process data
    with Pool(processes=processes) as pool:
        results = list(tqdm(pool.imap(process_chromosome, task_data), total=len(task_data), desc="Processing Chromosomes"))

    # Collect results into a dictionary
    chrom_data = {chrom: data for chrom, data in results}

    # Save all chromosomes' data in a single .npz file
    output_file = f"{output_path}/{prefix}chromosomes.npz"
    np.savez(output_file, **chrom_data)

def get_chromosome_names(clr) -> list:
    return clr.chromnames

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process cooler files and save processed data.")
    parser.add_argument('cooler_path', type=str, help="Path to the cooler file.")
    parser.add_argument('--output_path', type=str, default='.', help="Directory to save output files.")
    parser.add_argument('--prefix', type=str, default='', help="File prefix for the output.")
    parser.add_argument('--chromosomes', type=str, required=True, help="Chromosome or range of chromosomes to process, e.g., '19' or '19-22'")
    parser.add_argument('--processes', type=int, default=4, help="Number of processes to use.")
    parser.add_argument('--percentile', type=float, default=99.9, help="Percentile cutoff for outlier removal.")
    parser.add_argument('--cutoff', type=float, default=None, help="Direct cutoff value for capping matrix values.")
    parser.add_argument('--balance', action="store_true", help="Use balanced data")
    parser.add_argument('--normalize', action="store_true", help="Min-max nomraliztion between 0-1")
    args = parser.parse_args()
    main(args.cooler_path, args.output_path, args.prefix, args.chromosomes, args.processes, args.percentile, args.cutoff)
