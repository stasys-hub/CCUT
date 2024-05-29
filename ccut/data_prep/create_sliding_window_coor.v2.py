import argparse
import cooler
import pandas as pd





def generate_sliding_window_coords(clr, chrom, window_size=200, resolution=10000, max_distance=2000000, step=1.0, threshold=0):
    chrom_size = clr.chromsizes[chrom]
    if threshold > 0:
        chrom_size = max(0, chrom_size - threshold)  # Adjust chrom size based on threshold

    step_size = int(window_size * resolution * step)
    exact_window_size = window_size * resolution  # This is the exact size each window should be

    coordinates = []
    for i in range(0, chrom_size, step_size):
        start1 = i
        stop1 = start1 + exact_window_size

        # Check if the window exceeds the chromosome size
        if stop1 > chrom_size:
            continue  # Skip this window as it does not fit the exact size requirement

        for j in range(i, chrom_size, step_size):
            start2 = j
            stop2 = start2 + exact_window_size

            # Check if the window exceeds the chromosome size
            if stop2 > chrom_size:
                continue  # Skip this window as it does not fit the exact size requirement

            # Check the distance constraint
            if abs(start1 - start2) <= max_distance:
                coordinates.append([chrom, start1, stop1, start2, stop2])

    return coordinates

def main(args):
    clr = cooler.Cooler(args.cooler_file)
    chroms = args.chromosomes.split(',')

    all_coords = []
    for chrom in chroms:
        coords = generate_sliding_window_coords(clr, chrom, args.window_size, args.resolution, args.max_distance, args.step, args.threshold)
        all_coords.extend(coords)

    df = pd.DataFrame(all_coords, columns=['CHR', 'START1', 'STOP1', 'START2', 'STOP2'])
    df.to_csv(args.output_path, index=False)
    print(f"Coordinates saved to {args.output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate sliding window coordinates for cooler files with distance filter and step parameter")
    parser.add_argument("--cooler_file", type=str, required=True, help="Path to the cooler file")
    parser.add_argument("--chromosomes", type=str, required=True, help="Comma-separated list of chromosomes")
    parser.add_argument("--window_size", type=int, default=200, help="Window size for the sliding window")
    parser.add_argument("--resolution", type=int, default=10000, help="Resolution of the cooler file")
    parser.add_argument("--max_distance", type=int, default=2000000, help="Maximum distance between windows for inclusion")
    parser.add_argument("--threshold", type=int, default=0, help="Substracts treshold from maximum chrom size")
    parser.add_argument("--step", type=float, default=1.0, help="Step size as a fraction of window size for overlapping windows")
    parser.add_argument("--output_path", type=str, default="coordinates.csv", help="Path to save the coordinates CSV")

    args = parser.parse_args()
    main(args)
