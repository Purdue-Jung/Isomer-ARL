import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import random

# CONSTANTS
CALIB_OFFSET = 1.785
CALIB_SLOPE = 0.99943
DEFAULT_MATRIX_SIZE = 4096
energy_axis = CALIB_OFFSET + CALIB_SLOPE * np.arange(DEFAULT_MATRIX_SIZE)
y_margin = 50
x_spacing = 3.0

# Generates Bands from E(level)(keV)
class BandFromLevels:
    def __init__(self, levels: List[Tuple[str, float]], branching_prob: float = 0.3):
        self.levels = sorted(levels, key=lambda x: -x[1])  # sort by energy descending
        self.nodes = {f"L0_{i}": level[1] for i, level in enumerate(self.levels)}
        self.labels = {f"L0_{i}": level[0] for i, level in enumerate(self.levels)}
        self.edges = self.generate_edges(branching_prob)

    def generate_edges(self, branching_prob: float) -> List[Tuple[str, str]]:
        edges = []
        keys = list(self.nodes.keys())
        n = len(keys)
        for i in range(n - 1):
            edges.append((keys[i], keys[i + 1]))  # main decay path
            for j in range(i + 2, n):
                if np.random.rand() < branching_prob:
                    edges.append((keys[i], keys[j]))  # optional branch
        return edges


'''# Randomly split levels into multiple bands
def split_into_bands(levels: List[Tuple[str, float]], n_bands: int = 3) -> List[BandFromLevels]:
    random.shuffle(levels)
    bands = [[] for _ in range(n_bands)]
    for i, level in enumerate(levels):
        bands[i % n_bands].append(level)
    return [BandFromLevels(band, branching_prob=0.3) for band in bands]
'''
#Split levels into bands, Hf173 = 8 bands
def split_into_bands(levels: List[Tuple[str, float]]) -> List[BandFromLevels]:
    random.shuffle(levels)
    n_bands = 8
    bands = [[] for _ in range(n_bands)]
    for i, level in enumerate(levels):
        bands[i % n_bands].append(level)
    return [BandFromLevels(band, branching_prob=0.3) for band in bands]



# Plot multiple bands side by side
def plot_multiple_bands(bands: List[BandFromLevels], title: str = "Level Schemes"):
    fig, ax = plt.subplots(figsize=(10, 6))

    for band_idx, band in enumerate(bands):
        nodes = band.nodes
        labels = band.labels
        edges = band.edges

        # Shift band horizontally
        x_offset = band_idx * x_spacing
        sorted_levels = sorted(nodes.items(), key=lambda x: x[1], reverse=False)

        # Draw levels
        for i, (label, energy) in enumerate(sorted_levels):
            ax.hlines(energy, x_offset, x_offset + 1, color='black')
            ax.text(x_offset - 0.2, energy, labels[label],
                    va='center', ha='right', fontsize=9)

        # Draw arrows for transitions
        for src, dst in edges:
            y1, y2 = nodes[src], nodes[dst]
            if y1 < y2:
                y1, y2 = y2, y1
            x = x_offset + 0.5
            ax.annotate("", xy=(x, y2), xytext=(x, y1),
                        arrowprops=dict(arrowstyle='->', color='blue'))
            energy_diff = abs(nodes[src] - nodes[dst])
            y_mid = (y1 + y2) / 2
            ax.text(x + 0.1, y_mid, f"{energy_diff:.1f}",
                    va='center', fontsize=8, color='blue')

        # Label the band
        ax.text(x_offset + 0.5, -20, f"Band {band_idx + 1}",
                ha='center', va='top', fontsize=10, fontweight='bold')

    # Formatting
    ax.set_ylabel("Energy (keV)")
    ax.set_title(title)
    ax.set_ylim(0, max(max(b.nodes.values()) for b in bands) + y_margin)
    ax.set_xlim(-1, len(bands) * x_spacing)
    ax.set_xticks([])   # remove x-axis ticks
    ax.set_xlabel("")   # remove x-axis label
    ax.grid(False)

    plt.tight_layout()
    plt.show()


# Read the csv file
def read_file(csv_path: str) -> List[Tuple[str, float]]:
    df = pd.read_csv(csv_path)
    df.columns = [col.strip() for col in df.columns]

    energy_col = "E(level)(keV)"
    jpi_col = "JPi(level)"

    if energy_col not in df.columns or jpi_col not in df.columns:
        raise ValueError(f"CSV must contain '{energy_col}' and '{jpi_col}' columns.")

    # Parse energies and spin-parity
    def extract_energy(val):
        try:
            return float(str(val).strip().split()[0])
        except:
            return np.nan

    df["Energy"] = df[energy_col].apply(extract_energy)
    df["Jpi"] = df[jpi_col].astype(str)

    df = df.dropna(subset=["Energy", "Jpi"])

    levels = list(zip(df["Jpi"], df["Energy"]))

    if not levels:
        raise ValueError("No valid levels found in file.")

    return levels


# Main !!!
if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Plot nuclear level schemes from CSV.")
    parser.add_argument("csv_file", nargs='?', help="CSV input file")
    parser.add_argument("--bands", type=int, default=3, help="Number of bands to split into")
    args = parser.parse_args()

    if not args.csv_file or not os.path.isfile(args.csv_file):
        print("Please provide a valid CSV file path.")
    else:
        try:
            levels = read_file(args.csv_file)
            bands = split_into_bands(levels, n_bands=args.bands)
            plot_multiple_bands(bands, title="Multiple-Band Level Scheme")
        except Exception as e:
            print(f"Error: {e}")
