import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import random

#CONSTANTS
CALIB_OFFSET = 1.785
CALIB_SLOPE = 0.99943
DEFAULT_MATRIX_SIZE = 4096
energy_axis = CALIB_OFFSET + CALIB_SLOPE * np.arange(DEFAULT_MATRIX_SIZE)
y_margin = 50
x_spacing = 3.0

#Generates Bands from E(level)(keV)
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

#Plots all levels into one band for now
def plot_single_band(band: BandFromLevels, title: str = "Level Scheme"):
    fig, ax = plt.subplots(figsize=(6, 6))
    nodes = band.nodes
    labels = band.labels
    edges = band.edges

    x_offset = 1.0
    sorted_levels = sorted(nodes.items(), key=lambda x: x[1], reverse=False)

    for i, (label, energy) in enumerate(sorted_levels):
        ax.hlines(energy, x_offset, x_offset + 1, color='black')
        ax.text(x_offset - 0.2, energy, labels[label], va='center', ha='right', fontsize=9)

    for src, dst in edges:
        y1, y2 = nodes[src], nodes[dst]
        if y1 < y2:
            y1, y2 = y2, y1
        x = x_offset + 0.5
        ax.annotate("", xy=(x, y2), xytext=(x, y1), arrowprops=dict(arrowstyle='->', color='blue'))

        energy_diff = abs(nodes[src] - nodes[dst])
        y_mid = (y1 + y2) / 2
        ax.text(x + 0.1, y_mid, f"{energy_diff:.1f}", va='center', fontsize=8, color='blue')

    ax.set_xlabel("Single Band")
    ax.set_ylabel("Energy (keV)")
    ax.set_title(title)

    all_energies = list(nodes.values())
    ax.set_ylim(0, max(all_energies) + y_margin)
    ax.set_xlim(0, 3)
    ax.grid(False)
    plt.tight_layout()
    plt.show()

#Read the csv file
def read_file(csv_path: str) -> BandFromLevels:
    df = pd.read_csv(csv_path)
    df.columns = [col.strip() for col in df.columns]

    energy_col = "E(level)(keV)"
    jpi_col = "JPi(level)"

    if energy_col not in df.columns or jpi_col not in df.columns:
        raise ValueError(f"CSV must contain '{energy_col}' and '{jpi_col}' columns.")

    #Parse energies and spin-parity
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

    return BandFromLevels(levels, branching_prob=0.3)


#Main !!!
if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Plot single nuclear level scheme from CSV.")
    parser.add_argument("csv_file", nargs='?', help="CSV input file")
    args = parser.parse_args()

    if not args.csv_file or not os.path.isfile(args.csv_file):
        print("Please provide a valid CSV file path.")
    else:
        try:
            band = read_file(args.csv_file)
            plot_single_band(band, title="Single-Band Level Scheme")
        except Exception as e:
            print(f"Error: {e}")
