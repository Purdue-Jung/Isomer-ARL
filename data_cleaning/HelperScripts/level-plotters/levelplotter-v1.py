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

#open and read ENSDF .csv file 
def read_file(csv_path: str, max_bands: int = 3) -> List[Dict]:
    df = pd.read_csv(csv_path)
    df.columns = [col.strip() for col in df.columns]

    # We know the column format from your example
    energy_col = "E()(keV)"
    band_col = "BandNum"

    # Check required columns exist
    if energy_col not in df.columns:
        raise ValueError(f"Missing required column: {energy_col}")
    if band_col not in df.columns:
        raise ValueError(f"Missing required column: {band_col}")

    # Extract the numeric part of gamma energy (some cells have multiple numbers like "11.9 2")
    def extract_energy(val):
        try:
            return float(str(val).split()[0])  # take first number only
        except:
            return np.nan

    df['E_gamma'] = df[energy_col].apply(extract_energy)
    df = df.dropna(subset=['E_gamma', band_col])  # Drop invalid rows

    # Convert BandNum to integer for grouping
    df['BandNum'] = df['BandNum'].astype(int)

    # Group by band and process
    band_groups = df.groupby('BandNum')
    bands = []

    for i, (band_id, group) in enumerate(band_groups):
        if i >= max_bands:
            break
        gamma_list = group['E_gamma'].tolist()
        band = Band(band_id=i, gamma_energies=gamma_list, branching_prob=0.4)
        bands.append({
            "name": f"Band {i}",
            "nodes": band.nodes,
            "edges": band.edges
        })

    return bands

'''
def read_file(csv_path: str, max_bands: int = 3) -> List[Dict]:
    df = pd.read_csv(csv_path)
    df.columns = [col.strip() for col in df.columns]

    # Accept flexible naming
    possible_energy_columns = ['E_gamma', 'E()(keV)', 'Egamma', 'Energy (keV)']
    energy_col = next((col for col in df.columns if col in possible_energy_columns), None)
    if not energy_col:
        raise ValueError(f"Could not find a gamma energy column. Expected one of: {possible_energy_columns}")

    if 'BandNum' not in df.columns:
        raise ValueError("CSV must contain a 'BandNum' column to group transitions into bands.")

    def extract_float(val):
        try:
            return float(str(val).split()[0])
        except:
            return np.nan

    df['E_gamma'] = df[energy_col].apply(extract_float)
    df = df.dropna(subset=['E_gamma'])

    band_groups = df.groupby('BandNum')

    bands = []
    for i, (band_id, group) in enumerate(band_groups):
        if i >= max_bands:
            break
        gamma_list = group['E_gamma'].tolist()
        band = Band(band_id=i, gamma_energies=gamma_list, branching_prob=0.4)
        bands.append({
            "name": f"Band {i}",
            "nodes": band.nodes,
            "edges": band.edges
        })
    return bands
'''
#Converting ENSDF gamma energies to level structure
class Band:
    def __init__(self, band_id: int, gamma_energies: List[float], branching_prob: float = 0.3):
        self.band_id = band_id
        self.levels = self.linear_gamma_to_levels(gamma_energies)
        self.nodes = {f'L{band_id}_{i}': lvl for i, lvl in enumerate(self.levels)}
        self.edges = self.generate_edges(branching_prob)

    @staticmethod
    def linear_gamma_to_levels(gamma_energies: List[float]) -> List[float]:
        E_top = sum(gamma_energies)
        return [E_top - sum(gamma_energies[:i]) for i in range(len(gamma_energies) + 1)]

    def generate_edges(self, branching_prob: float) -> List[Tuple[str, str]]:
        edges = []
        n = len(self.levels)
        keys = list(self.nodes.keys())
        for i in range(1, n):
            edges.append((keys[i], keys[i - 1]))
            for j in range(i - 2, -1, -1):
                if np.random.rand() < branching_prob:
                    edges.append((keys[i], keys[j]))
        return edges

#Ladder generator (not used directly here, kept for optional simulation)
class Ladder:
    def __init__(self, max_rungs: int = 3, spacing: int = 350, jitter: int = 15):
        self.max_rungs = max_rungs
        self.spacing = spacing
        self.jitter = jitter
        self.positions = []

    def generate_ladder(self):
        self.positions = [0]
        for _ in range(self.max_rungs):
            step = self.spacing + np.random.randint(-self.jitter, self.jitter)
            self.positions.append(self.positions[-1] + step)

#Adds inter-band transitions
def add_interband_branching(bands: List[Dict], max_connections: int = 5, prob: float = 0.3) -> List[Tuple[Tuple[int, str], Tuple[int, str]]]:
    cross_edges = []
    for i, band_a in enumerate(bands):
        for j, band_b in enumerate(bands):
            if i >= j:
                continue
            for src_lbl, src_E in band_a["nodes"].items():
                for tgt_lbl, tgt_E in band_b["nodes"].items():
                    if src_E > tgt_E and np.random.rand() < prob:
                        cross_edges.append(((i, src_lbl), (j, tgt_lbl)))
                        if len(cross_edges) >= max_connections:
                            return cross_edges
    return cross_edges

#Plotting function for band-based level scheme
def plot_level_bands(bands: List[Dict], cross_edges: List[Tuple[Tuple[int, str], Tuple[int, str]]] = None, title: str = "Nuclear Level Scheme"):
    fig, ax = plt.subplots(figsize=(10, 6))
    y_margin = 50
    x_spacing = 3.0

    for i, band in enumerate(bands):
        nodes = band["nodes"]
        edges = band["edges"]
        x_offset = i * x_spacing

        ax.text(x_offset + 0.4, -y_margin, f"Band {i + 1}", ha='center', fontsize=11, weight='bold')

        sorted_levels = sorted(nodes.items(), key=lambda x: x[1])
        for level_idx, (label, energy) in enumerate(sorted_levels):
            ax.hlines(energy, x_offset, x_offset + 1, color='black')
            ax.text(x_offset - 0.2, energy, f"b{i+1} lvl {level_idx}", va='center', ha='right', fontsize=9)

        for src, dst in edges:
            y1, y2 = nodes[src], nodes[dst]
            if y1 < y2:
                y1, y2 = y2, y1
            x = x_offset + 0.5
            ax.annotate("", xy=(x, y2), xytext=(x, y1), arrowprops=dict(arrowstyle='->', color='blue'))

        for src, dst in edges:
            if src in nodes and dst in nodes:
                y1, y2 = nodes[src], nodes[dst]
                energy_diff = abs(y1 - y2)
                x = x_offset + 0.6
                y_mid = (y1 + y2) / 2
                ax.text(x, y_mid, f"{energy_diff:.1f}", va='center', fontsize=8, color='blue')

    if cross_edges:
        for (i, src), (j, dst) in cross_edges:
            x1 = i * x_spacing + 0.5
            x2 = j * x_spacing + 0.5
            y1 = bands[i]["nodes"][src]
            y2 = bands[j]["nodes"][dst]
            ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle='->', color='red', linestyle='dashed'))

    ax.set_xlabel("Bands")
    ax.set_ylabel("Energy (keV)")
    ax.set_title(title)
    ax.set_xlim(-1, len(bands) * x_spacing)
    ax.set_ylim(0, max([max(band["nodes"].values()) for band in bands]) + y_margin)
    ax.grid(False)
    plt.tight_layout()
    plt.show()

'''
# Parse the ENSDF-style CSV
def parse_ensdf_to_bands(csv_path: str, max_bands: int = 3) -> List[Dict]:
    df = pd.read_csv(csv_path)
    df.columns = [col.strip() for col in df.columns]

    def extract_float(val):
        try:
            return float(str(val).split()[0])
        except:
            return np.nan

    df['E_gamma'] = df['E()(keV)'].apply(extract_float)
    df = df.dropna(subset=['E_gamma'])

    band_groups = df.groupby('BandNum')

    bands = []
    for i, (band_id, group) in enumerate(band_groups):
        if i >= max_bands:
            break
        gamma_list = group['E_gamma'].tolist()
        band = Band(band_id=i, gamma_energies=gamma_list, branching_prob=0.4)
        bands.append({
            "name": f"Band {i}",
            "nodes": band.nodes,
            "edges": band.edges
        })
    return bands
'''

#Entry point
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Plot level scheme from ENSDF CSV using Band structure.")
    parser.add_argument("csv_file", help="CSV input file")
    args = parser.parse_args()

    bands = read_file(args.csv_file, max_bands=3)
    cross = add_interband_branching(bands, max_connections=6, prob=0.3)
    plot_level_bands(bands, cross_edges=cross, title="Hf-173 Realistic Level Scheme")
