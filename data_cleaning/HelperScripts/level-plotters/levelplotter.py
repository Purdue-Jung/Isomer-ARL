import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#open and read the ENSDF .csv file 
def parse_ensdf_csv(csv_file):
    df = pd.read_csv(csv_file)
    df.columns = [col.strip() for col in df.columns]  #clean headers

    def extract_float(value):
        try:
            return float(str(value).split()[0])
        except:
            return np.nan

    df['E_level_keV'] = df['E(level)(keV)'].apply(extract_float)
    df['E_gamma_keV'] = df['E()(keV)'].apply(extract_float)
    df['Intensity'] = df['I()'].apply(extract_float)
    df['E_final_keV'] = df['E_level_keV'] - df['E_gamma_keV']
    return df

#plots level scheme
def plot_level_scheme(df, title="Nuclear Level Scheme", output_file=None):
    fig, ax = plt.subplots(figsize=(10, 6))

    unique_levels = sorted(df['E_level_keV'].dropna().unique())
    x_center = 0
    level_width = 0.6

    for energy in unique_levels:
        ax.hlines(energy, x_center - level_width/2, x_center + level_width/2, color='black', linewidth=2)

    for _, row in df.iterrows():
        if not np.isnan(row['E_gamma_keV']) and not np.isnan(row['E_final_keV']):
            y_start = row['E_level_keV']
            y_end = row['E_final_keV']
            intensity = row['Intensity'] if not np.isnan(row['Intensity']) else 10
            linewidth = 0.5 + 3 * (intensity / 100)

            ax.annotate(
                '', xy=(x_center, y_end), xytext=(x_center, y_start),
                arrowprops=dict(facecolor='blue', width=linewidth, headwidth=6, alpha=0.7)
            )
    #Organizes when energy gets in weird levels 
    for energy in unique_levels:
        jpi_rows = df[df['E_level_keV'] == energy]['JPi(level)'].dropna().unique()
        if len(jpi_rows) > 0:
            jpi_label = jpi_rows[0]
            ax.text(x_center + 0.4, energy, jpi_label, fontsize=10, va='center')

    ax.set_ylim(-10, max(unique_levels) + 20)
    ax.set_xlim(-1, 1.5)
    ax.set_xlabel("Arbitrary x-position")
    ax.set_ylabel("Energy (keV)")
    ax.set_title(title)
    ax.get_xaxis().set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.tight_layout()
    if output_file:
        plt.savefig(output_file)
    else:
        plt.show()

#Main !!!
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot nuclear level scheme from ENSDF-style CSV.")
    parser.add_argument("csv_file", help="Path to CSV file with ENSDF decay data")
    parser.add_argument("--output", help="Optional output image file", default=None)

    args = parser.parse_args()
    df = parse_ensdf_csv(args.csv_file)
    plot_level_scheme(df, title="Hf-173 Level Scheme", output_file=args.output)
