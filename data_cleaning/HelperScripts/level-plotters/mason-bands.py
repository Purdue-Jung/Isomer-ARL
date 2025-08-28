#bands is the dictionary
 
def plot_level_bands(bands, spacing=3, line_length=0.5, label_offset=0.6, energy_offset=0.3):
    plt.figure(figsize=(spacing * len(bands), 10))  # increased vertical size
    ax = plt.gca()
 
    for i, band in enumerate(bands):
        x_offset = i * spacing
        nodes = band["nodes"]
        edges = band["edges"]
 
        for label, energy in nodes.items():

            ax.hlines(y=energy,
                      xmin=x_offset - line_length / 2,
                      xmax=x_offset + line_length / 2,
                      color='black', linewidth=2)
 
            # labels
            ax.text(x_offset + label_offset, energy, label,
                    verticalalignment='center', horizontalalignment='left',
                    fontsize=11, fontweight='bold')
 
        # transitions
        for u, v in edges:
            y_start = nodes[u]
            y_end = nodes[v]
            mid_y = (y_start + y_end) / 2
            delta_E = y_start - y_end
 
            ax.annotate("",

                        xy=(x_offset, y_end),
                        xytext=(x_offset, y_start),
                        arrowprops=dict(arrowstyle="->", lw=1.5))
 
            ax.text(x_offset - energy_offset, mid_y, f"{delta_E:.1f} keV",

                    fontsize=9, va='center', ha='right')
 
        # band label

        top_energy = max(nodes.values())

        ax.text(x_offset, top_energy + 100, band["name"],
                ha='center', va='bottom', fontsize=13, fontweight='bold')
 
    # formatting

    max_energy = max(max(b["nodes"].values()) for b in bands)

    ax.set_ylim(-50, max_energy + 200)  # more space on top

    ax.set_xlim(-1, len(bands) * spacing)

    ax.set_title("Level Scheme", fontsize=16, pad=40)  # padded title

    ax.axis('off')

    plt.show()  # removed tight_layout to avoid clipping title
 