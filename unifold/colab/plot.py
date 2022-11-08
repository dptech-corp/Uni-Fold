# Construct multiclass b-factors to indicate confidence bands
# 0=very low, 1=low, 2=confident, 3=very high
# Color bands for visualizing plddt
import os
import numpy as np
import py3Dmol
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from IPython import display
from ipywidgets import GridspecLayout
from ipywidgets import Output
from typing import *
from unifold.data import protein


def colab_plot(
    best_result: Mapping[str, Any],
    output_dir: str,
    show_sidechains: bool = False,
    dpi: int = 100,
):
    best_protein = best_result["protein"]
    best_plddt = best_result["plddt"]
    best_pae = best_result.get("pae", None)
    
    to_visualize_pdb = protein.to_pdb(best_protein)

    # --- Visualise the prediction & confidence ---
    if best_pae is not None:
        multichain_view = py3Dmol.view(width=800, height=600)
        multichain_view.addModelsAsFrames(to_visualize_pdb)
        multichain_style = {'cartoon': {'colorscheme': 'chain'}}
        multichain_view.setStyle({'model': -1}, multichain_style)
        multichain_view.zoomTo()
        multichain_view.show()

    # Color the structure by per-residue pLDDT
    view = py3Dmol.view(width=800, height=600)
    view.addModelsAsFrames(to_visualize_pdb)
    style = {'cartoon': {'colorscheme': {'prop':'b','gradient': 'roygb','min':0.5,'max':0.9}}}
    if show_sidechains:
        style['stick'] = {}
    view.setStyle({'model':-1}, style)
    view.zoomTo()

    grid = GridspecLayout(1, 2)
    out = Output()
    with out:
        view.show()
    grid[0, 0] = out

    out = Output()
    with out:
        plot_plddt_legend().show()
    grid[0, 1] = out

    display.display(grid)

    # Display pLDDT and predicted aligned error (if output by the model).
    num_plots = 1 if best_pae is None else 2

    plt.figure(figsize=[8 * num_plots , 6])
    plt.subplot(1, num_plots, 1)
    plt.plot(best_plddt * 100)
    plt.title('Predicted LDDT')
    plt.xlabel('Residue')
    plt.ylabel('pLDDT')
    plt.grid()
    plddt_svg_path = os.path.join(output_dir, 'plddt.svg')
    plt.savefig(plddt_svg_path, dpi=dpi, bbox_inches='tight')

    if best_pae is not None:
        plt.subplot(1, 2, 2)
        max_pae = np.max(best_pae)
        colors = ['#0F006F','#245AE6','#55CCFF','#FFFFFF']

        cmap = LinearSegmentedColormap.from_list('mymap', colors)
        im = plt.imshow(best_pae, vmin=0., vmax=max_pae, cmap=cmap)
        plt.colorbar(im, fraction=0.046, pad=0.04)

        # Display lines at chain boundaries.
        total_num_res = best_protein.residue_index.shape[-1]
        chain_ids = best_protein.chain_index
        for chain_boundary in np.nonzero(chain_ids[:-1] - chain_ids[1:]):
            if chain_boundary.size:
                plt.plot([0, total_num_res], [chain_boundary, chain_boundary], color='red')
                plt.plot([chain_boundary, chain_boundary], [0, total_num_res], color='red')

        plt.title('Predicted Aligned Error')
        plt.xlabel('Scored residue')
        plt.ylabel('Aligned residue')
        pae_svg_path = os.path.join(output_dir, 'pae.svg')
        plt.savefig(pae_svg_path, dpi=dpi, bbox_inches='tight')


PLDDT_BANDS = [(0., 0.50, '#FF7D45'),
            (0.50, 0.70, '#FFDB13'),
            (0.70, 0.90, '#65CBF3'),
            (0.90, 1.00, '#0053D6')]


def plot_plddt_legend():
    """Plots the legend for pLDDT."""
    thresh = ['Very low (pLDDT < 50)',
            'Low (70 > pLDDT > 50)',
            'Confident (90 > pLDDT > 70)',
            'Very high (pLDDT > 90)']

    colors = [x[2] for x in PLDDT_BANDS]

    plt.figure(figsize=(2, 2))
    for c in colors:
        plt.bar(0, 0, color=c)
    plt.legend(thresh, frameon=False, loc='center', fontsize=20)
    plt.xticks([])
    plt.yticks([])
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    plt.title('Model Confidence', fontsize=20, pad=20)
    return plt
