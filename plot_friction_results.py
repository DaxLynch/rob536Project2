import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


def plot_friction_success(csv_path: str, label: str = None, fig=None, ax=None):
    """
    Plot friction coefficient vs success rate from a CSV file.
    
    Args:
        csv_path: Path to the CSV file
        label: Label for this line in the legend
        fig: Optional existing figure to plot on
        ax: Optional existing axes to plot on
    
    Returns:
        fig, ax: The figure and axes objects
    """
    df = pd.read_csv(csv_path)
    
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_xlabel('Friction Coefficient', fontsize=12)
        ax.set_ylabel('Success Rate (%)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 105)
    
    line_label = label if label else Path(csv_path).stem
    ax.plot(df['friction'], df['success_rate'], marker='o', linewidth=2, markersize=8, label=line_label)
    
    return fig, ax


def plot_multiple(csv_label_pairs: list, title: str = None, save_path: str = None, show: bool = True):
    """
    Plot multiple CSV files on the same axes.
    
    Args:
        csv_label_pairs: List of (csv_path, label) tuples
        title: Title for the plot
        save_path: Optional path to save the figure
        show: Whether to call plt.show()
    
    Returns:
        fig, ax: The figure and axes objects
    """
    fig, ax = None, None
    
    for csv_path, label in csv_label_pairs:
        fig, ax = plot_friction_success(csv_path, label=label, fig=fig, ax=ax)
    
    if title:
        ax.set_title(title, fontsize=14)
    
    ax.legend(loc='best')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved plot to {save_path}")
    
    if show:
        plt.show()
    
    return fig, ax


if __name__ == "__main__":
    # Example: Plot multiple CSVs
    pairs = [
        ("friction_eval_results/vanillaModelVaryingFriction.csv", "Friction Unaware"),
        ("friction_eval_results/frictionAwareModelVaryingFriction.csv", "Friction Aware"),
    ]
    plot_multiple(pairs, title="Model Comparison - Success Rate vs Friction")
    
    # Or chain calls manually:
    # fig, ax = plot_friction_success("friction_eval_results/file1.csv", label="Model 1")
    # fig, ax = plot_friction_success("friction_eval_results/file2.csv", label="Model 2", fig=fig, ax=ax)
    # ax.legend()
    # plt.show()

