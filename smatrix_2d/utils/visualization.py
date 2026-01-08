"""Simple visualization utilities for dose distributions."""

import numpy as np
import matplotlib.pyplot as plt


def plot_dose_map(
    dose: np.ndarray,
    x_grid: np.ndarray,
    z_grid: np.ndarray,
    title: str = 'Dose Distribution',
    save_path: str = None,
):
    """Create 2D dose heatmap visualization.

    Args:
        dose: Dose distribution [Nz, Nx]
        x_grid: X coordinates [mm]
        z_grid: Z coordinates [mm]
        title: Plot title
        save_path: If provided, save to file
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    im = ax.imshow(
        dose.T,  # Transpose to match x-z orientation
        origin='lower',
        aspect='auto',
        extent=[
            x_grid[0], x_grid[-1],
            z_grid[0], z_grid[-1],
        ],
        cmap='viridis',
    )

    plt.colorbar(im, ax=ax, label='Dose [MeV]')
    ax.set_xlabel('x [mm]')
    ax.set_ylabel('z [mm]')
    ax.set_title(title)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_depth_dose(
    dose: np.ndarray,
    z_grid: np.ndarray,
    title: str = 'Depth-Dose Curve',
    save_path: str = None,
):
    """Plot integrated dose vs depth.

    Args:
        dose: Dose distribution [Nz, Nx]
        z_grid: Z coordinates [mm]
        title: Plot title
        save_path: If provided, save to file
    """
    depth_dose = np.sum(dose, axis=1)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(z_grid, depth_dose, linewidth=2)
    ax.set_xlabel('Depth z [mm]')
    ax.set_ylabel('Dose [MeV]')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_lateral_profile(
    dose: np.ndarray,
    x_grid: np.ndarray,
    z_grid: np.ndarray,
    z_peak: float,
    title: str = 'Lateral Profile at Bragg Peak',
    save_path: str = None,
):
    """Plot lateral dose profile at depth of maximum dose.

    Args:
        dose: Dose distribution [Nz, Nx]
        x_grid: X coordinates [mm]
        z_grid: Z coordinates [mm]
        z_peak: Depth of maximum dose [mm]
        title: Plot title
        save_path: If provided, save to file
    """
    iz_peak = np.argmin(np.abs(z_grid - z_peak))
    lateral_profile = dose[iz_peak, :]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(x_grid, lateral_profile, linewidth=2)
    ax.set_xlabel('x [mm]')
    ax.set_ylabel('Dose [MeV]')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()

    plt.close()
