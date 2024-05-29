from typing import Callable, List, Optional, Union

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from .metrics import compare_images, compare_signals


def plot_mat(
    ccmat: Union[np.ndarray, torch.Tensor], transform: list[Callable] = None, cmap=""
) -> None:
    # check if some functions were passed
    if transform:
        # if only one function were passsed -> turn it to list
        if type(transform) is not list:
            transform = [transform]
        # apply iterativly to ccmat
        for func in transform:
            ccmat = func(ccmat)

    plt.imshow(ccmat, cmap="hot", interpolation="nearest")


def plot_comparison(
    mat1: np.ndarray,
    mat2: np.ndarray,
    mat3: np.ndarray,
    ax_1_title: str = "",
    ax_2_title: str = "",
    ax_3_title: str = "",
    gen_pos: Optional[dict] = None,  # {"CHR": "...", "START": "...", "STOP": "..."}
    compare: bool = False,
    transform: Optional[List[Callable]] = None,
    save_path: Optional[str] = None,
    cmap: str = "afmhot",
    figsize: tuple[int] = (12, 5),
) -> None:
    """
    Plot a comparison between three matrices, potentially applying a transformation function to each.
    This can be used for visualizing and comparing images/matrices side by side.

    Args:
    - mat1 (np.ndarray): The first matrix or image.
    - mat2 (np.ndarray): The second matrix or image.
    - mat3 (np.ndarray): The third matrix or image.
    - ax_1_title (str): Title for the first subplot.
    - ax_2_title (str): Title for the second subplot.
    - ax_3_title (str): Title for the third subplot.
    - compare (bool): If True, compute comparison metrics between the images.
    - transform (list[Callable], optional): A list of functions to apply to the matrices.
    - save_path (str, optional): If provided, the path where to save the figure.
    """

    # Apply transformations to each matrix if any transformation functions are provided
    if transform:
        # Ensure transform is a list even if a single function is passed
        transform = transform if isinstance(transform, list) else [transform]
        # Apply each function to the matrices
        for func in transform:
            mat1, mat2, mat3 = func(mat1), func(mat2), func(mat3)

    # Create a figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
    if gen_pos:
        y_axis_text = f"{gen_pos['CHR']}: {gen_pos['START']}-{gen_pos['STOP']}"
        fig.text(0.1, 0.5, y_axis_text, va="center", rotation="vertical")
    # Adjust labelpad for spacing between the figures and the x-labels
    ax1.xaxis.labelpad = 10
    ax2.xaxis.labelpad = 10
    ax3.xaxis.labelpad = 10

    # Add title
    ax1.set_title(ax_1_title)
    ax2.set_title(ax_2_title)
    ax3.set_title(ax_3_title)

    # remove ticks
    ax1.xaxis.set_major_locator(ticker.NullLocator())
    ax2.xaxis.set_major_locator(ticker.NullLocator())
    ax3.xaxis.set_major_locator(ticker.NullLocator())
    ax1.yaxis.set_major_locator(ticker.NullLocator())
    ax2.yaxis.set_major_locator(ticker.NullLocator())
    ax3.yaxis.set_major_locator(ticker.NullLocator())

    # Compare matrices and add comparison metrics to the subplot titles if comparison is requested
    if compare:
        mtr_hr_enhanced = compare_images(
            mat2, mat3
        )  # Assumes comparison function returns metrics
        mtr_lr_hr = compare_images(mat1, mat2)
        ax3.set_xlabel(
            f"PSNR: {mtr_hr_enhanced[0]:.3f}, MSE: {mtr_hr_enhanced[1]:.3f}, SSIM: {mtr_hr_enhanced[3]:.3f}"
        )
        ax1.set_xlabel(
            f"PSNR: {mtr_lr_hr[0]:.3f}, MSE: {mtr_lr_hr[1]:.3f}, SSIM: {mtr_lr_hr[3]:.3f}"
        )
        ax2.set_xlabel("Reference")

    # Display each matrix on its respective axes
    ax1.imshow(mat1, cmap=cmap, interpolation="nearest")
    ax2.imshow(mat2, cmap=cmap, interpolation="nearest")
    ax3.imshow(mat3, cmap=cmap, interpolation="nearest")

    # Save the figure if a save path is provided
    if save_path:
        plt.savefig(save_path)
        plt.close(fig)  # Close the figure to free up memory
    else:
        plt.show()  # Or display the figure in a window


def test_numpy_plot() -> None:
    test_array = np.random.rand(48, 48)
    plot_mat(test_array, [test_transform_1, test_transform_2])


def test_transform_1(mat: np.ndarray) -> np.ndarray:
    return mat + 1


def test_transform_2(mat: np.ndarray) -> np.ndarray:
    return mat / 2


def plot_matrices(
    mat1: np.ndarray,
    mat2: np.ndarray,
    chromosome: str,
    start_pos: int,
    resolution: int,
    tick_interval: int = 50,
    ax=None,
    diag: bool = False,
    figsize: tuple[int, int] = (10, 10),
    cmap: str = "afmhot",
    cbar: bool = False,
    only_outer_ticks: bool = False,
    tick_label_size: int = 10,
    text_upper_right=None,
    text_lower_left=None,
    fontdict={"family": "monospace", "color": "white", "weight": "normal", "size": 16},
    save_path=None,
):
    """
    Plots two matrices side by side with the upper triangle representing 'a' and the lower triangle representing 'b',
    and includes a cleaner genomic coordinate axis at the bottom styled with Seaborn.

    Parameters:
        a (np.ndarray): The first matrix to be plotted in the upper triangle.
        b (np.ndarray): The second matrix to be plotted in the lower triangle.
        chromosome (str): The chromosome number to label the axis.
        start_pos (int): The starting position of the genomic data.
        resolution (int): The resolution of the bins in base pairs.
        figsize (tuple[int, int], optional): Figure size (width, height). Default is (10, 10).
        cmap (str, optional): Colormap to use for the plot. Default is 'afmhot'.
        only_outer_ticks (bool, optional): If True, only show ticks at the outer edges. Default is False.
        tick_label_size (int): Size of the tick labels. Default is 10.
        save_path (str, optional): If provided, the plot will be saved to this file path. Default is None.

    Returns:
        None
    """
    # Set the Seaborn style
    sns.set(style="white")
    b = mat2.copy()
    # get for diag length
    num_bins = mat1.shape[0]

    # Create a mask to make the lower triangle of 'a' transparent
    mask = np.triu(np.ones_like(mat1, dtype=bool))

    # Create a new figure with the specified size
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Plot the upper triangle (matrix 'a') using the specified colormap
    cax = ax.matshow(mat1, cmap=cmap)

    # Set the lower triangle (matrix 'b') by masking it with NaN values
    b[mask] = np.nan
    ax.matshow(b, cmap=cmap)

    # Calculate the positions for the genomic coordinates
    num_bins = mat1.shape[0]

    # Include the last bin position manually
    tick_positions = np.arange(0, num_bins, tick_interval)
    if num_bins - 1 not in tick_positions:
        tick_positions = np.append(tick_positions, num_bins - 1)

    # Create tick labels, including the label for the last position
    if only_outer_ticks:
        tick_labels = [
            ""
            if i != 0 and i != num_bins - 1
            else f"{(start_pos + i * resolution)/1_000_000:,.1f}"
            for i in tick_positions
        ]
    else:
        tick_labels = [
            f"{(start_pos + i * resolution)/1_000_000:,.1f}" for i in tick_positions
        ]

    # Append 'Mb' to the last tick label
    tick_labels[-1] += " Mb"

    # Update the axis with genomic coordinate labels
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="center")
    ax.set_yticks([])  # Hide y-axis ticks
    ax.xaxis.set_ticks_position("bottom")  # Show ticks only at the bottom

    # Set the tick label size
    ax.tick_params(axis="x", labelsize=tick_label_size)

    # Create a floating axis appearance
    ax.spines["bottom"].set_position(("outward", 10))
    sns.despine(
        ax=ax, offset=10, trim=True, left=True
    )  # Apply Seaborn despine to make it look better

    # Set the chromosome label
    ax.set_xlabel(f"{chromosome}")

    # Add colorbar if required
    if cbar:
        # Create an axis for the colorbar
        cbar_ax = fig.add_axes(
            [
                ax.get_position().x1 + 0.01,
                ax.get_position().y0,
                0.02,
                ax.get_position().height,
            ]
        )
        fig.colorbar(cax, cax=cbar_ax)
        cbar_ax.tick_params(labelsize=tick_label_size)

    # Add optional text in the upper right corner
    if text_upper_right:
        ax.text(
            0.95,
            0.95,
            text_upper_right,
            transform=ax.transAxes,
            horizontalalignment="right",
            verticalalignment="top",
            fontdict=fontdict,
        )

    # Add optional text in the lower left corner
    if text_lower_left:
        ax.text(
            0.05,
            0.05,
            text_lower_left,
            transform=ax.transAxes,
            horizontalalignment="left",
            verticalalignment="bottom",
            fontdict=fontdict,
        )
    if diag:
        ax.plot(
            [0, num_bins - 1],
            [0, num_bins - 1],
            linestyle="--",
            color="grey",
            linewidth=1,
        )
    if save_path:
        print(f"Saving Plot: {save_path}")
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close()
    else:
        # Otherwise, display the plot
        plt.show()


def plot_insulation_scores(
    insulation_scores_pred,
    insulation_scores_lr,
    insulation_scores_hr,
    start_pos,
    resolution=10_000,
    figsize=(20, 6),
    save_path=None,
):
    """
    Plot insulation scores for different resolutions with genomic coordinates.

    Args:
    - insulation_scores_pred (np.ndarray): Insulation scores for the prediction.
    - insulation_scores_lr (np.ndarray): Insulation scores for low resolution.
    - insulation_scores_hr (np.ndarray): Insulation scores for high resolution.
    - start_pos (int): Starting genomic position.
    - resolution (int): The resolution of the genomic data.
    - figsize (tuple): Figure size.
    """

    plt.figure(figsize=figsize)
    genomic_end = insulation_scores_hr.shape[0] - 50
    indices = np.arange(0, genomic_end)  # Assuming these are your data indices
    x_values = (
        indices * resolution + start_pos
    ) / 1_000_000  # Converting indices to genomic coordinates

    plt.plot(
        x_values,
        insulation_scores_pred[0:genomic_end],
        label="Insulation Score Prediction",
    )
    plt.plot(
        x_values,
        insulation_scores_lr[0:genomic_end],
        label="Insulation Score Low Resolution",
    )
    plt.plot(
        x_values,
        insulation_scores_hr[0:genomic_end],
        label="Insulation Score High Resolution",
    )
    plt.xlabel("Genomic Coordinates [Mbp]")
    plt.ylabel("Insulation Score")

    # Overlay for a specific genomic region, adjust start and end indices as needed
    start_index = 524
    end_index = 562
    start_genomic = (start_index * resolution + start_pos) / 1_000_000
    end_genomic = (end_index * resolution + start_pos) / 1_000_000
    plt.axvspan(start_genomic, end_genomic, color="grey", alpha=0.5)
    plt.text(
        end_genomic + 4.3,
        -0.01,
        "Centromer Region",
        horizontalalignment="center",
        verticalalignment="center",
        fontsize=12,
        color="grey",
    )

    plt.legend()
    # Save the figure if a save path is provided
    if save_path:
        print(f"Saving plot to: {save_path}")
        plt.savefig(save_path, dpi=600)
        plt.close()  # Close the figure to free up memory
    else:
        plt.show()  # Or display the figure in a window

def plot_insulation_scores_with_metrics(insulation_scores_pred, insulation_scores_lr, insulation_scores_hr, start_pos, resolution=10_000, figsize=(24, 6), save_path=None):
    """
    Plot insulation scores for different resolutions with genomic coordinates and include a 2x2 table of metrics.

    Args:
    - insulation_scores_pred (np.ndarray): Insulation scores for the prediction.
    - insulation_scores_lr (np.ndarray): Insulation scores for low resolution.
    - insulation_scores_hr (np.ndarray): Insulation scores for high resolution.
    - start_pos (int): Starting genomic position.
    - resolution (int): The resolution of the genomic data.
    - figsize (tuple): Figure size.
    - save_path (str, optional): If provided, the path where to save the figure.
    """

    # Calculate metrics
    pred_vs_hr = compare_signals(insulation_scores_pred, insulation_scores_hr)
    lr_vs_hr = compare_signals(insulation_scores_lr, insulation_scores_hr)
    
    metrics = [
        ["MAE", f"{lr_vs_hr['mae']:.4f}", f"{pred_vs_hr['mae']:.4f}*"],
        ["MSE", f"{lr_vs_hr['mse']:.4f}", f"{pred_vs_hr['mse']:.4f}*"],
        ["RMSE", f"{lr_vs_hr['rmse']:.4f}", f"{pred_vs_hr['rmse']:.4f}*"],
        ["Pearson's r", f"{lr_vs_hr['pearson_correlation']:.4f}", f"{pred_vs_hr['pearson_correlation']:.4f}*"],
        ["Spearman's ρ", f"{lr_vs_hr['spearman_correlation']:.4f}", f"{pred_vs_hr['spearman_correlation']:.4f}*"]
    ]
    column_labels = ["", "HR vs. LR", "HR vs. Pred"]

    plt.figure(figsize=figsize)
    ax1 = plt.subplot2grid((1, 5), (0, 0), colspan=4)  # Plot area for the line plot
    ax2 = plt.subplot2grid((1, 5), (0, 4))  # Plot area for the table

    genomic_end = insulation_scores_hr.shape[0] - 50
    indices = np.arange(0, genomic_end)
    x_values = (indices * resolution + start_pos)/1_000_000  # Converting indices to genomic coordinates [Mbp]

    ax1.plot(x_values, insulation_scores_pred[0:genomic_end], label='Insulation Score Prediction')
    ax1.plot(x_values, insulation_scores_lr[0:genomic_end], label='Insulation Score Low Resolution')
    ax1.plot(x_values, insulation_scores_hr[0:genomic_end], label='Insulation Score High Resolution')
    ax1.set_xlabel('Genomic Coordinates [Mbp]')
    ax1.set_ylabel('Insulation Score')
    
    start_index = 524
    end_index = 562
    start_genomic = (start_index * resolution + start_pos)/1_000_000
    end_genomic = (end_index * resolution + start_pos)/1_000_000
    ax1.axvspan(start_genomic, end_genomic, color='grey', alpha=0.5)
    ax1.text(end_genomic + 4.3, -0.01, 'Centromer Region', horizontalalignment='center', verticalalignment='center', fontsize=12, color='grey')

    ax1.legend()

    # Table for displaying metrics
    ax2.axis('off')
    table = ax2.table(cellText=metrics, colLabels=column_labels, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)  # Adjust scaling to fit the subplot

    plt.tight_layout()
    
    # Save the figure if a save path is provided
    if save_path:
        print(f"Saving plot to: {save_path}")
        plt.savefig(save_path, dpi=600)
        plt.close()  # Close the figure to free up memory
    else:
        plt.show()  # Or display the figure in a window



def plot_insulation_scores2(
    insulation_scores_pred1,
    insulation_scores_pred2,
    insulation_scores_lr1,
    insulation_scores_lr2,
    insulation_scores_hr1,
    insulation_scores_hr2,
    start_pos,
    resolution=10_000,
    figsize=(24, 8),
    save_path=None,
):
    """
    Plot insulation scores for different resolutions with genomic coordinates and include two 2x2 tables of metrics.

    Args:
    - insulation_scores_pred1 (np.ndarray): Insulation scores for the first prediction.
    - insulation_scores_pred2 (np.ndarray): Insulation scores for the second prediction.
    - insulation_scores_lr (np.ndarray): Insulation scores for low resolution.
    - insulation_scores_hr (np.ndarray): Insulation scores for high resolution.
    - start_pos (int): Starting genomic position.
    - resolution (int): The resolution of the genomic data.
    - figsize (tuple): Figure size.
    - save_path (str, optional): If provided, the path where to save the figure.
    """

    # Calculate metrics for the first prediction
    pred1_vs_hr1 = compare_signals(insulation_scores_pred1, insulation_scores_hr1)
    lr1_vs_hr1 = compare_signals(insulation_scores_lr1, insulation_scores_hr1)

    # Calculate metrics for the first prediction
    pred2_vs_hr2 = compare_signals(insulation_scores_pred2, insulation_scores_hr2)
    lr2_vs_hr2 = compare_signals(insulation_scores_lr2, insulation_scores_hr2)

    # Metrics tables2
    metrics1 = [
        ["MAE", f"{lr1_vs_hr1['mae']:.4f}", f"{pred1_vs_hr1['mae']:.4f}*"],
        ["MSE", f"{lr1_vs_hr1['mse']:.4f}", f"{pred1_vs_hr1['mse']:.4f}*"],
        ["RMSE", f"{lr1_vs_hr1['rmse']:.4f}", f"{pred1_vs_hr1['rmse']:.4f}*"],
        [
            "Pearson's r",
            f"{lr1_vs_hr1['pearson_correlation']:.4f}*",
            f"{pred1_vs_hr1['pearson_correlation']:.4f}",
        ],
        ["Spearman's ρ", f"{lr1_vs_hr1['spearman_correlation']:.4f}", f"{pred1_vs_hr1['spearman_correlation']:.4f}*"]
    ]
    metrics2 = [
        ["MAE", f"{lr2_vs_hr2['mae']:.4f}", f"{pred2_vs_hr2['mae']:.4f}*"],
        ["MSE", f"{lr2_vs_hr2['mse']:.4f}", f"{pred2_vs_hr2['mse']:.4f}*"],
        ["RMSE", f"{lr2_vs_hr2['rmse']:.4f}", f"{pred2_vs_hr2['rmse']:.4f}*"],
        [
            "Pearson's r",
            f"{lr2_vs_hr2['pearson_correlation']:.4f}*",
            f"{pred2_vs_hr2['pearson_correlation']:.4f}",
        ],
        ["Spearman's ρ", f"{lr2_vs_hr2['spearman_correlation']:.4f}", f"{pred2_vs_hr2['spearman_correlation']:.4f}*"]
    ]
    column_labels1 = ["@30Kbp", "HR vs. LR", "HR vs. Pred"]
    column_labels2 = ["@50Kbp", "HR vs. LR", "HR vs. Pred"]

    plt.figure(figsize=figsize)
    ax1 = plt.subplot2grid(
        (2, 5), (0, 0), colspan=4, rowspan=2
    )  # Plot area for the line plot
    ax2 = plt.subplot2grid((2, 5), (0, 4), rowspan=1)  # Plot area for the first table
    ax3 = plt.subplot2grid((2, 5), (1, 4), rowspan=1)  # Plot area for the second table

    genomic_end = insulation_scores_hr1.shape[0] - 50
    indices = np.arange(0, genomic_end)
    x_values = (
        indices * resolution + start_pos
    ) / 1_000_000  # Converting indices to genomic coordinates [Mbp]

    ax1.plot(
        x_values,
        insulation_scores_pred2[0:genomic_end],
        label="Insulation Score Prediction",
    )
    ax1.plot(
        x_values,
        insulation_scores_lr2[0:genomic_end],
        label="Insulation Score Low Resolution",
    )
    ax1.plot(
        x_values,
        insulation_scores_hr2[0:genomic_end],
        label="Insulation Score High Resolution",
    )
    ax1.set_ylabel("Insulation Score")

    # Add the following lines after setting labels
    current_ticks = ax1.get_xticks()  # Get the current tick locations
    new_labels = [f"{tick:.0f} MB" for tick in current_ticks]  # Create new labels with "MB" suffix
    ax1.set_xticklabels(new_labels)  # Set new labels


    start_index = 524*5
    end_index = 562*5
    start_genomic = (start_index * resolution + start_pos) / 1_000_000
    end_genomic = (end_index * resolution + start_pos) / 1_000_000
    ax1.axvspan(start_genomic, end_genomic, color="grey", alpha=0.5)
    ax1.text(
        end_genomic + 4.5,
        -0.01,
        "Centromer Region",
        horizontalalignment="center",
        verticalalignment="center",
        fontsize=12,
        color="grey",
    )

    ax1.legend()

    # First table for displaying metrics
    ax2.axis("off")
    table1 = ax2.table(
        cellText=metrics1, colLabels=column_labels1, loc="center", cellLoc="center"
    )
    table1.auto_set_font_size(False)
    table1.set_fontsize(10)
    table1.scale(1.2, 1.8)  # Adjust scaling to fit the subplot

    # Second table for displaying metrics
    ax3.axis("off")
    table2 = ax3.table(
        cellText=metrics2, colLabels=column_labels2, loc="center", cellLoc="center"
    )
    table2.auto_set_font_size(False)
    table2.set_fontsize(10)
    table2.scale(1.2, 1.8)  # Adjust scaling to fit the subplot

    plt.tight_layout()

    # Save the figure if a save path is provided
    if save_path:
        print(f"Saving plot to: {save_path}")
        plt.savefig(save_path, dpi=600)
        plt.close()  # Close the figure to free up memory
    else:
        plt.show()  # Or display the figure in a window