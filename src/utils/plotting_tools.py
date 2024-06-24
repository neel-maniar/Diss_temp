"""
Plotting utilities for the 2D field.
"""

import sys

import matplotlib.pyplot as plt

sys.path.append("src")
from utils.performance import rmse

from os.path import join, exists
import os

import matplotlib as mpl

mpl.rcParams["savefig.dpi"] = 300
mpl.rcParams["savefig.format"] = "pdf"
mpl.rcParams["savefig.bbox"] = "tight"
mpl.rcParams["figure.autolayout"] = True

if not exists("figures"):
    os.makedirs("figures")


def plot_data(x, y, xtest, ytest, filename=None):
    """
    Plot the observed values and true field vectors.

    Parameters
    ----------
    x (numpy.ndarray):
        Array of observed values for x coordinates.
    y (numpy.ndarray):
        Array of observed values for y coordinates.
    xtest (numpy.ndarray):
        Array of true field vectors for x coordinates.
    ytest (numpy.ndarray):
        Array of true field vectors for y coordinates.
    filename (str, optional):
        Name of the file to save the plot. Defaults to None.

    Returns:
        None
    """

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    ax.quiver(
        xtest[:, 0],
        xtest[:, 1],
        ytest[:, 0],
        ytest[:, 1],
        color="grey",
        label="True field",
        angles="xy",
    )

    ax.quiver(
        x[:, 0],
        x[:, 1],
        y[:, 0],
        y[:, 1],
        color="red",
        label="Observed Values",
        angles="xy",
    )
    ax.set(
        xlabel="$x_1$",
        ylabel="$x_2$",
    )
    ax.legend(
        framealpha=0.0,
        ncols=2,
        fontsize="medium",
        bbox_to_anchor=(0.5, -0.2),
        loc="lower center",
    )
    if filename is not None:
        filename = filename.replace(" ", "")
        plt.savefig(join("figures", filename))


def plot_pred(x, y, xtest, ytest, ypred, kernel_name="kernel", filename=None):
    """
    Plot the true field, predicted field, and residual field.

    Parameters
    ----------
    - x (numpy.ndarray): Input data for observed values.
    - y (numpy.ndarray): Observed values.
    - xtest (numpy.ndarray): Input data for test values.
    - ytest (numpy.ndarray): True field values for test data.
    - ypred (numpy.ndarray): Predicted field values for test data.
    - kernel_name (str, optional): Name of the kernel used for prediction. Defaults to "kernel".
    - filename (str, optional): Name of the file to save the plot. Defaults to None.
    """
    output_dim = y.shape[1]
    ypred = ypred.reshape(-1, output_dim, order="F")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].quiver(
        xtest[:, 0],
        xtest[:, 1],
        ytest[:, 0],
        ytest[:, 1],
        color="grey",
        label="True field",
        angles="xy",
    )

    axes[1].quiver(
        xtest[:, 0],
        xtest[:, 1],
        ypred[:, 0],
        ypred[:, 1],
        color="grey",
        label=f"Predicted field ({kernel_name})",
        angles="xy",
    )

    axes[2].quiver(
        xtest[:, 0],
        xtest[:, 1],
        ytest[:, 0] - ypred[:, 0],
        ytest[:, 1] - ypred[:, 1],
        color="grey",
        label="Residual field",
        angles="xy",
        scale=30,
    )

    for i in range(3):
        axes[i].quiver(
            x[:, 0],
            x[:, 1],
            y[:, 0],
            y[:, 1],
            color="red",
            label="Observed values",
            angles="xy",
        )

        axes[i].set(
            xlabel="$x_1$",
            ylabel="$x_2$",
        )
        axes[i].legend(
            framealpha=0.0,
            ncols=2,
            fontsize="medium",
            bbox_to_anchor=(0.5, -0.2),
            loc="lower center",
        )
    fig.suptitle(
        f"True, Predicted and Residuals using a prior with {kernel_name}. RMSE: {rmse(ytest, ypred):.4f}"
    )
    if filename is not None:
        filename = filename.replace(" ", "")
        filename = filename.replace("$", "")
        plt.savefig(join("figures", filename))
