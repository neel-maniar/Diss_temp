"""
Plotting Tools
==============
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("src")
from utils.performance import rmse

from os.path import join, exists
import os
import shutil

from matplotlib import rcParams
from mpl_toolkits.mplot3d.art3d import Line3DCollection

if shutil.which("latex"):
    style_path = join(os.path.dirname(__file__), "..", "gpjax.mplstyle")
else:
    style_path = join(os.path.dirname(__file__), "..", "nonlatex.mplstyle")
plt.style.use(style_path)
colors = rcParams["axes.prop_cycle"].by_key()["color"]

if not exists("figures"):
    os.makedirs("figures")


def plot_data(x, y, xtest, ytest, directory):
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
    if directory is not None:
        if not exists(join("figures", directory)):
            os.makedirs(join("figures", directory))
        path = join("figures", directory, "2DSimulationData.pdf")
        plt.savefig(path)
        print(f"Visualisation of 2D simulation data saved to `{path}`")


def plot_pred(x, y, xtest, ytest, ypred, kernel_name, directory, regular):
    """
    Plot the true field, predicted field, and residual field.

    Parameters
    ----------
    x: numpy.ndarray
        Input data for observed values.
    y: numpy.ndarray
        Observed values.
    xtest: numpy.ndarray
        Input data for test values.
    ytest: numpy.ndarray
        True field values for test data.
    ypred: numpy.ndarray
        Predicted field values for test data.
    kernel_name: str, optional
        Name of the kernel used for prediction. Defaults to "kernel".
    filename: str, optional
        Name of the file to save the plot. Defaults to None.
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

    if directory is not None:
        if not exists(join("figures", directory)):
            os.makedirs(join("figures", directory))
        filename = f"{'regular_' if regular else ''}{kernel_name}"
        filename = filename.replace(" ", "")
        filename = filename.replace("$", "")
        path = join("figures", directory, filename)
        plt.savefig(path)
        print(f"Visualisation of diagonal kernel predictions saved to `{path}.pdf`")


# Copilot code
def plot_3D_data(pos, magnitude):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection="3d")
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]

    # Create line segments
    points = np.array([x, y, z]).T.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Create a Line3DCollection
    norm = plt.Normalize(magnitude.min(), magnitude.max())
    lc = Line3DCollection(segments, cmap="viridis", norm=norm)
    lc.set_array(magnitude)
    lc.set_linewidth(2.0)  # Set line width to 2.0

    # Add collection to the plot
    ax.add_collection(lc)

    # Set plot parameters
    ax.view_init(elev=20, azim=225)
    ax.set_xlabel("$x_1$[m]")
    ax.set_ylabel("$x_2$[m]")
    ax.set_zlabel("$x_3$[m]")
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.grid(False)

    # Set the limits of the axes
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())
    ax.set_zlim(z.min(), z.max())

    # Add a color bar
    cbar = plt.colorbar(
        lc, ax=ax, fraction=0.03, pad=0.04
    )  # Make the color bar smaller
    cbar.set_label("Magnitude of the magnetic field [T]")

    # Show and save the plot
    plt.savefig("figures/observed_path_3D_coloured.pdf", dpi=300, format="pdf")


# Ground truth + diagonal estimate + residuals plot (Courtesy Mate Balogh)
def plot_3D_pred(dataset_train, dataset_test, ypred, filename, kernel_name):

    fig = plt.figure(figsize=(11, 4))

    # Experiment with this
    fig.canvas.toolbar_visible = False
    fig.canvas.header_visible = False
    fig.canvas.footer_visible = False

    N_TEST = dataset_test.X.shape[0] // 3
    N_TRAIN = dataset_train.X.shape[0] // 3
    N_TRAIN_PLOT = 50
    N_TEST_PLOT = 100
    a = N_TRAIN // N_TRAIN_PLOT
    b = N_TEST // N_TEST_PLOT

    ax0 = fig.add_subplot(131, projection="3d")
    ax0.set_title("True Field")

    ax0.quiver(
        dataset_test.X[:N_TEST:b, 0],
        dataset_test.X[:N_TEST:b, 1],
        dataset_test.X[:N_TEST:b, 2],
        dataset_test.y[:N_TEST:b].squeeze(),
        dataset_test.y[N_TEST : 2 * N_TEST : b].squeeze(),
        dataset_test.y[2 * N_TEST :: b].squeeze(),
        length=0.3,
        normalize=False,
        linewidth=1,
        color="grey",
        label="True field",
    )

    ax1 = fig.add_subplot(132, projection="3d")
    ax1.set_title("Predicted Field")

    ax1.quiver(
        dataset_test.X[:N_TEST:b, 0],
        dataset_test.X[:N_TEST:b, 1],
        dataset_test.X[:N_TEST:b, 2],
        ypred[:N_TEST:b].squeeze(),
        ypred[N_TEST : 2 * N_TEST : b].squeeze(),
        ypred[2 * N_TEST :: b].squeeze(),
        color="grey",
        label=f"Predicted field",
        length=0.3,
        normalize=False,
        linewidth=1,
    )

    ax2 = fig.add_subplot(133, projection="3d")

    ax2.set_title("Residual Field")

    residuals = dataset_test.y.squeeze() - ypred.squeeze()

    ax2.quiver(
        dataset_test.X[:N_TEST:b, 0],
        dataset_test.X[:N_TEST:b, 1],
        dataset_test.X[:N_TEST:b, 2],
        residuals[:N_TEST:b].squeeze(),
        residuals[N_TEST : 2 * N_TEST : b].squeeze(),
        residuals[2 * N_TEST :: b].squeeze(),
        length=0.3,
        normalize=False,
        linewidth=1,
        color="grey",
        label=f"Residual field",
    )

    for ax_ in [ax0, ax1, ax2]:
        ax_.quiver(
            dataset_train.X[:N_TRAIN:a, 0],
            dataset_train.X[:N_TRAIN:a, 1],
            dataset_train.X[:N_TRAIN:a, 2],
            dataset_train.y[:N_TRAIN:a].squeeze(),
            dataset_train.y[N_TRAIN : 2 * N_TRAIN : a].squeeze(),
            dataset_train.y[2 * N_TRAIN :: a].squeeze(),
            length=0.3,
            normalize=False,
            linewidth=1,
            color="red",
            label=f"Observed field",
        )

        ax_.set_xlabel("$x$")
        ax_.set_ylabel("$y$")
        ax_.set_zlabel("$z$")
        ax_.legend(fontsize="medium", bbox_to_anchor=(0.5, -0.2), loc="lower center")

    fig.suptitle(
        f"True, Predicted and Residuals using a prior with {kernel_name}. RMSE: {rmse(dataset_test.y, ypred):.4f}"
    )

    if filename is not None:
        filename = filename.replace(" ", "")
        filename = filename.replace("$", "")
        plt.savefig(join("figures", filename))
