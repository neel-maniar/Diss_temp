from os.path import join

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import jax.numpy as jnp
import os
from os.path import exists

import shutil

if shutil.which("latex"):
    style_path = join(os.path.dirname(__file__), "gpjax.mplstyle")
else:
    style_path = join(os.path.dirname(__file__), "nonlatex.mplstyle")
plt.style.use(style_path)
colors = rcParams["axes.prop_cycle"].by_key()["color"]


def plot_results(params, perf_metric):
    """
    Plot the aggregate results of RMSE or NLPD.

    Parameters
    ----------
        params : dict
            A dictionary containing the parameters for the plot.

            - name : str
                Name of the plot.
            - N_c_list : list
                List of number of artificial points.
            - nrRepeat : int
                Number of trials.
        perf_metric : str
            Performance metric to plot. Can be 'RMSE' or 'NLPD'.

    Raises
    ------
        NotImplementedError: If the performance metric is not 'RMSE' or 'NLPD'.

    Returns
    -------
        None
    """
    NAME = params["name"]
    N_C_LIST = params["N_c_list"]
    NRREPEAT = params["nrRepeat"]
    directory = join("./results", NAME)

    perf_metric = perf_metric.upper()

    if perf_metric == "RMSE":
        errCust_all = np.load(join(directory, "RMSE (Custom Kernel).npy"))
        errDiagObs_all = np.load(join(directory, "RMSE (Artificial Kernel).npy"))
        errDiag_all = np.load(join(directory, "RMSE (Diagonal Kernel).npy"))

    elif perf_metric == "NLPD":
        errCust_all = np.load(join(directory, "NLPD (Custom Kernel).npy"))
        errDiagObs_all = np.load(join(directory, "NLPD (Artificial Kernel).npy"))
        errDiag_all = np.load(join(directory, "NLPD (Diagonal Kernel).npy"))

    else:
        raise NotImplementedError(
            f"Performance metric {perf_metric} not implemented. Try 'rmse' or 'nlpd'."
        )

    # Plot the results
    fig, ax = plt.subplots(figsize=(10, 5))

    denom = np.sqrt(50)

    errDiag = errDiag_all.mean()
    errDiagStd = errDiag_all.std() / denom

    errCust = errCust_all.mean()
    errCustStd = errCust_all.std() / denom

    errDiagObs = errDiagObs_all.mean(axis=0)
    errDiagObsStd = errDiagObs_all.std(axis=0) / denom

    errDiag_plot = np.array([errDiag] * len(N_C_LIST))
    errCust_plot = np.array([errCust] * len(N_C_LIST))

    ax.plot(
        N_C_LIST,
        errDiag_plot,
        label="Diagonal Kernel ($\mu \pm 1\sigma$)",
        color=colors[0],
    )
    ax.fill_between(
        N_C_LIST,
        errDiag_plot - errDiagStd,
        errDiag_plot + errDiagStd,
        alpha=0.2,
        color=colors[0],
    )

    ax.plot(
        N_C_LIST,
        errCust_plot,
        label="Custom Kernel ($\mu \pm 1\sigma$)",
        color=colors[1],
    )
    ax.fill_between(
        N_C_LIST,
        errCust_plot - errCustStd,
        errCust_plot + errCustStd,
        alpha=0.2,
        color=colors[1],
    )

    ax.plot(
        N_C_LIST,
        errDiagObs,
        label="Artificial Kernel ($\mu \pm 1\sigma$)",
        color=colors[2],
    )
    ax.fill_between(
        N_C_LIST,
        errDiagObs - errDiagObsStd,
        errDiagObs + errDiagObsStd,
        alpha=0.2,
        color=colors[2],
    )

    # if perf_metric == "RMSE":
    #     plt.axhline(
    #         1.5813374006063812, color="black", linestyle="--", label="Zero prediction"
    #     )
    # else:
    #     plt.axhline(
    #         1735.402016386357, color="black", linestyle="--", label="Zero prediction"
    #     )

    plt.legend()
    ax.set_title(f"{perf_metric} over {NRREPEAT} trials")
    ax.set_xscale("log")
    ax.set_xticks(N_C_LIST)  # Set the xticks to match the N_C_LIST values
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Number of Artificial Points")
    ax.set_ylabel(perf_metric)

    # Create the directory if it doesn't exist
    if not exists(f"figures/{NAME}"):
        os.makedirs(f"figures/{NAME}")

    plt.savefig(join(f"figures/{NAME}/{perf_metric}.pdf"))


def plotter(params):
    """
    Plot the aggregate results of RMSE and NLPD.

    Parameters
    ----------
        params : dict
            A dictionary containing the parameters for the plot.

    Returns
    -------
        None
    """
    plot_results(params, "rmse")
    plot_results(params, "nlpd")


if __name__ == "__main__":
    import toml
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, help="Name of the plot")
    parser.add_argument("--3D", action="store_true", help="Plot 3D results")
    parser.add_argument("--2D", action="store_true", help="Plot 2D results")
    args = parser.parse_args()

    params_global = toml.load("params.toml")["Global"]
    if args.name:
        params_global["name"] = args.name

    if args.__getattribute__("2D"):
        params = toml.load("params.toml")["2D"]
    elif args.__getattribute__("3D"):
        params = toml.load("params.toml")["3D"]
    else:
        raise ValueError("Please specify the 2D or 3D case with --2D or --3D")
    params.update(params_global)
    plotter(params)
