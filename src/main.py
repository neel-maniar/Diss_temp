from jax import config

config.update("jax_enable_x64", True)
import sys
import numpy as np
import jax.random as jr
import matplotlib.pyplot as plt
from jaxtyping import install_import_hook
from matplotlib import rcParams
from tqdm.auto import tqdm

sys.path.append("src")
from script_2D import script_2D
from script_3D import script_3D

plt.style.use(
    "https://raw.githubusercontent.com/JaxGaussianProcesses/GPJax/main/docs/examples/gpjax.mplstyle"
)
colors = rcParams["axes.prop_cycle"].by_key()["color"]

import argparse
import toml

# Import global defaults
params = toml.load("params.toml")["Global"]

# Argument parser for custom parameters
parser = argparse.ArgumentParser()
parser.add_argument("--plot", action="store_true")
parser.add_argument("--adam", action="store_true")
parser.add_argument("--regular", action="store_true")
parser.add_argument("--train", action="store_true")
parser.add_argument("--3D", action="store_true")
parser.add_argument("--2D", action="store_true")

args = parser.parse_args()

# Load the relevant default parameters
if args.__getattribute__("2D"):
    params.update(toml.load("params.toml")["2D"])
elif args.__getattribute__("3D"):
    params.update(toml.load("params.toml")["3D"])
else:
    raise ValueError("Please specify the 2D or 3D case with --2D or --3D")

params["plot"] = args.plot

# Overwrite defaults with custom parameters
if args.adam:
    params["optimiser"] = "adam"
if args.regular:
    params["regular"] = True
if args.train:
    params["train_artificial"] = True

if params["plot"] or params["regular"]:
    params["nrRepeat"] = 1

# Run the relevant script
if args.__getattribute__("2D"):
    script_2D(params)
elif args.__getattribute__("3D"):
    script_3D(params)
