"""
Main script from which the 2D and 3D scripts are called.

This script 
- Creates any essential non-existent directories
- Loads the default parameters
- Parses command line arguments
- Calls the relevant script (2D or 3D) with the specified parameters
"""

from jax import config

config.update("jax_enable_x64", True)
import sys
import os
from os.path import exists

sys.path.append("src")
from script_2D import script_2D
from script_3D import script_3D
from plotter import plotter

import argparse
import toml

# Create directories if they don't exist
if not exists("results"):
    os.makedirs("results")
if not exists("figures"):
    os.makedirs("figures")

# Import global defaults
params = toml.load("params.toml")["Global"]

# Argument parser for custom parameters
parser = argparse.ArgumentParser()
parser.add_argument("--3D", action="store_true", help="Run the 3D script")
parser.add_argument("--2D", action="store_true", help="Run the 2D script")
parser.add_argument(
    "--plot", action="store_true", help="Create visualisations of individual fits"
)
parser.add_argument("--adam", action="store_true", help="Use Adam optimiser")
parser.add_argument(
    "--regular", action="store_true", help="Use equally spaced train/test points"
)
parser.add_argument("--train", action="store_true", help="Train the artificial kernel")
parser.add_argument("--name", type=str, help="Name for the results directory")
parser.add_argument(
    "--noise", type=float, help="Std dev of Gaussian noise to add to training data"
)
parser.add_argument("--single-run", action="store_true", help="Run a single experiment")
parser.add_argument(
    "--nrRepeat", type=int, help="Number of repetitions of the experiment"
)

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
if args.noise is not None:
    params["noise"] = args.noise
if args.nrRepeat is not None:
    params["nrRepeat"] = args.nrRepeat
if args.name is not None:
    params["name"] = args.name

# Check that user wants to overwrite an existing result if existing name is used
# if params["name"] not in ["test3D", "test2D"]:
#     if os.path.exists(f"results/{params['name']}"):
#         overwrite = input(
#             f"Results with name {params['name']} already exist. Overwrite? (y/[n]) "
#         )
#         if overwrite.lower() != "y":
#             raise ValueError(
#                 "Please choose a different name or delete existing results."
#             )


# Sometimes we only want to run a single experiment for speedy plots, reproducibility
# testing or debugging.
if params["plot"] or params["regular"] or args.single_run:
    params["nrRepeat"] = 1

NAME = params["name"]

if not exists(f"results/{NAME}"):
    os.makedirs(f"results/{NAME}")
toml.dump(params, open(f"results/{NAME}/params.toml", "w"))

# Run the relevant script
if args.__getattribute__("2D"):
    script_2D(params)
elif args.__getattribute__("3D"):
    script_3D(params)

# Create plots for the aggregate results
if params["nrRepeat"] > 1:
    plotter(params)
