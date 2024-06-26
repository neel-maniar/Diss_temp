# About
This is Neel Maniar's Gitlab repository for the dissertation.

# Installation
The repository is best accessed by cloning via git. This can be done via `ssh` with the following command:
```bash
git clone git@gitlab.developers.cam.ac.uk:phy/data-intensive-science-mphil/A2_MED_Assessment/nm741.git
```
The project can either be accessed via `conda` directly or via `docker`.
### Conda
In order to set up the environment, you can use `conda`. Ensure that you have either `anaconda` or `miniconda` installed and accessible from the command line, and then run the following command:
```bash
conda create --file environment.yml
```
The environment should take around 2-4 minutes to build.

### Docker
In order to use `docker`, ensure that you have cloned the repository and that you have Docker Desktop installed. Create the image with
```bash
docker build -t IMAGE_TAG .
```

Then create an instance of a container by running
```bash
docker run -d -t --name=CONTAINER_NAME IMAGE_TAG
```

Then create a terminal by running
```bash
docker exec -ti CONTAINER_NAME bash
```

There are many programs that produce figures. These cannot be viewed within the docker container, as it is a basic Linux terminal. However, they can be copied into your local system via:

```bash
docker cp CONTAINER_NAME:/dissertation/figures /HOST/PATH
```

After you are done, the terminal may be closed by pressing `Ctrl`+`D`. The container may be closed by running
```bash
docker stop CONTAINER_NAME
```
# Running

To run the code, activate the conda environment:

```console
$ conda activate dissertation
```

Then, run the code like so (replacing `<N>` with `2` or `3`):

```console
$ python src/main.py --<N>D <options>
```

## Arguments

### -h, --help

Show a help message and exit

### Compulsory

#### --2D

Run the 2D script. 

#### --3D

Run the 3D script. 

### Optional

#### --plot

Create visualisations of individual fits. (Only one run completed)

#### --adam

Use Adam optimiser

#### --regular

Use equally spaced train/test points (Only one run completed. For reproducibility testing)

#### --train

Train the artificial kernel

#### --name

Name for the results directory

#### --noise

Std dev of Gaussian noise to add to training data

#### --single-run

Run a single experiment (useful for testing and debugging)

#### --nrRepeat

Number of repetitions of the experiment


# Plots
[FILLIN]

# Contents
## Scripts

## Auxiliary files

## Documentation
Documentation can be produced locally by running `make html` whilst inside the `docs` folder. This will generate a `html` folder containing documentation. The `index.html` file may be viewed in any browser.

# Use of Generative AI

# Acknowledgements
I would like to thank my supervisor, Dr Henry Moss, for their guidance and support throughout this project, and the course director, Dr James Fergusson.

# Contact
If there are any comments, questions or corrections, please email me at:
nm741@cam.ac.uk
