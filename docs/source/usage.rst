Usage
=====
.. _installation:

Installation
------------

To run code in the dissertation, first install the dependencies using conda:

.. code-block:: console

   $ conda env create -f environment.yml

.. _running:

Running
-------

To run the code, activate the conda environment:

.. code-block:: console

   $ conda activate dissertation

Then, run the code like so (replacing `<N>` with `2` or `3`):

.. code-block:: console

   $ python src/main.py --<N>D <options>

.. _arguments:

Arguments
---------
-h, --help
    Show a help message and exit

Compulsory
~~~~~~~~~~
--2D
  Run the 2D script. 

--3D
  Run the 3D script. 

Optional
~~~~~~~~
--plot
    Create visualisations of individual fits. (Only one run completed)

--adam
    Use Adam optimiser

--regular
    Use equally spaced train/test points (Only one run completed. For reproducibility testing)

--train
    Train the artificial kernel

--name
    Name for the results directory

--noise
    Std dev of Gaussian noise to add to training data

--single-run
    Run a single experiment (useful for testing and debugging)

--nrRepeat
    Number of repetitions of the experiment

