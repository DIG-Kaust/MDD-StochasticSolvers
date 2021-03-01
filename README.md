# MDD-StochasticSolvers

Repository containing material for the `Multi-dimensional deconvolution with stochastic solvers` project.


## Project structure
This repository is organized as follows:

* **Data**: folder where input data must be placed (cannot be uploaded directly to Github and will be shared separately)
* **calibrate.py**: routine for calibration of seismic data prior to wavefield separation
* **AngleGather.py**: routine for computation of angle gathers


* **WavefieldSeparation-dipping_MDD_dipping.ipynb**: notebook performing wavefield separation for synthetic obc data with dipping
seabed followed by standard MDD
* **Marchenko_angle_gathers.ipynb**: notebook performing marchenko redatuming + MDD + angle gathers for a number of depth levels


## Getting started
To ensure reproducibility of the results, we suggest using the `environment.yml` file when creating an environment.
Provided that Anaconda (or Miniconda) is avaiable in your system, to create a new environment simply run:

```
conda env create -f environment.yml
```

to create an environment called `mdd-stochastic`.
