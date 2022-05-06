![MDD-StochasticSolvers](https://github.com/DIG-Kaust/MDD-StochasticSolvers/blob/main/logo.png)

Reproducible material for **Stochastic Multi-dimensional Deconvolution -
Ravasi M., Selvan, T., Luiken N. ** - [ArXiv Paper](https://arxiv.org/abs/2202.04486).


## Project structure
This repository is organized as follows:

- **stochmdd**: python library containing routines for stochastic mdd
- **notebooks**: set of jupyter notebooks reproducing the experiments in the paper (see the README file inside this folder for more details)
* **data**: folder where input data must be placed

**NOTE**: due to their large size, the various datasets cannot be shared directly in this repository. If interested, contact the authors directly!


## Data and Experiements

Four experiments are considered:

* **Hyperbolic**: set of synthetically generated hyperbolic events for both model and kernel operator.
* **Dipping_OBC**: layered model with dipping seabed and receivers at the seabed in OBC style acquisition. 
  Data are created via FD modelling followed by up/down separation.
* **Salt**: salt model from [Vargas et al. (2021)](https://library.seg.org/doi/full/10.1190/geo2020-0939.1). 
  Data are created via Scattering-Rayleigh-Marchenko redatuming.
* **Synthetic_Volve**: synthetic model that resembles the Volve field subsurface model. See [VolveSynthetic](https://github.com/DIG-Kaust/VolveSynthetic)
  for more details regarding the generation of the model and seismic data. Data are created via FD modelling followed by up/down separation.
* **Field_Volve**: field Volve OBC dataset, data are pre-processed by up/down separation.


## Getting started
To ensure reproducibility of the results, we suggest using the `environment.yml` file when creating an environment.

When working within KAUST computational environment, first install Miniconda using [https://github.com/kaust-rccl/ibex-miniconda-install](https://github.com/kaust-rccl/ibex-miniconda-install).

Once you have made sure that Anaconda (or Miniconda) is available in your system, to create a new environment simply run:

```
conda env create -f environment.yml
```

to create an environment called `mdd-stochastic`.
