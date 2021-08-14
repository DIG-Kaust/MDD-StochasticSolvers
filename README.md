# MDD-StochasticSolvers

Repository containing material for the `Multi-dimensional deconvolution with stochastic solvers` project.

Two synthetic examples are considered:

* Hyperbolic: Set of synthetically generated hyperbolic events for both model and kernel operator
* WavefieldSeparation-dipping: Layered model with dipping seabed and receivers in obc acquisition. Data are created via FD modelling up/down separation


## Project structure
This repository is organized as follows:

* **Data**: folder where input data must be placed (cannot be uploaded directly to Github and will be shared separately)

* **stochmdd.py**: set of routines to perform MDD with stochastic gradients

* **Hyperbolic_MDD_basic.ipynb**: notebook performing a basic example of MDC and MDD for both single and multiple virtual sources to familiarize with the forward modelling and inverse problem

* **Hyperbolic_MDD_basic_trackednorms.ipynb**: notebook performing the basic example of MDC and MDD for a single virtual source with additional code for tracking norm of residual and error

* **Hyperbolic_MDD_stochastic.ipynb**: notebook performing MDD on single virtual source with full batch stochastic solvers

* **Hyperbolic_MDD_stochasticminibatch.ipynb**: notebook performing MDD on single virtual source with mini batch stochastic solvers

* **WavefieldSeparation-dipping_MDD_basic.ipynb**: notebook performing wavefield separation for synthetic obc data with dipping seabed followed by standard MDD

* **WavefieldSeparation-dipping_MDD_stochastic_singlesource.ipynb**: notebook performing MDD on single virtual source with stochastic solvers

* **WavefieldSeparation-dipping_MDD_stochastic_multisource.ipynb**: notebook performing MDD on multiple virtual sources with stochastic solvers



## Getting started
To ensure reproducibility of the results, we suggest using the `environment.yml` file when creating an environment.

When working within KAUST computational environment, first install Miniconda using [https://github.com/kaust-rccl/ibex-miniconda-install](https://github.com/kaust-rccl/ibex-miniconda-install).

Once you have made sure that Anaconda (or Miniconda) is available in your system, to create a new environment simply run:

```
conda env create -f environment.yml
```

to create an environment called `mdd-stochastic`.
