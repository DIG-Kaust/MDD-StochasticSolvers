# MDD-StochasticSolvers

Repository containing material for the `Multi-dimensional deconvolution with stochastic solvers` project.


## Project structure
This repository is organized as follows:

* **Data**: folder where input data must be placed (cannot be uploaded directly to Github and will be shared separately)
* **calibrate.py**: routine for calibration of seismic data prior to wavefield separation
* **AngleGather.py**: routine for computation of angle gathers

* **MDD_basic.ipynb**: notebook performing a basic example of MDC and MDD to familiarize with the forward modelling and inverse problem

* **MDD_basic_trackednorms.ipynb**: notebook performing the basic example of MDC and MDD with additional code for tracking norm of residual and error

* **MDD_stochasticgradient.ipynb**: notebook performing a stochastic gradient descent example of MDC and MDD
* **MDD_minibatchGradient.ipynb**: notebook performing minibatch gradient descent example of MDC and MDD
* **MDD_Permutated_MiniBatch_gradient_descent.ipynb**: notebook performing minibatch gradient descent example of MDC and MDD with permutation of sources in Gwav

* **Advanced_stochastic_solvers.ipynb**: notebook performing MDC with advanced gradient based solvers
* **Minibatch_Advanced_solver**: notebook performing advanced gradient based solvers with minibatches
* **Advanced_Solvers_Func.ipynb**: notebook performs and compares MDC with advanced gradient solvers and compares different learning rate schedulers

* **WavefieldSeparation-dipping_MDD_dipping.ipynb**: notebook performing wavefield separation for synthetic obc data with dipping
seabed followed by standard MDD

* **Marchenko_angle_gathers.ipynb**: notebook performing marchenko redatuming + MDD + angle gathers for a number of depth levels


## Getting started
To ensure reproducibility of the results, we suggest using the `environment.yml` file when creating an environment.

When working within KAUST computational environment, first install Miniconda using [https://github.com/kaust-rccl/ibex-miniconda-install](https://github.com/kaust-rccl/ibex-miniconda-install).

Once you have made sure that Anaconda (or Miniconda) is available in your system, to create a new environment simply run:

```
conda env create -f environment.yml
```

to create an environment called `mdd-stochastic`.
