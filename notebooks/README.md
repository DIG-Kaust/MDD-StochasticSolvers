# Notebooks

The notebooks are divided into five main directories:

* **Hyperbolic**:

  - ``Hyperbolic_MDD_basic.ipynb``: notebook performing a basic example of MDC and MDD for both single and multiple virtual sources
  - ``Hyperbolic_MDD_stochasticminibatch.ipynb``: notebook performing MDD on single virtual source with mini batch stochastic solvers
  - ``Hyperbolic_MDD_stochasticminibatch_numpy.ipynb``: same as above but purely implemented in numpy

* **Dipping_OBC**: 
  - ``WavefieldSeparation-dipping_MDD_basic.ipynb``**``: notebook performing wavefield separation for synthetic obc data with dipping seabed followed by standard MDD
  - ``WavefieldSeparation-dipping_MDD_steepest_singlesource.ipynb``: notebook performing MDD on single virtual source with steepest descent solver
  - ``WavefieldSeparation-dipping_MDD_stochastic_singlesource.ipynb``: notebook performing MDD on single virtual source with stochastic solvers
  - ``WavefieldSeparation-dipping_MDD_stochastic_singlesource_numpy.ipynb``: same as above but purely implemented in numpy
  - ``WavefieldSeparation-dipping_MDD_stochastic_singlesource_torch_vs_numpy.ipynb``: same as above but benchmark of torch vs numpy implementation
  - ``WavefieldSeparation-dipping_MDD_stochastic_multisource.ipynb``: notebook performing MDD on multiple virtual sources with stochastic solvers
  - ``WavefieldSeparation-dipping_MDD_stochastic_multisource_numpy.ipynb``: same as above but purely implemented in numpy
  - ``WavefieldSeparation-dipping_MDD_stochastic_multisource_torch_vs_numpy.ipynb``: same as above but benchmark of torch vs numpy implementation

* **Salt**: 
  - ``MarchenkoSalt_MDD_stochastic_singlesource.ipynb``: notebook performing MDD on single virtual source with stochastic solvers
  - ``MarchenkoSalt_MDD_stochastic_multisource.ipynb``: notebook performing MDD on multiple virtual source with stochastic solvers
  - ``MarchenkoSalt_MDD_stochastic_multisource_reciprocity.ipynb``: same as above but with additional reciprocity preconditioner

* **Synthetic_Volve**: 
    - ``Volve_synthetic_MDD_stochastic.ipynb``: notebook performing MDD on single and multiple virtual sources with stochastic solvers
  
* **Field_Volve**: 
    - ``Volve_MDD_stochastic.ipynb``: notebook performing MDD on single and multiple virtual sources with stochastic solvers
    - ``Volve_MDD_stochastic_masked.ipynb``: same as above but with additional time window preconditioner
  
* **Others**: 
  - ``SteepestDescent.ipynb``: notebook applying steepest descent algorithm on simple problem
  - ``SteepestDescent_stochastic.ipynb``: notebook comparing steepest descent and stochastic gradient on simple problem
  - ``Paper_figures.ipynb``: notebook recreating all figures in XX paper.
