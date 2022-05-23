# Notebooks

The notebooks are divided into five main directories:

* **:open_file_folder: Hyperbolic**:

  - :orange_book: ``Hyperbolic_MDD_basic.ipynb``: notebook performing a basic example of MDC and MDD for both single and multiple virtual sources
  - :orange_book: ``Hyperbolic_MDD_stochasticminibatch.ipynb``: notebook performing MDD on single virtual source with mini batch stochastic solvers
  - :orange_book: ``Hyperbolic_MDD_stochasticminibatch_numpy.ipynb``: same as above but purely implemented in numpy

* **:open_file_folder: Dipping_OBC**: 
  - ``WavefieldSeparation-dipping_MDD_basic.ipynb``**``: notebook performing wavefield separation for synthetic obc data with dipping seabed followed by standard MDD
  - ``WavefieldSeparation-dipping_MDD_steepest_singlesource.ipynb``: notebook performing MDD on single virtual source with steepest descent solver
  - ``WavefieldSeparation-dipping_MDD_stochastic_singlesource.ipynb``: notebook performing MDD on single virtual source with stochastic solvers
  - ``WavefieldSeparation-dipping_MDD_stochastic_singlesource_numpy.ipynb``: same as above but purely implemented in numpy
  - ``WavefieldSeparation-dipping_MDD_stochastic_singlesource_torch_vs_numpy.ipynb``: same as above but benchmark of torch vs numpy implementation
  - ``WavefieldSeparation-dipping_MDD_stochastic_multisource.ipynb``: notebook performing MDD on multiple virtual sources with stochastic solvers
  - ``WavefieldSeparation-dipping_MDD_stochastic_multisource_numpy.ipynb``: same as above but purely implemented in numpy
  - ``WavefieldSeparation-dipping_MDD_stochastic_multisource_torch_vs_numpy.ipynb``: same as above but benchmark of torch vs numpy implementation

* **:open_file_folder: Salt**: 
  - ``MarchenkoSalt_MDD_stochastic_singlesource.ipynb``: notebook performing MDD on single virtual source with stochastic solvers
  - ``MarchenkoSalt_MDD_stochastic_multisource.ipynb``: notebook performing MDD on multiple virtual source with stochastic solvers
  - ``MarchenkoSalt_MDD_stochastic_multisource_reciprocity.ipynb``: same as above but with additional reciprocity preconditioner
  
* **:open_file_folder: Synthetic_Volve**: 
  - :orange_book: ``Volve_synthetic_MDD_stochastic.ipynb``: notebook performing MDD on single and multiple virtual sources with stochastic solvers
  
* **:open_file_folder: Field_Volve**: 
  - :orange_book: ``Volve_MDD_stochastic.ipynb``: notebook performing MDD on single and multiple virtual sources with stochastic solvers
  - :orange_book: ``Volve_MDD_stochastic_masked.ipynb``: same as above but with additional time window preconditioner
  
* **:open_file_folder: Figures**: 
  - :orange_book: ``Paper_figures.ipynb``: notebook recreating all figures in XX paper.

* **:open_file_folder: Others**: 
  - :orange_book: ``SteepestDescent.ipynb``: notebook applying steepest descent algorithm on simple problem
  - :orange_book: ``SteepestDescent_stochastic.ipynb``: notebook comparing steepest descent and various stochastic gradient algorithms on linear regression problem
  - :orange_book: ``LogisticReg_stochastic.ipynb``: notebook comparing various stochastic gradient algorithms on logistic regression problem (formulation1)
  - :orange_book: ``LogisticReg1_stochastic.ipynb``: notebook comparing various stochastic gradient algorithms on logistic regression problem (formulation2)
