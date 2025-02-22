# AQA: Analytical Quantum Alchemy
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14885476.svg)](https://doi.org/10.5281/zenodo.14885476)

This code is a modification of [giorgiodomen/Supplementary_code_for_Quantum_Alchemy](https://github.com/giorgiodomen/Supplementary_code_for_Quantum_Alchemy)
developed by Dr. Giorgio Domenichini. Please refer to articles related to the original code:

Article | Authors | Title
--------|---------|--------
[J. Phys. Chem. 2022](https://aip.scitation.org/doi/10.1063/5.0085817) | Giorgio Domenichini, O. Anatole von Lilienfeld | Alchemical geometry relaxation
[arXiv 2023](https://doi.org/10.48550/arXiv.2306.16409) | O. Anatole von Lilienfeld, Giorgio Domenichini | Even order contributions to relative energies vanish for antisymmetric perturbations
[J. Phys. Chem. 2024](https://doi.org/10.1063/5.0196383) | Giorgio Domenichini | Extending the definition of atomic basis sets to atoms with fractional nuclear charge

For the license, please refer to the LICENSE section.
AQA is also based on PySCF ([pyscf/pyscf](https://github.com/pyscf/pyscf)).
The license and notice files of PySCF is included in `./aqa/`.

AQA is still under development and contain bugs.


## Modification
Analytical alchemical energy derivatives up to the third order in Kohn-Sham density
functional theory (KSDFT) as well as quantum alchemy-based rules and analysis methods
are added.
LDA, GGA, and global hybrid functionals are supported.
AQA was used in the following work:

Article | Authors | Title
--------|---------|--------
[arXiv 2025](https://arxiv.org/abs/2502.12761) | Takafumi Shiraogawa, Simon León Krug, Masahiro Ehara, O. Anatole von Lilienfeld | Antisymmetry rules of response properties in certain chemical spaces

The following modifications are made for PySCF:
- SCF convergence criteria are modified for APDFT calculations.
- The analytical derivatives are implemented based on the implementation of PySCF.


## Installation
```
git clone https://github.com/takafumi-shiraogawa/AQA
```
Please add a path of `aqa/` to PYTHONPATH since the current version of AQA
is not packaged.
Generation of chemical space with unique molecules depends on QML ([qmlcode/qml](https://github.com/qmlcode/qml)).
Please compile `frepresentations.f90` in `./aqa/mini_qml/` with F2PY.
The example of the command can be found in `./aqa/mini_qml/compile.sh`.
The license file of QML is included in `./aqa/mini_qml/`.

### Requirements
To install the required packages, please use `aqa_env.yml` or run the following commands:
```
conda install -c pyscf pyscf  
pip install basis_set_exchange  
conda install scipy  
conda install numpy  
conda install matplotlib  
conda install pandas  
conda install xarray  
pip install git+https://github.com/pyscf/properties  
```
Only PySCF ver. 2.2 is tested. The older version of PySCF cannot be used.
Please add a path of `properties/` to PYSCF_EXT_PATH.


## License
This software is released under the MIT License and includes the license of the original code
[giorgiodomen/Supplementary_code_for_Quantum_Alchemy](https://github.com/giorgiodomen/Supplementary_code_for_Quantum_Alchemy)
in `LICENSE`.


## Tests and examples
Jupyter Notebooks and Python files in `./test/aqa` include, for example:
- Numerical validation of alchemical derivatives using CO and CH3OH with symmetrized or rotated- and translated- coordinates  
  (co/co.ipynb, co_rot-trans.ipynb, co_pcx2.ipynb, ch3oh/ch3oh.ipynb, ch3oh_rot-trans.ipynb,  
  co/hartree-fock/co.ipynb, co_rot-trans.ipynb, ch3oh/hartree-fock/ch3oh.ipynb, ch3oh_rot-trans.ipynb)
- Alchemical perturbation density functional theory (APDFT) calculation of CO with reference N2  
  (n2/n2-co_apdft.ipynb, n2-co_apdft_hf.ipynb)
- Alchemical relative energy estimate for N2-CO-BF  
  (co/co_relative-apdft.ipynb)
- Alchemical even energy estimate for N2 with reference CO  
  (co/co_even_estimate.ipynb)
- Alchemical calculator class  
  (co/co_alch_calc_class.ipynb, co/hartree-fock/co_alch_calc_class.ipynb, n2/n2-co_apdft_hf.ipynb)
- Levy's estimate of relative energy E(N2) - E(BF)  
  (co/co_Levy.ipynb)
- Specification of perturbed atoms  
  (ch3oh/ch3oh_rot-trans_atomlist.ipynb)
- Calculations of BN-doped benzene derivatives  
  (benzene/benzene.ipynb, benzene_pcx2.ipynb, all_derivatives_pc2pcx2/all_benzene.ipynb)
- Calculations of BN-doped toluene derivatives  
  (toluene/calc_aqa.py)
- Efficient generation of chemical space of unique molecules  
  (toluene/efficient_gener_chem_space/efficient_gener_chem_space.py, benzene/efficient_gener_chem_space/gener_ccs.ipynb)
- Identifying alchemical enantiomers and diastereomers  
  (toluene/classify_alchemical_isomers/test_classifier.ipynb, benzene/classify_alchemical_isomers/get_alchemical_enantiomers.py)
- Relative energies between alchemical enantiomers and diastereomers  
  (benzene/relative_energies/calc_rel_ene.py, toluene/relative_energies/calc_rel_ene.py)
- SCF calculations of electronic energies of multiple molecules  
  (co/multiple_scf/co_multiscf.ipynb)
