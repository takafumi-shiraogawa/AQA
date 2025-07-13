import basis_set_exchange as bse

target_mol = """
 C                  0.00389100    0.90882400    0.00000000
 C                  0.00721400    0.19373700   -1.19405500
 C                  0.00721400   -1.19254800   -1.19710300
 C                  0.00612000   -1.89206800    0.00000000
 C                  0.00721400   -1.19254800    1.19710300
 C                  0.00721400    0.19373700    1.19405500
 H                  0.01240500    0.73222400   -2.13552200
 H                  0.01184800   -1.72839800   -2.13871300
 H                  0.00914900   -2.97506300    0.00000000
 H                  0.01184800   -1.72839800    2.13871300
 H                  0.01240500    0.73222400    2.13552200
 C                 -0.02734900    2.40735700    0.00000000
 H                 -1.05675000    2.77687400    0.00000000
 H                  0.46499300    2.81580200    0.88343200
 H                  0.46499300    2.81580200   -0.88343200
"""
dft_functional = "pbe0"
name_basis_set = {"H":bse.get_basis("pc-2",fmt="nwchem",elements=[1]),'C':bse.get_basis("pcX-2",fmt="nwchem",elements=[6])}

from aqa.alch_calc import alchemical_calculator as ac

ac_mol = ac(target_mol, name_basis_set, dft_functional, sites=[0,1,2,3,4,5], bse_off=True, grid_level=5)

ac_mol.calc_all_derivatives()

from aqa.alch_calc_utils import get_multi_ele_energies, write_csv_output_APDFT_energies, write_csv_output_ele_ene_derivatives

ele_enes_apdft = get_multi_ele_energies(ac_mol)
write_csv_output_APDFT_energies(ele_enes_apdft)
write_csv_output_ele_ene_derivatives(ac_mol)

