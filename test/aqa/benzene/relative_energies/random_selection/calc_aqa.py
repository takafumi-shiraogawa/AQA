# Set parameters
path_xyz_file = "./benzene.xyz"
mutation_sites = range(6)

# Calculation
from aqa.alch_calc_utils import read_xyz_file, generate_data_string
import basis_set_exchange as bse
nuclear_charges, coordinates = read_xyz_file(path_xyz_file)
atom_strings = [bse.lut.element_sym_from_Z(part, normalize=True) for part in nuclear_charges]
target_mol = generate_data_string(atom_strings, coordinates)
dft_functional = None
name_basis_set = "3-21G"
from aqa.alch_calc import alchemical_calculator as ac
ac_mol = ac(target_mol, name_basis_set, dft_functional, sites=mutation_sites, bse_off=False)
ac_mol.calc_all_derivatives()

# Write results
from aqa.alch_calc_utils import get_multi_ele_energies, write_csv_output_APDFT_energies, write_csv_output_ele_ene_derivatives
ele_enes_apdft = get_multi_ele_energies(ac_mol)
write_csv_output_APDFT_energies(ele_enes_apdft)
write_csv_output_ele_ene_derivatives(ac_mol)
