# Set parameters
path_xyz_file = "./benzene.xyz"
mutation_sites = range(6)

# Calculation
from aqa.alch_calc_utils import read_xyz_file, generate_data_string
import basis_set_exchange as bse
nuclear_charges, coordinates = read_xyz_file(path_xyz_file)
atom_strings = [bse.lut.element_sym_from_Z(part, normalize=True) for part in nuclear_charges]
target_mol = generate_data_string(atom_strings, coordinates)
dft_functional = "pbe0"

from aqa.alch_calc_utils import calc_scf_ele_energies_from_nuc_charge_vecs, write_scf_ele_ene_dat
import basis_set_exchange as bse
name_basis_set = '3-21G'
ele_enes = (calc_scf_ele_energies_from_nuc_charge_vecs(
      target_mol, mutation_sites, nuc_charges_vecs=None, dft_functional=None,
      name_basis_set=name_basis_set, bse_off=False, parallel=True, num_parallel=4, num_cpus=4))
write_scf_ele_ene_dat(ele_enes)
