from alch_calc_utils import calc_relative_energies_from_output, calc_scf_relative_energies_from_output
from alch_calc_utils import calc_scf_ele_energies_from_nuc_charge_vecs, write_scf_ele_ene_dat
from alch_calc_utils import read_xyz_file, generate_data_string
from alch_calc_utils import get_randomly_selected_alch_isomers, get_mols_in_alch_isomers, \
    get_selected_unique_nuclear_numbers_list_from_mol_indexes
from utils import write_relative_energies
import basis_set_exchange as bse

# Inputs of SCF calculations
mutation_sites = range(6)
path_xyz_file = "./benzene.xyz"
name_basis_set = '3-21G'
dft_functional = None
num_selected_pairs_enantiomers = 2

selected_alch_isomers = get_randomly_selected_alch_isomers(num_selected_pairs_enantiomers=num_selected_pairs_enantiomers)
mol_indexes_in_alch_isomers = get_mols_in_alch_isomers(selected_alch_isomers)

# Generate randomly_selected_target_molecules.inp and randomly_selected_target_molecules_indexes.dat
selected_nuc_charges_vecs = get_selected_unique_nuclear_numbers_list_from_mol_indexes(mol_indexes_in_alch_isomers)
nuclear_charges, coordinates = read_xyz_file(path_xyz_file)
atom_strings = [bse.lut.element_sym_from_Z(part, normalize=True) for part in nuclear_charges]
target_mol = generate_data_string(atom_strings, coordinates)

# SCF calculations of randomly selected molecules
ele_enes = calc_scf_ele_energies_from_nuc_charge_vecs(
    target_mol, mutation_sites, nuc_charges_vecs=selected_nuc_charges_vecs, dft_functional=dft_functional,
    name_basis_set=name_basis_set, bse_off=False, parallel=True, num_parallel=4, num_cpus=4)
write_scf_ele_ene_dat(ele_enes, output_file="randomly_selected_scf_ele_ene.dat")

# Calculate relative energies of randomly selected pairs of alchemical isomers
selected_scf_relative_energies = calc_scf_relative_energies_from_output(path_xyz_file,
                                                                        unified_alch_isomers=selected_alch_isomers,
                                                                        path_output_file="randomly_selected_scf_ele_ene.dat",
                                                                        mode='selection')
selected_apdft_relative_energies = calc_relative_energies_from_output(path_xyz_file, unified_alch_isomers=selected_alch_isomers,
                                                                      path_output_file="apdft_energies.csv")
write_relative_energies(selected_alch_isomers, selected_apdft_relative_energies,
                        scf_rel_ene=selected_scf_relative_energies,
                        file_path_enantiomers="./randomly_selected_rel_ene_enantiomers.dat",
                        file_path_diastereomers="./randomly_selected_rel_ene_diastereomers.dat")
