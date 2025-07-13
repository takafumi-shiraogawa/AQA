from aqa.alch_calc_utils import read_xyz_file, generate_data_string
from aqa.alch_calc_utils import get_randomly_selected_alch_isomers, get_mols_in_alch_isomers, \
    get_selected_unique_nuclear_numbers_list_from_mol_indexes
from aqa.utils import write_alch_isomers
import basis_set_exchange as bse
from aqa.gener_chem_space import write_all_gjf_gen

# Inputs of SCF calculations
path_xyz_file = "./benzene.xyz"
basis_set_dict = {1:bse.get_basis("pc-2",fmt="gaussian94",elements=[1], header=False),
                  5:bse.get_basis("pcX-2",fmt="gaussian94",elements=[5], header=False),
                  6:bse.get_basis("pcX-2",fmt="gaussian94",elements=[6], header=False),
                  7:bse.get_basis("pcX-2",fmt="gaussian94",elements=[7], header=False)}
num_selected_pairs_enantiomers = 2

selected_alch_isomers = get_randomly_selected_alch_isomers(num_selected_pairs_enantiomers=num_selected_pairs_enantiomers)
mol_indexes_in_alch_isomers = get_mols_in_alch_isomers(selected_alch_isomers)

# Generate randomly_selected_target_molecules.inp and randomly_selected_target_molecules_indexes.dat
selected_nuc_charges_vecs = get_selected_unique_nuclear_numbers_list_from_mol_indexes(mol_indexes_in_alch_isomers)
nuclear_charges, coordinates = read_xyz_file(path_xyz_file)
atom_strings = [bse.lut.element_sym_from_Z(part, normalize=True) for part in nuclear_charges]
target_mol = generate_data_string(atom_strings, coordinates)

# write randomly selected alchemical isomers
write_alch_isomers(selected_alch_isomers, output_file_prefix="randomly_selected")

# Convert nuclear charges vectors to nuclear charges
# TODO: remove this very redundant step
selected_nuc_charges = []
for selected_nuc_charges_vec in selected_nuc_charges_vecs:
    selected_nuc_charges.append(selected_nuc_charges_vec + nuclear_charges)

# write Gaussian inputs of randomly selected molecules
write_all_gjf_gen(selected_nuc_charges, coordinates, basis_set_dict)
