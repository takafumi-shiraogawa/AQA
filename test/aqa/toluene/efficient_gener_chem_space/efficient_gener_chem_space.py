from aqa.gener_chem_space import efficiently_generate_chemical_space, write_unique_nuclear_numbers_list, write_all_gjf
from aqa.alch_calc_utils import read_xyz_file

nuclear_charges, coordinates = read_xyz_file("./toluene_cs_opt_reorder.xyz")
unique_nuclear_numbers_list = efficiently_generate_chemical_space(nuclear_charges,
                                                                  coordinates,
                                                                  mutation_sites=range(6),
                                                                  maximum_atom_numbers_for_each_species={"B":3, "C":7, "N":3, "H":8},
                                                                  matrix_size=12)

write_unique_nuclear_numbers_list(unique_nuclear_numbers_list, "target_molecules.inp")
write_all_gjf(unique_nuclear_numbers_list, coordinates, flag_nested=True)
