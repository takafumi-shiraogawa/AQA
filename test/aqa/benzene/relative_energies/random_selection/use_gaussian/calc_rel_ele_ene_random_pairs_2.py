from alch_calc_utils import calc_relative_energies_from_output, calc_scf_relative_energies_from_output
from utils import read_alch_isomers, write_relative_energies

path_xyz_file = "./benzene.xyz"

selected_alch_isomers = read_alch_isomers(file_path_enantiomers="randomly_selected_enantiomers.dat",
                                          file_path_diastereomers="randomly_selected_diastereomers.dat")

# Assume that the SCF calculations have been done and the output file (randomly_selected_scf_ele_ene.dat) is available
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
