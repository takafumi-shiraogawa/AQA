from alch_calc_utils import calc_relative_energies_from_output, \
    calc_scf_relative_energies_from_output, get_alch_isomers_indexes
from utils import write_relative_energies

path_xyz_file = "./toluene_cs.xyz"
alch_isomers = get_alch_isomers_indexes(path_xyz_file)
rel_ene = calc_relative_energies_from_output(path_xyz_file, alch_isomers)
scf_rel_ene = calc_scf_relative_energies_from_output(path_xyz_file, alch_isomers)
write_relative_energies(alch_isomers, rel_ene, scf_rel_ene=scf_rel_ene)
