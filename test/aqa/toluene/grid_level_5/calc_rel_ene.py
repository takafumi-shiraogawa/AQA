from aqa.alch_calc_utils import calc_relative_energies_from_output, \
    calc_scf_relative_energies_from_output, get_alch_diastereomers

alch_diastereomers = get_alch_diastereomers()
rel_ene = calc_relative_energies_from_output()
scf_rel_ene = calc_scf_relative_energies_from_output()

for i in range(len(alch_diastereomers)):
    print(alch_diastereomers[i][0], alch_diastereomers[i][1],
          rel_ene[0, i], rel_ene[1, i],
          scf_rel_ene[i], (rel_ene[0, i] - scf_rel_ene[i])*1000,
          (rel_ene[1, i] - scf_rel_ene[i])*1000)
