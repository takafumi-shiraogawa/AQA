import numpy as np
from utils import scatter_plot_relative_energies, plot_relative_energies

range_enantiomers = list(range(0, 5, 1))
range_diastereomers = list(range(0, 19, 6))
given_x_range = {"plot_enantiomers_APDFT1":range_enantiomers,
                 "plot_enantiomers_APDFT3":range_enantiomers,
                 "plot_enantiomers_APDFT1and3":range_enantiomers,
                 "plot_enantiomers_APDFT1-3":range_enantiomers,
                 "plot_enantiomers_APDFT1and3andSCF":range_enantiomers,
                 "plot_enantiomers_APDFT1and3-SCF":range_enantiomers,
                 "plot_diastereomers_APDFT1":range_diastereomers,
                 "plot_diastereomers_APDFT3":range_diastereomers,
                 "plot_diastereomers_APDFT1and3":range_diastereomers,
                 "plot_diastereomers_APDFT1-3":range_diastereomers,
                 "plot_diastereomers_APDFT1and3andSCF":range_diastereomers,
                 "plot_diastereomers_APDFT1and3-SCF":range_diastereomers}
given_y_range = {"plot_enantiomers_APDFT1":None,
                 "plot_enantiomers_APDFT3":None,
                 "plot_enantiomers_APDFT1and3":None,
                 "plot_enantiomers_APDFT1-3":np.linspace(0.0, 0.03, 4),
                 "plot_enantiomers_APDFT1and3andSCF":np.linspace(0, 0.03, 4),
                 "plot_enantiomers_APDFT1and3-SCF":np.linspace(0, 0.02, 5),
                 "plot_diastereomers_APDFT1":None,
                 "plot_diastereomers_APDFT3":None,
                 "plot_diastereomers_APDFT1and3":None,
                 "plot_diastereomers_APDFT1-3":np.linspace(0.0, 0.04, 5),
                 "plot_diastereomers_APDFT1and3andSCF":None,
                 "plot_diastereomers_APDFT1and3-SCF":np.linspace(0.0, 0.04, 5)}
scatter_range_diastereomers = np.linspace(0, 16, 5)
scatter_range_enantiomers = np.linspace(0.0, 0.04, 5)
given_xy_range = {"scatter_plot_enantiomers_APDFT1":scatter_range_enantiomers,
                  "scatter_plot_enantiomers_APDFT3":scatter_range_enantiomers,
                  "scatter_plot_enantiomers_APDFT1and3":scatter_range_enantiomers,
                  "scatter_plot_diastereomers_APDFT1":scatter_range_diastereomers,
                  "scatter_plot_diastereomers_APDFT3":scatter_range_diastereomers,
                  "scatter_plot_diastereomers_APDFT1and3":scatter_range_diastereomers}

scatter_plot_relative_energies(given_range_x=given_xy_range, given_range_y=given_xy_range,
                               file_path_enantiomers="randomly_selected_rel_ene_enantiomers.dat",
                               file_path_diastereomers="randomly_selected_rel_ene_diastereomers.dat")
plot_relative_energies(given_range_x=given_x_range, given_range_y=given_y_range,
                       file_path_enantiomers="randomly_selected_rel_ene_enantiomers.dat",
                       file_path_diastereomers="randomly_selected_rel_ene_diastereomers.dat")
# scatter_plot_relative_energies(file_path_enantiomers="randomly_selected_rel_ene_enantiomers.dat",
#                                file_path_diastereomers="randomly_selected_rel_ene_diastereomers.dat")
