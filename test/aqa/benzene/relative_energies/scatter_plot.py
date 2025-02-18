import numpy as np
from utils import scatter_plot_relative_energies

scatter_range_enantiomers = np.linspace(0, 0.1, 6)
scatter_range_diastereomers = None
given_xy_range = {"scatter_plot_enantiomers_APDFT1":scatter_range_enantiomers,
                  "scatter_plot_enantiomers_APDFT3":scatter_range_enantiomers,
                  "scatter_plot_enantiomers_APDFT1and3":scatter_range_enantiomers,
                  "scatter_plot_diastereomers_APDFT1":scatter_range_diastereomers,
                  "scatter_plot_diastereomers_APDFT3":scatter_range_diastereomers,
                  "scatter_plot_diastereomers_APDFT1and3":scatter_range_diastereomers}

scatter_plot_relative_energies(given_range_x=given_xy_range, given_range_y=given_xy_range)
