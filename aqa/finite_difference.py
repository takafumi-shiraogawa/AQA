import numpy as np
from alch_calc import alchemical_calculator as ac

def get_finite_difference_coefficients(h, accuracy):
    """ Get finite difference coefficients.

    For h, sssume that h for x and y are the same.

    References:
        [1] Handbook of Mathematical Functions with Formulas, Graphs, and Mathematical Tables
        [2] "Combined Compact Difference Scheme for Linear Second-Order Partial Differential Equations with Mixed Derivative"
    """

    if accuracy not in ["h2", "h4"]:
        raise ValueError("Argument accuracy of set_stencil_coefficients should be 'h2' or 'h4' in the current implementation.")

    # 1d
    # x
    positions_finite_difference_1d_h2 = [[[1], [-1]]]
    finite_difference_1d_h2 = [np.asarray([1.0, -1.0]) / (2.0 * h)]
    positions_finite_difference_1d_h4 = [[[-2], [-1], [1], [2]]]
    finite_difference_1d_h4 = [np.asarray([1.0/12.0, -2.0/3.0, 2.0/3.0, -1.0/12.0]) / h]

    # 2d
    # xx
    # positions: [x]
    positions_finite_difference_2d_xx_h2 = [[[0], [1], [-1]]]
    finite_difference_2d_xx_h2 = [np.asarray([-2.0, 1.0, 1.0]) / (h ** 2.0)]
    positions_finite_difference_2d_xx_h4 = [[[0], [2], [1], [-1], [-2]]]
    finite_difference_2d_xx_h4 = [np.asarray([-30.0, -1.0, 16.0, 16.0, -1.0]) / (12.0 * (h ** 2.0))]

    # xy
    # positions: [x], [y], [xy]
    positions_finite_difference_2d_xy_h2 = [[[0], [1, 0], [-1, 0]],
                                            [[0, 1], [0, -1]],
                                            [[1, 1], [-1, -1]]]
    finite_difference_2d_xy_h2 = [np.asarray([-2.0, 1.0, 1.0]) / (-2.0 * (h ** 2.0)),  # x
                                  np.asarray([1.0, 1.0]) / (-2.0 * (h ** 2.0)),  # y
                                  np.asarray([-1.0, -1.0]) / (-2.0 * (h ** 2.0))]  # xy

    # Forth order mixed derivative requires analytical first derivatives for x and y.
    positions_finite_difference_2d_xy_h4 = [[[1, 0], [-1, 0]],
                                            [[0, 1], [0, -1]],
                                            [[1, 1], [-1, 1], [-1, -1], [1, -1]]]
    finite_difference_2d_xy_h4 = [np.asarray([1.0, -1.0]) / (2.0 * h),  # x
                                  np.asarray([1.0, -1.0]) / (2.0 * h),  # y
                                  np.asarray([1.0, -1.0, 1.0, -1.0]) / (-4.0 * (h ** 2.0))]  # xy

    if accuracy == "h2":
        dict_positions = {"x": positions_finite_difference_1d_h2,
                          "xx": positions_finite_difference_2d_xx_h2,
                          "xy": positions_finite_difference_2d_xy_h2}
        dict_coefficients = {"x": finite_difference_1d_h2,
                             "xx": finite_difference_2d_xx_h2,
                             "xy": finite_difference_2d_xy_h2}
    elif accuracy == "h4":
        raise NotImplementedError("Accuracy h4 is not implemented yet.")
        # TODO: Implement h4. It requires handling analytical first derivatives for x and y.

        dict_positions = {"x": positions_finite_difference_1d_h4,
                          "xx": positions_finite_difference_2d_xx_h4,
                          "xy": positions_finite_difference_2d_xy_h4}
        dict_coefficients = {"x": finite_difference_1d_h4,
                             "xx": finite_difference_2d_xx_h4,
                             "xy": finite_difference_2d_xy_h4}

    return dict_positions, dict_coefficients

def get_unique_perturbation_strength(stencil_positions):
    unique_perturbation_strength = []

    for i in stencil_positions.values():
        for j in i:
            for k in j:
                # convert to real number
                real_values = [float(l) for l in k]
                nonzero_real_values = [l for l in real_values if l != 0.0]
                if nonzero_real_values not in unique_perturbation_strength and \
                    real_values not in unique_perturbation_strength:
                    unique_perturbation_strength.append(real_values)

    return unique_perturbation_strength

def get_stencil_positions_corresponding_to_unique_perturbation_strength(stencil_positions):
    """ Get stencil positions corresponding to unique_perturbation_strength.
    Each obtained stencil position corresponds to unique_perturbation_strength.
    """

    num_unique_perturbation_strength = len(get_unique_perturbation_strength(stencil_positions))

    corresponding_stencil_positions_keys = [[] for _ in range(num_unique_perturbation_strength)]
    corresponding_stencil_positions = [[] for _ in range(num_unique_perturbation_strength)]
    unique_perturbation_strength = []

    for key, i in stencil_positions.items():
        for j in i:
            for k in j:
                # convert to real number
                real_values = [float(l) for l in k]
                nonzero_real_values = [l for l in real_values if l != 0.0]
                if nonzero_real_values not in unique_perturbation_strength and \
                    real_values not in unique_perturbation_strength:
                        unique_perturbation_strength.append(real_values)
                        corresponding_stencil_positions[len(unique_perturbation_strength) - 1].append(k)
                        corresponding_stencil_positions_keys[len(unique_perturbation_strength) - 1].append(key)
                else:
                    try:
                        index = unique_perturbation_strength.index(nonzero_real_values)
                    except:
                        index = unique_perturbation_strength.index(real_values)

                    corresponding_stencil_positions[index].append(k)
                    corresponding_stencil_positions_keys[index].append(key)

    return corresponding_stencil_positions_keys, corresponding_stencil_positions

def get_list_unique_additional_fractional_charges(perturbed_atom_sites, h, stencil_positions,
                                                  cost_estimate_mode=False):
    """ Get unique list of additional fractional charges for finite differentiation.

    Args:
        perturbed_atom_sites (list): List of perturbed atom sites.

    Returns:
        list: List of unique additional fractional charges for finite differentiation.
    """

    unique_perturbation_strength =  get_unique_perturbation_strength(stencil_positions)

    # Get all the list of fractional charges for finite differentiation
    # One site
    list_unique_additional_fractional_charges = []
    for i in range(len(perturbed_atom_sites)):
        for perturbation_strength in unique_perturbation_strength:
            if len(perturbation_strength) == 1:
                fractional_charges = [0.0 for _ in range(len(perturbed_atom_sites))]
                fractional_charges[i] += perturbation_strength[0] * h
                if fractional_charges not in list_unique_additional_fractional_charges:
                    list_unique_additional_fractional_charges.append(fractional_charges)
    # Two sites
    for i in range(len(perturbed_atom_sites)):
        for j in range(len(perturbed_atom_sites)):
            if j > i:
                for perturbation_strength in unique_perturbation_strength:
                    if len(perturbation_strength) == 2:
                        fractional_charges = [0.0 for _ in range(len(perturbed_atom_sites))]
                        fractional_charges[i] += perturbation_strength[0] * h
                        fractional_charges[j] += perturbation_strength[1] * h
                        if fractional_charges not in list_unique_additional_fractional_charges:
                            list_unique_additional_fractional_charges.append(fractional_charges)

    if not cost_estimate_mode:
        return list_unique_additional_fractional_charges
    else:
        return len(list_unique_additional_fractional_charges)

def get_stencil_positions_corresponding_to_unique_additional_fractional_charges(
    perturbed_atom_sites, h, stencil_positions):

    corresponding_stencil_positions_keys_to_unique_perturbation_strength, \
        corresponding_stencil_positions_to_unique_perturbation_strength = \
            get_stencil_positions_corresponding_to_unique_perturbation_strength(stencil_positions)

    unique_perturbation_strength = get_unique_perturbation_strength(stencil_positions)

    corresponding_stencil_positions_keys_to_unique_additional_fractional_charges = []
    corresponding_stencil_positions_to_unique_additional_fractional_charges = []
    atom_subgroup_list_unique_additional_fractional_charges = []

    # Get all the list of fractional charges for finite differentiation
    # One site
    list_unique_additional_fractional_charges = []
    for i in range(len(perturbed_atom_sites)):
        corresponding_stencil_positions_keys_to_unique_additional_fractional_charges.append([])
        corresponding_stencil_positions_to_unique_additional_fractional_charges.append([])
        atom_subgroup_list_unique_additional_fractional_charges.append([])

        for perturbation_strength in unique_perturbation_strength:
            if len(perturbation_strength) == 1:
                fractional_charges = [0.0 for _ in range(len(perturbed_atom_sites))]
                fractional_charges[i] += perturbation_strength[0] * h

                list_unique_additional_fractional_charges.append(fractional_charges)

                corresponding_stencil_positions_keys_to_unique_additional_fractional_charges[-1].append(
                    corresponding_stencil_positions_keys_to_unique_perturbation_strength[
                        unique_perturbation_strength.index(perturbation_strength)])
                corresponding_stencil_positions_to_unique_additional_fractional_charges[-1].append(
                    corresponding_stencil_positions_to_unique_perturbation_strength[
                        unique_perturbation_strength.index(perturbation_strength)])
                atom_subgroup_list_unique_additional_fractional_charges[-1].append(fractional_charges)

    # Two sites
    for i in range(len(perturbed_atom_sites)):
        for j in range(len(perturbed_atom_sites)):
            if j > i:
                corresponding_stencil_positions_keys_to_unique_additional_fractional_charges.append([])
                corresponding_stencil_positions_to_unique_additional_fractional_charges.append([])
                atom_subgroup_list_unique_additional_fractional_charges.append([])

                for perturbation_strength in unique_perturbation_strength:
                    if len(perturbation_strength) == 2:
                        fractional_charges = [0.0 for _ in range(len(perturbed_atom_sites))]
                        fractional_charges[i] += perturbation_strength[0] * h
                        fractional_charges[j] += perturbation_strength[1] * h

                        list_unique_additional_fractional_charges.append(fractional_charges)

                        corresponding_stencil_positions_keys_to_unique_additional_fractional_charges[-1].append(
                            corresponding_stencil_positions_keys_to_unique_perturbation_strength[
                                unique_perturbation_strength.index(perturbation_strength)])
                        corresponding_stencil_positions_to_unique_additional_fractional_charges[-1].append(
                            corresponding_stencil_positions_to_unique_perturbation_strength[
                                unique_perturbation_strength.index(perturbation_strength)])
                        atom_subgroup_list_unique_additional_fractional_charges[-1].append(fractional_charges)

    return corresponding_stencil_positions_keys_to_unique_additional_fractional_charges, \
        corresponding_stencil_positions_to_unique_additional_fractional_charges, \
        atom_subgroup_list_unique_additional_fractional_charges

def get_analytical_electronic_derivatives_index(perturbed_atom_sites, h, stencil_positions,
                                                index_atom_site, derivative_symbol, given_stencil_position,
                                                list_unique_additional_fractional_charges, debug_mode=False):
    if derivative_symbol not in ['x', 'xx', 'xy']:
        raise ValueError("derivative_symbol must be 'x', 'xx', or 'xy'.")

    corresponding_stencil_positions_keys_to_unique_additional_fractional_charges, \
        corresponding_stencil_positions_to_unique_additional_fractional_charges, \
            atom_subgroup_list_unique_additional_fractional_charges = \
                get_stencil_positions_corresponding_to_unique_additional_fractional_charges(
                    perturbed_atom_sites, h, stencil_positions)


    def get_all_indices(value, derivative_symbol):
        indices = []
        index = 0
        while True:
            try:
                index = value.index(derivative_symbol, index)
                indices.append(index)
                index += 1
            except ValueError:
                break
        return indices if indices else [None]

    if derivative_symbol == 'x':
        derivative_symbol_index = []

        # TODO: This assumes that only one symbol and should be generalized.
        for i, value in enumerate(corresponding_stencil_positions_keys_to_unique_additional_fractional_charges[index_atom_site]):
            try:
                derivative_symbol_index.append(value.index(derivative_symbol))
            except:
                derivative_symbol_index.append(None)

        unique_count = 0
        for i, candidate_stencil_position in enumerate(corresponding_stencil_positions_to_unique_additional_fractional_charges[index_atom_site]):
            if derivative_symbol_index[i] is not None:
                if given_stencil_position == candidate_stencil_position[derivative_symbol_index[i]]:
                    unique_count += 1
                    target_fractional_charges = atom_subgroup_list_unique_additional_fractional_charges[index_atom_site][i]

        if unique_count != 1:
            raise ValueError("unique_count is %i must be 1." % unique_count)

    elif derivative_symbol == 'xx':
        derivative_symbol_index = []

        # TODO: This assumes that only one symbol and should be generalized.
        for i, value in enumerate(corresponding_stencil_positions_keys_to_unique_additional_fractional_charges[index_atom_site]):
            try:
                derivative_symbol_index.append(value.index(derivative_symbol))
            except:
                derivative_symbol_index.append(None)
        if debug_mode:
            print("derivative_symbol_index:", derivative_symbol_index)
            print("corresponding_stencil_positions_to_unique_additional_fractional_charges[index_atom_site]")
            print(corresponding_stencil_positions_to_unique_additional_fractional_charges[index_atom_site])

        unique_count = 0
        for i, candidate_stencil_position in enumerate(corresponding_stencil_positions_to_unique_additional_fractional_charges[index_atom_site]):
            if derivative_symbol_index[i] is not None:
                if given_stencil_position == candidate_stencil_position[derivative_symbol_index[i]]:
                    unique_count += 1
                    target_fractional_charges = atom_subgroup_list_unique_additional_fractional_charges[index_atom_site][i]

        if unique_count != 1:
            raise ValueError("unique_count is %i must be 1." % unique_count)

    elif derivative_symbol == 'xy':
        derivative_symbol_indices = []
        for i, value in enumerate(corresponding_stencil_positions_keys_to_unique_additional_fractional_charges[index_atom_site]):
            indices = get_all_indices(value, derivative_symbol)
            derivative_symbol_indices.append(indices)

        if debug_mode:
            print("derivative_symbol_indices", derivative_symbol_indices)
            print("corresponding_stencil_positions_to_unique_additional_fractional_charges[index_atom_site]")
            print(corresponding_stencil_positions_to_unique_additional_fractional_charges[index_atom_site])

        unique_count = 0
        for i, candidate_stencil_position in enumerate(corresponding_stencil_positions_to_unique_additional_fractional_charges[index_atom_site]):
            for j, derivative_symbol_index in enumerate(derivative_symbol_indices[i]):
                if derivative_symbol_index is not None:
                    if given_stencil_position == candidate_stencil_position[derivative_symbol_index]:
                        unique_count += 1
                        target_fractional_charges = atom_subgroup_list_unique_additional_fractional_charges[index_atom_site][i]

        if unique_count != 1:
            raise ValueError("unique_count is %i must be 1." % unique_count)

    return list_unique_additional_fractional_charges.index(target_fractional_charges)

def calc_analytical_electronic_derivatives_with_fractional_charges(derivatives_rank,
    target_mol, name_basis_set, dft_functional, sites, list_unique_additional_fractional_charges,
    **kwargs):
    """ Calculate analytical electronic derivatives with fractional charges. """
    if derivatives_rank not in [0, 1, 2, 3]:
        raise ValueError("derivatives_rank must be 0, 1, 2, or 3.")

    analytical_electronic_derivatives = []
    for fractional_charges in list_unique_additional_fractional_charges:
        if derivatives_rank == 0:
            AP_skip = True
        else:
            AP_skip = False
        ac_mol_fc = ac(target_mol, name_basis_set, dft_functional, sites=sites, guess="1e",
                       fractional_charges_calc=True, fractional_charges=fractional_charges,
                       AP_skip=AP_skip, **kwargs)

        if derivatives_rank == 0:
            analytical_electronic_derivatives.append(ac_mol_fc.get_elec_energy())
        elif derivatives_rank == 1:
            analytical_electronic_derivatives.append(ac_mol_fc.ap.build_elec_gradient())
        elif derivatives_rank == 2:
            analytical_electronic_derivatives.append(ac_mol_fc.ap.build_elec_hessian())
        elif derivatives_rank == 3:
            analytical_electronic_derivatives.append(ac_mol_fc.ap.build_cubic())

        del ac_mol_fc

    return analytical_electronic_derivatives

def calc_analytical_electronic_derivatives(derivatives_rank,
    target_mol, name_basis_set, dft_functional, sites, ac_mol=None, **kwargs):
    """ Calculate analytical electronic derivatives with fractional charges. """
    if derivatives_rank not in [0, 1, 2, 3]:
        raise ValueError("derivatives_rank must be 0, 1, 2, or 3.")

    # For consistency with calc_analytical_electronic_derivatives_with_fractional_charges,
    # guess="1e" is used.
    if ac_mol is None:
        ac_mol = ac(target_mol, name_basis_set, dft_functional, sites=sites, guess="1e", **kwargs)

        if derivatives_rank == 0:
            return ac_mol, ac_mol.get_elec_energy()
        elif derivatives_rank == 1:
            return ac_mol, ac_mol.ap.build_elec_gradient()
        elif derivatives_rank == 2:
            return ac_mol, ac_mol.ap.build_elec_hessian()
        elif derivatives_rank == 3:
            return ac_mol, ac_mol.ap.build_cubic()

    else:
        if derivatives_rank == 0:
            return ac_mol.ap.get_elec_energy()
        elif derivatives_rank == 1:
            return ac_mol.ap.build_elec_gradient()
        elif derivatives_rank == 2:
            return ac_mol.ap.build_elec_hessian()
        elif derivatives_rank == 3:
            return ac_mol.ap.build_cubic()

def calc_finite_difference(target_mol, name_basis_set, dft_functional, sites, h, accuracy,
                           base_derivative_rank, mode="calculation", return_ac_mol=False,
                           **kwargs):
    """ Calculate finite difference derivatives.

    Note:
        Both analytical and numerical derivatives adopt the same approximation that ignores
        the dependence of the basis set and numerical grids on the nuclear charges.
    """
    if mode not in ["calculation", "check_accuracy", "cost_estimation", "debug"]:
        raise ValueError("mode must be 'calculation' or 'check_accuracy' or 'cost_estimation' or 'debug'.")

    if base_derivative_rank not in [0, 1, 2, 3]:
        raise ValueError("base_derivative_rank must be 0, 1, 2, or 3.")

    stencil_positions, stencil_coefficients = get_finite_difference_coefficients(h, accuracy)
    list_unique_additional_fractional_charges = get_list_unique_additional_fractional_charges(
        sites, h, stencil_positions)

    if mode == "debug":
        corresponding_stencil_positions_keys_to_unique_additional_fractional_charges, \
            corresponding_stencil_positions_to_unique_additional_fractional_charges, \
            atom_subgroup_list_unique_additional_fractional_charges = \
                get_stencil_positions_corresponding_to_unique_additional_fractional_charges(
                    sites, h, stencil_positions)

        print("corresponding_stencil_positions_keys_to_unique_additional_fractional_charges")
        for i in corresponding_stencil_positions_keys_to_unique_additional_fractional_charges:
            print(i)
        print()

        print("corresponding_stencil_positions_to_unique_additional_fractional_charges")
        for i in corresponding_stencil_positions_to_unique_additional_fractional_charges:
            print(i)
        print()

        print("atom_subgroup_list_unique_additional_fractional_charges")
        for i in atom_subgroup_list_unique_additional_fractional_charges:
            print(i)
        print()

        print("stencil_positions")
        print(stencil_positions)
        print()

        print("stencil_coefficients")
        print(stencil_coefficients)
        print()

        print("list_unique_additional_fractional_charges")
        print(list_unique_additional_fractional_charges)
        print()

    if mode == "cost_estimation":
        print("The number of QM calculations required for finite differentiation:")
        print(get_list_unique_additional_fractional_charges(
            sites, h, stencil_positions, cost_estimate_mode=True))
        return

    # Calculate analytical electronic derivatives for finite difference derivatives
    analytical_electronic_derivatives = calc_analytical_electronic_derivatives_with_fractional_charges(
        base_derivative_rank, target_mol, name_basis_set, dft_functional, sites,
        list_unique_additional_fractional_charges, **kwargs)

    if mode == "debug":
        print()
        print("analytical_electronic_derivatives")
        for i, value in enumerate(analytical_electronic_derivatives):
            print(i, value)
        print()

    # Calculate finite difference derivatives
    if mode == "debug":
        flag_debug_mode = True
    else:
        flag_debug_mode = False

    # x
    if mode == "debug":
        print("x")
    finite_difference_derivatives_x = []
    for index_atom_site in range(len(sites)):
        if mode == "debug":
            print("atom_site: {}".format(sites[index_atom_site]))
        if base_derivative_rank == 0:
            finite_difference_derivative = 0.0
        elif base_derivative_rank == 1:
            finite_difference_derivative = [0.0 for _ in range(len(sites))]
        elif base_derivative_rank == 2:
            finite_difference_derivative = [[0.0 for _ in range(len(sites))] for _ in range(len(sites))]
        elif base_derivative_rank == 3:
            finite_difference_derivative = [[[0.0 for _ in range(len(sites))] for _ in range(len(sites))] for _ in range(len(sites))]
        for i, stencil_position in enumerate(stencil_positions["x"][0]):
            index_derivative = get_analytical_electronic_derivatives_index(sites, h, stencil_positions,
                                                                           index_atom_site, 'x',
                                                                           stencil_position,
                                                                           list_unique_additional_fractional_charges,
                                                                           flag_debug_mode)
            finite_difference_derivative += stencil_coefficients["x"][0][i] * \
                analytical_electronic_derivatives[index_derivative]

            if mode == "debug":
                print("stencil position: {}".format(stencil_position))
                print("index_derivative: {}".format(index_derivative))
                print("stencil_coefficients['x'][0][i]: {}".format(stencil_coefficients["x"][0][i]))
                print("list_unique_additional_fractional_charges[index_derivative]: {}".format(
                    list_unique_additional_fractional_charges[index_derivative]))
                print()

        finite_difference_derivatives_x.append(finite_difference_derivative)
    if mode == "debug":
        print("finite_difference_derivatives_x:")
        print(finite_difference_derivatives_x)
        print()

    # xx
    if mode == "debug":
        print("xx")
    finite_difference_derivatives_xx = []
    for index_atom_site in range(len(sites)):
        if mode == "debug":
            print("atom_site: {}".format(sites[index_atom_site]))
        finite_difference_derivative = 0.0
        for i, stencil_position in enumerate(stencil_positions["xx"][0]):
            index_derivative = get_analytical_electronic_derivatives_index(sites, h, stencil_positions,
                                                                           index_atom_site, 'xx',
                                                                           stencil_position,
                                                                           list_unique_additional_fractional_charges,
                                                                           flag_debug_mode)
            finite_difference_derivative += stencil_coefficients["xx"][0][i] * analytical_electronic_derivatives[index_derivative]

            if mode == "debug":
                print("stencil position: {}".format(stencil_position))
                print("index_derivative: {}".format(index_derivative))
                print("stencil_coefficients['xx'][0][i]: {}".format(stencil_coefficients["xx"][0][i]))
                print("list_unique_additional_fractional_charges[index_derivative]: {}".format(
                    list_unique_additional_fractional_charges[index_derivative]))
                print()

        finite_difference_derivatives_xx.append(finite_difference_derivative)
    if mode == "debug":
        print("finite_difference_derivatives_xx:")
        print(finite_difference_derivatives_xx)
        print()

    # xy
    if mode == "debug":
        print("xy")
    index_atom_site_xy = len(sites) - 1
    finite_difference_derivatives_xy = []
    for index_atom_site1 in range(len(sites)):
        for index_atom_site2 in range(len(sites)):
            if index_atom_site2 > index_atom_site1:
                index_atom_site_xy += 1
                finite_difference_derivative = 0.0

                if mode == "debug":
                    print("atom_site1: {}".format(sites[index_atom_site1]))
                    print("atom_site2: {}".format(sites[index_atom_site2]))

                # x, y, xy
                for index_variable in range(3):
                    for i, stencil_position in enumerate(stencil_positions["xy"][index_variable]):
                        if index_variable == 0:
                            index_atom_site = index_atom_site1
                        elif index_variable == 1:
                            index_atom_site = index_atom_site2
                        elif index_variable == 2:
                            index_atom_site = index_atom_site_xy

                        index_derivative = get_analytical_electronic_derivatives_index(sites, h, stencil_positions,
                                                                                       index_atom_site, 'xy',
                                                                                       stencil_position,
                                                                                       list_unique_additional_fractional_charges,
                                                                                       flag_debug_mode)
                        finite_difference_derivative += stencil_coefficients["xy"][index_variable][i] * analytical_electronic_derivatives[index_derivative]

                        if mode == "debug":
                            print("stencil position: {}".format(stencil_position))
                            print("index_derivative: {}".format(index_derivative))
                            print("stencil_coefficients['xy'][index_variable][i]: {}".format(
                                stencil_coefficients["xy"][index_variable][i]))
                            print("list_unique_additional_fractional_charges[index_derivative]: {}".format(
                                list_unique_additional_fractional_charges[index_derivative]))
                            print()

                finite_difference_derivatives_xy.append(finite_difference_derivative)
    if mode == "debug":
        print("finite_difference_derivatives_xy:")
        print(finite_difference_derivatives_xy)
        print()

    # Assume the third order derivatives
    if base_derivative_rank == 0:
        finite_difference_derivatives_general_xy = np.zeros((len(sites), len(sites)))
    elif base_derivative_rank == 1:
        finite_difference_derivatives_general_xy = np.zeros((len(sites), len(sites), len(sites)))
    elif base_derivative_rank == 2:
        finite_difference_derivatives_general_xy = np.zeros((len(sites), len(sites), len(sites), len(sites)))
    elif base_derivative_rank == 3:
        finite_difference_derivatives_general_xy = np.zeros((len(sites), len(sites), len(sites), len(sites), len(sites)))

    count_xy = -1
    for index_atom_site1 in range(len(sites)):
        for index_atom_site2 in range(len(sites)):
            if index_atom_site1 == index_atom_site2:
                if base_derivative_rank == 0:
                    finite_difference_derivatives_general_xy[index_atom_site1, index_atom_site1] = \
                        finite_difference_derivatives_xx[index_atom_site1]
                elif base_derivative_rank == 1:
                    finite_difference_derivatives_general_xy[:, index_atom_site1, index_atom_site1] = \
                        finite_difference_derivatives_xx[index_atom_site1]
                elif base_derivative_rank == 2:
                    finite_difference_derivatives_general_xy[:, :, index_atom_site1, index_atom_site1] = \
                        finite_difference_derivatives_xx[index_atom_site1]
                elif base_derivative_rank == 3:
                    finite_difference_derivatives_general_xy[:, :, :, index_atom_site1, index_atom_site1] = \
                        finite_difference_derivatives_xx[index_atom_site1]
            elif index_atom_site2 > index_atom_site1:
                count_xy += 1
                if base_derivative_rank == 0:
                    finite_difference_derivatives_general_xy[index_atom_site1, index_atom_site2] = \
                        finite_difference_derivatives_xy[count_xy]
                    finite_difference_derivatives_general_xy[index_atom_site2, index_atom_site1] = \
                        finite_difference_derivatives_xy[count_xy]
                elif base_derivative_rank == 1:
                    finite_difference_derivatives_general_xy[:, index_atom_site1, index_atom_site2] = \
                        finite_difference_derivatives_xy[count_xy]
                    finite_difference_derivatives_general_xy[:, index_atom_site2, index_atom_site1] = \
                        finite_difference_derivatives_xy[count_xy]
                elif base_derivative_rank == 2:
                    finite_difference_derivatives_general_xy[:, :, index_atom_site1, index_atom_site2] = \
                        finite_difference_derivatives_xy[count_xy]
                    finite_difference_derivatives_general_xy[:, :, index_atom_site2, index_atom_site1] = \
                        finite_difference_derivatives_xy[count_xy]
                elif base_derivative_rank == 3:
                    finite_difference_derivatives_general_xy[:, :, :, index_atom_site1, index_atom_site2] = \
                        finite_difference_derivatives_xy[count_xy]
                    finite_difference_derivatives_general_xy[:, :, :, index_atom_site2, index_atom_site1] = \
                        finite_difference_derivatives_xy[count_xy]

    if mode == "check_accuracy":
        if base_derivative_rank not in [0, 1, 2]:
            raise NotImplementedError("For 'check_accuracy' mode, base_derivative_rank > 2 is not implemented yet.")

        # Calculate analytical derivatives
        ac_mol, calculated_analytical_electronic_derivatives_rank_plus1 = calc_analytical_electronic_derivatives(
            base_derivative_rank + 1, target_mol, name_basis_set, dft_functional, sites, **kwargs)
        if base_derivative_rank in [0, 1]:
            calculated_analytical_electronic_derivatives_rank_plus2 = calc_analytical_electronic_derivatives(
                base_derivative_rank + 2, target_mol, name_basis_set, dft_functional, sites, ac_mol=ac_mol, **kwargs)

            del ac_mol

            calculated_analytical_electronic_derivatives_rank_plus2_xx = []
            calculated_analytical_electronic_derivatives_rank_plus2_xy = []
            for i in range(len(sites)):
                for j in range(len(sites)):
                    if i == j:
                        if base_derivative_rank == 0:
                            calculated_analytical_electronic_derivatives_rank_plus2_xx.append(
                                calculated_analytical_electronic_derivatives_rank_plus2[i, i])
                        elif base_derivative_rank == 1:
                            calculated_analytical_electronic_derivatives_rank_plus2_xx.append(
                                calculated_analytical_electronic_derivatives_rank_plus2[:, i, i])
                    elif j > i:
                        if base_derivative_rank == 0:
                            calculated_analytical_electronic_derivatives_rank_plus2_xy.append(
                                calculated_analytical_electronic_derivatives_rank_plus2[i, j])
                        elif base_derivative_rank == 1:
                            calculated_analytical_electronic_derivatives_rank_plus2_xy.append(
                                calculated_analytical_electronic_derivatives_rank_plus2[:, i, j])

        if base_derivative_rank in [0, 1, 2]:
            print("----- Check Rank + 1 ---------------------------------------------")
            print("Analytical Rank + 1 derivatives:")
            print(calculated_analytical_electronic_derivatives_rank_plus1)
            print()
            print("Numerical Rank + 1 derivatives:")
            print(finite_difference_derivatives_x)
            print()
            print("Error:")
            print("finite_difference_derivatives_x - calculated_analytical_electronic_derivatives_rank_plus1")
            print(finite_difference_derivatives_x - calculated_analytical_electronic_derivatives_rank_plus1)
            print("----- Check Rank + 1 ---------------------------------------------")
            print()

        if base_derivative_rank in [0, 1]:
            print("----- Check Rank + 2 (diagonal terms) ---------------------------------------------")
            print("Analytical Rank + 2 derivatives (diagonal terms):")
            print(calculated_analytical_electronic_derivatives_rank_plus2_xx)
            print()
            print("Numerical Rank + 2 derivatives (diagonal terms):")
            print(finite_difference_derivatives_xx)
            print()
            print("Error:")
            print(np.asarray(finite_difference_derivatives_xx) - np.asarray(calculated_analytical_electronic_derivatives_rank_plus2_xx))
            print("----- Check Rank + 2 (diagonal terms) ---------------------------------------------")
            print()

            print("----- Check Rank + 2 (off-diagonal terms) ---------------------------------------------")
            print("Analytical Rank + 2 derivatives (off-diagonal terms):")
            print(calculated_analytical_electronic_derivatives_rank_plus2_xy)
            print()
            print("Numerical Rank + 2 derivatives (off-diagonal terms):")
            print(finite_difference_derivatives_xy)
            print()
            print("Error:")
            print(np.asarray(finite_difference_derivatives_xy) - np.asarray(calculated_analytical_electronic_derivatives_rank_plus2_xy))
            print("----- Check Rank + 2 (off-diagonal terms) ---------------------------------------------")
            print()

    if return_ac_mol:
        if base_derivative_rank == 0:
            _, analytical_electronic_energy = calc_analytical_electronic_derivatives(
                0, target_mol, name_basis_set, dft_functional, sites, **kwargs)
        elif base_derivative_rank == 1:
            ac_mol, analytical_electronic_energy = calc_analytical_electronic_derivatives(
                0, target_mol, name_basis_set, dft_functional, sites, **kwargs)
            analytical_1st_electronic_energy_derivatives = calc_analytical_electronic_derivatives(
                1, target_mol, name_basis_set, dft_functional, sites, ac_mol=ac_mol, **kwargs)
        elif base_derivative_rank == 2:
            ac_mol, analytical_electronic_energy = calc_analytical_electronic_derivatives(
                0, target_mol, name_basis_set, dft_functional, sites, **kwargs)
            analytical_1st_electronic_energy_derivatives = calc_analytical_electronic_derivatives(
                1, target_mol, name_basis_set, dft_functional, sites, ac_mol=ac_mol, **kwargs)
            analytical_2nd_electronic_energy_derivatives = calc_analytical_electronic_derivatives(
                2, target_mol, name_basis_set, dft_functional, sites, ac_mol=ac_mol, **kwargs)
        elif base_derivative_rank == 3:
            ac_mol, analytical_electronic_energy = calc_analytical_electronic_derivatives(
                0, target_mol, name_basis_set, dft_functional, sites, **kwargs)
            analytical_1st_electronic_energy_derivatives = calc_analytical_electronic_derivatives(
                1, target_mol, name_basis_set, dft_functional, sites, ac_mol=ac_mol, **kwargs)
            analytical_2nd_electronic_energy_derivatives = calc_analytical_electronic_derivatives(
                2, target_mol, name_basis_set, dft_functional, sites, ac_mol=ac_mol, **kwargs)
            analytical_3rd_electronic_energy_derivatives = calc_analytical_electronic_derivatives(
                3, target_mol, name_basis_set, dft_functional, sites, ac_mol=ac_mol, **kwargs)

    if not return_ac_mol:
        return finite_difference_derivatives_x, finite_difference_derivatives_general_xy
    else:
        ac_mol.ap.fourth_order_derivatives = finite_difference_derivatives_x
        ac_mol.ap.fifth_order_derivatives = finite_difference_derivatives_general_xy
        return ac_mol

def calc_finite_difference_APDFT_relative_electronic_energies(pvec, ac_mol):
    """ sc_mol is obtained from calc_finite_difference with return_ac_mol=True
        and base_derivative_rank=3.
    """

    # APDFT1, APDFT3, and APDFT5 relative electronic energies
    return ac_mol.ap.relative_elec_APDFT1(pvec), ac_mol.ap.relative_elec_APDFT3(pvec), \
        ac_mol.ap.relative_elec_APDFT5(pvec)

def calc_finite_difference_APDFT_electronic_energies(pvec, ac_mol):
    """ sc_mol is obtained from calc_finite_difference with return_ac_mol=True
        and base_derivative_rank=3.
    """

    # APDFT1-5 electronic energies
    return ac_mol.ap.elec_APDFT1(pvec), ac_mol.ap.elec_APDFT2(pvec), \
            ac_mol.ap.elec_APDFT3(pvec), ac_mol.ap.elec_APDFT4(pvec), \
            ac_mol.ap.elec_APDFT5(pvec)


if __name__ == '__main__':
    # Inputs
    # Molecules
    target_mol = "C 0 0 0; O 0 0 1.1"
    sites = [0, 1]
    # Perturbation
    h = 0.001
    accuracy = "h2"

    # QM calculations for finite differentiation
    dft_functional = "pbe0"  # "lda,vwn"
    # name_basis_set = "sto-3g"
    # name_basis_set = "6-311g"
    name_basis_set = "cc-pvtz"

    calc_finite_difference(target_mol, name_basis_set, dft_functional, sites, h, accuracy,
                           0, mode="check_accuracy")
    # calc_finite_difference(target_mol, name_basis_set, dft_functional, sites, h, accuracy,
    #                        1, mode="check_accuracy")
    # calc_finite_difference(target_mol, name_basis_set, dft_functional, sites, h, accuracy,
    #                        1, mode="debug")
