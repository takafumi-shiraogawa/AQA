"""
Alchemical rules based estimates of properties of molecules or materials.
"""


def sum_rule_based_estimate(prop_ref, prop_right, trunc_order,
                            second_order_alch_derivs=None):
    """
    Rules based on a sum of perturbation expansions for alchemical enantiomers
    and diastereomers.

    Args:
        - prop_ref (float): property of the reference molecule
        - prop_right (float): property of the right molecule
        - trunc_order (int): truncation order of the perturbation expansion
        - second_order_alch_derivs (float): 2nd-order alchemical derivatives

    Returns:
        - float: property of the left molecule

    Note:
        - The truncation order 3 requires the 2nd-order alchemical derivatives.
    """
    if trunc_order not in [0, 1, 3]:
        raise ValueError("Truncation order must be 0 or 1 or 3.")

    if trunc_order == 0:
        return prop_ref
    elif trunc_order == 1:
        return 2.0 * prop_ref - prop_right
    elif trunc_order == 3:
        return 2.0 * prop_ref - prop_right + second_order_alch_derivs


def sum_rule_based_estimate_from_apdft(apdft_prop_left, apdft_prop_right,
                                       prop_right, trunc_order):
    """
    Rules based on a sum of perturbation expansions for alchemical enantiomers
    and diastereomers from APDFT properties.

    Args:
        - apdft_prop_left (float): APDFT property of the left molecule
        - apdft_prop_right (float): APDFT property of the right molecule
        - prop_right (float): property of the right molecule
        - trunc_order (int): truncation order of the perturbation expansion

    Returns:
        - float: property of the left molecule
    """
    if trunc_order not in [0, 1, 3]:
        raise ValueError("Truncation order must be 0 or 1 or 3.")

    if trunc_order == 0:
        if apdft_prop_left != apdft_prop_right:
            raise ValueError()
        return apdft_prop_left - prop_right

    elif trunc_order > 0:
        return apdft_prop_left + apdft_prop_right - prop_right


def sum_rule_based_difference_estimate(prop_ref, prop_right, trunc_order,
                                       second_order_alch_derivs=None):
    """
    Rules based on a sum of perturbation expansions for alchemical enantiomers
    and diastereomers.

    Args:
        - prop_ref (float): property of the reference molecule
        - prop_right (float): property of the right molecule
        - trunc_order (int): truncation order of the perturbation expansion
        - second_order_alch_derivs (float): 2nd-order alchemical derivatives

    Returns:
        - float: property of the left molecule

    Note:
        - The truncation order 3 requires the 2nd-order alchemical derivatives.
    """
    if trunc_order not in [0, 1, 3]:
        raise ValueError("Truncation order must be 0 or 1 or 3.")

    if trunc_order == 0:
        return prop_ref - prop_right
    elif trunc_order == 1:
        return 2.0 * (prop_ref - prop_right)
    elif trunc_order == 3:
        return 2.0 * (prop_ref - prop_right) + second_order_alch_derivs
