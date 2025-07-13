import os
import numpy as np
import itertools as it
import copy
from .mini_qml import representations as amr
from scipy.linalg import norm
from .alch_calc_utils import read_xyz_file, write_gjf, write_gjf_gen

# 0. Group all those structures that have the same stoichiometry
# 1. Calculate norm of Coulomb matrix and group all those structures that have the same norm
# 2. Calculate BoB representation based distance between all structures that are in the same group of 1 and group all those that have zero distance
# 3. Calculate sorted Coulomb matrix distance to identify duplicates
# 4. Calculate sorted Coulomb matrix distance to identify duplicates


# 0. Group all those structures that have the same stoichiometry
# 1. Calculate norm of Coulomb matrix and group all those structures that have the same norm
def generate_mutated_nuclear_numbers_and_coulomb_group_indexes(nuclear_numbers, target_atom_positions, nuclear_coordinates, matrix_size):
    """
    This function generates all possible mutations of nuclear numbers and returns them along with their identification numbers
    for the same norm of the Coulomb matrix.

    Args:
    - nuclear_numbers (list): A list of integers representing the nuclear numbers of atoms.
    - target_atom_positions (list): A list of lists representing the target positions of atoms.
    - nuclear_coordinates (list): A list of lists representing the nuclear coordinates of atoms.

    Returns:
    - A tuple containing three elements:
        - all_nuclear_numbers_list (list): A list of lists representing all possible mutated nuclear numbers.
        - index_nuclear_numbers_same_norm_coulomb (list): A list of integers representing the identification numbers of the mutated nuclear numbers.
        - num_subgroup_same_norm_coulomb (list): A list of integers representing the number of subgroups with the same norm of the Coulomb matrix for each number of pairs of mutated atoms.
    """
    # Groups of all those structures that have the same norm
    all_nuclear_numbers_list = []
    all_nuclear_numbers_list.append(nuclear_numbers)

    num_mut_atoms = int(len(target_atom_positions))
    num_total_pairs_mut_atoms = int(num_mut_atoms / 2)

    # index_same_norm_coulomb contains indexes of nuclear charges (all_nuclear_numbers_list)
    # in subgroups with the same norm of the Coulomb matrix for each number of pairs of mutated atoms
    index_nuclear_numbers_same_norm_coulomb = [[] for _ in range(num_total_pairs_mut_atoms + 1)]
    index_nuclear_numbers_same_norm_coulomb[0].append([0])

    # num_subgroup_same_norm_coulomb contais the number of subgroups with the same norm of the Coulomb matrix
    # for each number of pairs of mutated atoms
    num_subgroup_same_norm_coulomb = np.zeros((num_total_pairs_mut_atoms + 1), dtype=int)
    num_subgroup_same_norm_coulomb[0] = 1

    # Convert into numpy array
    nuclear_numbers = np.array(nuclear_numbers)
    target_atom_positions = np.array(target_atom_positions)

    # Exceptional handling
    if len(nuclear_numbers) < len(target_atom_positions):
        raise ValueError("The number of mutations is larger than nuclear numbers.")

    index_nuclear_numbers = 0

    for num_pairs_mut_atoms in range(1, num_total_pairs_mut_atoms + 1):
        identification_number_list = []

        for mut_atom_positions in it.combinations(target_atom_positions, num_pairs_mut_atoms * 2):
            for positive_mut_atom_positions in it.combinations(
                mut_atom_positions, num_pairs_mut_atoms):

                index_nuclear_numbers += 1

                # Get mutated nuclear numbers
                mut_nuclear_numbers = copy.copy(nuclear_numbers)

                # Calculate the positions to increment and decrement
                inc_indices = np.array(list(positive_mut_atom_positions))
                dec_indices = np.setdiff1d(mut_atom_positions, inc_indices)

                # Increment and decrement in one step
                mut_nuclear_numbers[inc_indices] += 1
                mut_nuclear_numbers[dec_indices] -= 1

                # Calculate the Coulomb matrix, NOT sorted one
                coulomb_matrix = amr.generate_coulomb_matrix(
                    mut_nuclear_numbers, nuclear_coordinates, size=matrix_size)

                # Calculate the norm of the Coulomb matrix
                norm_coulomb_matrix = norm(coulomb_matrix)

                # Get the identification number for classifying the nuclear numbers
                # from the norm of the Coulomb matrix

                # Get the identification number for classifying the nuclear numbers
                # Here, round(norm_coulomb_matrix, 5) is used to avoid the floating point error, but it is arbitrary.
                identification_number = round(norm_coulomb_matrix, 5)

                # Save the identification number if it is not in the list
                if identification_number not in identification_number_list:
                    identification_number_list.append(identification_number)
                    group_index = len(identification_number_list) - 1
                    index_nuclear_numbers_same_norm_coulomb[num_pairs_mut_atoms].append([])
                else:
                    group_index = identification_number_list.index(identification_number)

                index_nuclear_numbers_same_norm_coulomb[num_pairs_mut_atoms][group_index].append(index_nuclear_numbers)
                all_nuclear_numbers_list.append(mut_nuclear_numbers)

        num_subgroup_same_norm_coulomb[num_pairs_mut_atoms] = len(index_nuclear_numbers_same_norm_coulomb[num_pairs_mut_atoms])

    return all_nuclear_numbers_list, index_nuclear_numbers_same_norm_coulomb, num_subgroup_same_norm_coulomb


# 2. Calculate BoB representation based distance between all structures that are in the same group of
#    1 and group all those that have zero distance
def generate_bob_group_indexes(nuclear_coordinates, all_nuclear_numbers_list,
                               index_nuclear_numbers_same_norm_coulomb, num_subgroup_same_norm_coulomb,
                               maximum_atom_numbers_for_each_species, matrix_size):
    num_total_pairs_mut_atoms = int(len(num_subgroup_same_norm_coulomb) - 1)

    index_nuclear_numbers_zero_bob_distance = [[] for _ in range(num_total_pairs_mut_atoms + 1)]
    # index_nuclear_numbers_zero_bob_distance[0].append([0])

    num_subgroup_zero_bob_distance = np.zeros((num_total_pairs_mut_atoms + 1), dtype=int)
    num_subgroup_zero_bob_distance[0] = 1

    # For each number of pairs of mutated atoms
    for num_pairs_mut_atoms in range(num_total_pairs_mut_atoms + 1):

        # For each subgroup with the same norm of the Coulomb matrix
        for index_subgroup in range(num_subgroup_same_norm_coulomb[num_pairs_mut_atoms]):
            identification_number_list = []

            offset = len(index_nuclear_numbers_zero_bob_distance[num_pairs_mut_atoms])

            first_index_in_subgroup = index_nuclear_numbers_same_norm_coulomb[num_pairs_mut_atoms][index_subgroup][0]
            bob_representation = amr.generate_bob(all_nuclear_numbers_list[first_index_in_subgroup], nuclear_coordinates,
                                                  size=matrix_size, asize=maximum_atom_numbers_for_each_species)
            identification_number = bob_representation
            identification_number_list.append(identification_number)
            group_index = len(identification_number_list) - 1
            index_nuclear_numbers_zero_bob_distance[num_pairs_mut_atoms].append([])
            index_nuclear_numbers_zero_bob_distance[num_pairs_mut_atoms][group_index + offset].append(first_index_in_subgroup)

            # For each pair of mutated atoms in the subgroup
            # index_in_subgroup is the index of nuclear numbers in the list of all nuclear numbers
            for index_in_subgroup in index_nuclear_numbers_same_norm_coulomb[num_pairs_mut_atoms][index_subgroup][1:]:

                bob_representation = amr.generate_bob(all_nuclear_numbers_list[index_in_subgroup], nuclear_coordinates,
                                                      size=matrix_size, asize=maximum_atom_numbers_for_each_species)
                identification_number = bob_representation

                judges = np.linalg.norm(np.subtract(identification_number_list, identification_number), axis=1) >= 1.e-10

                if np.all(judges):
                    identification_number_list.append(identification_number)
                    group_index = len(identification_number_list) - 1
                    index_nuclear_numbers_zero_bob_distance[num_pairs_mut_atoms].append([])
                else:
                    first_false_index = np.where(judges == False)[0][0]
                    group_index = first_false_index

                index_nuclear_numbers_zero_bob_distance[num_pairs_mut_atoms][group_index + offset].append(index_in_subgroup)

        num_subgroup_zero_bob_distance[num_pairs_mut_atoms] = len(index_nuclear_numbers_zero_bob_distance[num_pairs_mut_atoms])

    return index_nuclear_numbers_zero_bob_distance, num_subgroup_zero_bob_distance


# 3. Calculate Coulomb matrix eigenvalue distance to identify duplicates
def generate_eigenvalue_coulomb_group_indexes(nuclear_coordinates, all_nuclear_numbers_list,
                                          index_nuclear_numbers_zero_bob_distance, num_subgroup_zero_bob_distance, matrix_size):
    num_total_pairs_mut_atoms = int(len(num_subgroup_zero_bob_distance) - 1)

    unique_nuclear_numbers_list_in_subgroup = [[] for _ in range(num_total_pairs_mut_atoms + 1)]

    # For each number of pairs of mutated atoms
    for num_pairs_mut_atoms in range(num_total_pairs_mut_atoms + 1):

        # For each subgroup with the zero BoB distance
        for index_subgroup in range(num_subgroup_zero_bob_distance[num_pairs_mut_atoms]):

            identification_number_list = []
            initial_index_in_subgroup = index_nuclear_numbers_zero_bob_distance[num_pairs_mut_atoms][index_subgroup][0]
            eigenvalues_coulomb_matrix = amr.generate_eigenvalue_coulomb_matrix(all_nuclear_numbers_list[initial_index_in_subgroup],
                                                                                nuclear_coordinates, matrix_size)
            identification_number_list.append(eigenvalues_coulomb_matrix)
            unique_nuclear_numbers_list_in_subgroup[num_pairs_mut_atoms].append(all_nuclear_numbers_list[initial_index_in_subgroup])

            # For each pair of mutated atoms in the subgroup
            # index_in_subgroup is the index of nuclear numbers in the list of all nuclear numbers
            for index_in_subgroup in index_nuclear_numbers_zero_bob_distance[num_pairs_mut_atoms][index_subgroup][1:]:

                eigenvalues_coulomb_matrix = amr.generate_eigenvalue_coulomb_matrix(all_nuclear_numbers_list[index_in_subgroup],
                                                                                    nuclear_coordinates, matrix_size)
                judge = np.all(np.linalg.norm(np.subtract(identification_number_list,
                                                          eigenvalues_coulomb_matrix), axis=1) >= 0.01, axis=0)
                if judge:
                    identification_number_list.append(eigenvalues_coulomb_matrix)
                    unique_nuclear_numbers_list_in_subgroup[num_pairs_mut_atoms].append(all_nuclear_numbers_list[index_in_subgroup])

    return unique_nuclear_numbers_list_in_subgroup

# 4. Calculate sorted Coulomb matrix distance to identify duplicates
def generate_unique_nuclear_numbers_list(unique_nuclear_numbers_list_in_subgroup, nuclear_coordinates, matrix_size):
    num_total_pairs_mut_atoms = int(len(unique_nuclear_numbers_list_in_subgroup) - 1)

    unique_nuclear_numbers_list = [[] for _ in range(num_total_pairs_mut_atoms + 1)]

    # For each number of pairs of mutated atoms
    for num_pairs_mut_atoms in range(num_total_pairs_mut_atoms + 1):
        identification_number_list = []
        eigenvalues_coulomb_matrix = amr.generate_eigenvalue_coulomb_matrix(
            unique_nuclear_numbers_list_in_subgroup[num_pairs_mut_atoms][0], nuclear_coordinates, matrix_size)
        identification_number_list.append(eigenvalues_coulomb_matrix)

        unique_nuclear_numbers_list[num_pairs_mut_atoms].append(
            np.array(unique_nuclear_numbers_list_in_subgroup[num_pairs_mut_atoms][0]))

        for nuclear_numbers in unique_nuclear_numbers_list_in_subgroup[num_pairs_mut_atoms][1:]:
            eigenvalues_coulomb_matrix = amr.generate_eigenvalue_coulomb_matrix(
            nuclear_numbers, nuclear_coordinates, matrix_size)

            judge = np.all(np.linalg.norm(np.subtract(identification_number_list,
                                                      eigenvalues_coulomb_matrix), axis=1) >= 0.01, axis=0)
            if judge:
                identification_number_list.append(eigenvalues_coulomb_matrix)
                unique_nuclear_numbers_list[num_pairs_mut_atoms].append(nuclear_numbers)

    return unique_nuclear_numbers_list


def write_unique_nuclear_numbers_list(unique_nuclear_numbers_list, filename="target_molecules.inp"):
    with open(filename, "w") as f:
        for nuclear_numbers_subgroup in unique_nuclear_numbers_list:
            for nuclear_numbers in nuclear_numbers_subgroup:
                f.write(','.join(map(str, list(nuclear_numbers))) + "\n")


def write_all_gjf(unique_nuclear_numbers_list, coords, flag_nested=False,
                  dirname="target_geometries", filename="geom_target"):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    pre_name = dirname + "/" + filename
    if not flag_nested:
        for idx, target in enumerate(unique_nuclear_numbers_list):
            name = pre_name + str(idx + 1)
            write_gjf(target, coords, name)
    else:
        count = 0
        for target_subgroup in unique_nuclear_numbers_list:
            for target in target_subgroup:
                count += 1
                name = pre_name + str(count)
                write_gjf(target, coords, name)

def write_all_gjf_gen(unique_nuclear_numbers_list, coords, basis_set_dict, flag_nested=False,
                      dirname="target_geometries", filename="geom_target"):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    pre_name = dirname + "/" + filename
    if not flag_nested:
        for idx, target in enumerate(unique_nuclear_numbers_list):
            name = pre_name + str(idx + 1)
            write_gjf_gen(target, coords, name, basis_set_dict)
    else:
        count = 0
        for target_subgroup in unique_nuclear_numbers_list:
            for target in target_subgroup:
                count += 1
                name = pre_name + str(count)
                write_gjf_gen(target, coords, name, basis_set_dict)

def efficiently_generate_chemical_space(atom_numbers, nuclear_coords, mutation_sites,
                                        maximum_atom_numbers_for_each_species, matrix_size):
    """
    This function generates the chemical space of the mutated structures efficiently.

    Args:
    - atom_numbers (list): A list of integers representing the nuclear numbers of atoms.
    - nuclear_coords (list): A list of lists representing the nuclear coordinates of atoms.
    - mutation_sites (list): A list of integers representing the mutation sites.
    - maximum_atom_numbers_for_each_species (dict): A dictionary representing the maximum number of atoms for each species,
        e.g., {"C": 7, "H": 16, "O": 3, "N": 3, "S": 1}.
    - matirx_size (int): An integer representing the size of the Coulomb matrix.

    Returns:
    - unique_nuclear_numbers_list (list): A list of lists representing the unique nuclear numbers.
    """

    # Note that the last argument assumes that the number of atoms does not increase after mutation.
    all_nuclear_numbers_list, index_same_norm_coulomb, num_subgroup_same_norm_coulomb = \
        generate_mutated_nuclear_numbers_and_coulomb_group_indexes(atom_numbers,
                                                                   mutation_sites,
                                                                   nuclear_coords,
                                                                   len(nuclear_coords))

    index_nuclear_numbers_zero_bob_distance, num_subgroup_zero_bob_distance = \
        generate_bob_group_indexes(nuclear_coords,
                                   all_nuclear_numbers_list,
                                   index_same_norm_coulomb,
                                   num_subgroup_same_norm_coulomb,
                                   maximum_atom_numbers_for_each_species,
                                   matrix_size)

    unique_nuclear_numbers_list_in_subgroup = \
        generate_eigenvalue_coulomb_group_indexes(nuclear_coords,
                                                  all_nuclear_numbers_list,
                                                  index_nuclear_numbers_zero_bob_distance,
                                                  num_subgroup_zero_bob_distance,
                                                  matrix_size)

    unique_nuclear_numbers_list = \
        generate_unique_nuclear_numbers_list(unique_nuclear_numbers_list_in_subgroup,
                                             nuclear_coords,
                                             matrix_size)

    return unique_nuclear_numbers_list


def main():
    nuclear_charges, coordinates = read_xyz_file("./reference_mol.xyz")
    unique_nuclear_numbers_list = efficiently_generate_chemical_space(nuclear_charges,
                                                                      coordinates,
                                                                      range(6),
                                                                      {"B":3, "C":6, "N":3, "H":6},
                                                                      6)

    write_unique_nuclear_numbers_list(unique_nuclear_numbers_list)
    write_all_gjf(unique_nuclear_numbers_list, coordinates, flag_nested=True)


if __name__ == '__main__':
    main()
