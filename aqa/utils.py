import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd

# utils.py has minimal dependency on libraries
# in contrast to alch_calc_utils.py

def read_target_molecules(file_path="./target_molecules.inp"):
    data = []

    with open(file_path, 'r') as file:
        for line in file:
            values = line.strip().split(',')
            data.append([int(value) for value in values])

    return np.array(data)

def read_coordinate_from_xyz_file(file_path="./target_coord.xyz"):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    num_atoms = int(lines[0].strip())
    # atoms = []
    coordinates = []
    for line in lines[2:num_atoms + 2]:
        parts = line.split()
        # atomic_number = int(parts[0])
        x, y, z = map(float, parts[1:])
        # atoms.append(atomic_number)
        coordinates.append([x, y, z])

    coordinates = np.array(coordinates)
    return coordinates

def write_relative_energies(alch_isomers, rel_ene, scf_rel_ene=None,
                            file_path_enantiomers="./rel_ene_enantiomers.dat",
                            file_path_diastereomers="./rel_ene_diastereomers.dat",
                            flag_apdft5=False):
    # if scf_rel_ene is None:
    #     print("Index1", "Index2", "APDFT1 (Ha)", "APDFT3 (Ha)")
    # else:
    #     print("Index1", "Index2", "APDFT1 (Ha)", "APDFT3 (Ha)", "DFT (Ha)", "APDFT1 error (mHa)", "APDFT3 error (mHa)")
    if not flag_apdft5:
        with open(file_path_enantiomers, 'w') as f:
            for i in range(len(alch_isomers[0])):
                # Shift indexes to start from 1
                if scf_rel_ene is None:
                    print(alch_isomers[0][i][0] + 1, alch_isomers[0][i][1] + 1,
                        rel_ene["enantiomers"]["APDFT1"][i], rel_ene["enantiomers"]["APDFT3"][i],
                        file=f)
                else:
                    print(alch_isomers[0][i][0] + 1, alch_isomers[0][i][1] + 1,
                        rel_ene["enantiomers"]["APDFT1"][i], rel_ene["enantiomers"]["APDFT3"][i],
                        scf_rel_ene["enantiomers"][i],
                        (rel_ene["enantiomers"]["APDFT1"][i] - scf_rel_ene["enantiomers"][i])*1000,
                        (rel_ene["enantiomers"]["APDFT3"][i] - scf_rel_ene["enantiomers"][i])*1000, file=f)

        with open(file_path_diastereomers, 'w') as f:
            for i in range(len(alch_isomers[1])):
                # Shift indexes to start from 1
                if scf_rel_ene is None:
                    print(alch_isomers[1][i][0] + 1, alch_isomers[1][i][1] + 1,
                        rel_ene["diastereomers"]["APDFT1"][i], rel_ene["diastereomers"]["APDFT3"][i],
                        file=f)
                else:
                    print(alch_isomers[1][i][0] + 1, alch_isomers[1][i][1] + 1,
                        rel_ene["diastereomers"]["APDFT1"][i], rel_ene["diastereomers"]["APDFT3"][i],
                        scf_rel_ene["diastereomers"][i],
                        (rel_ene["diastereomers"]["APDFT1"][i] - scf_rel_ene["diastereomers"][i])*1000,
                        (rel_ene["diastereomers"]["APDFT3"][i] - scf_rel_ene["diastereomers"][i])*1000, file=f)
    else:
        with open(file_path_enantiomers, 'w') as f:
            for i in range(len(alch_isomers[0])):
                # Shift indexes to start from 1
                if scf_rel_ene is None:
                    print(alch_isomers[0][i][0] + 1, alch_isomers[0][i][1] + 1,
                        rel_ene["enantiomers"]["APDFT1"][i], rel_ene["enantiomers"]["APDFT3"][i],
                        rel_ene["enantiomers"]["APDFT5"][i], file=f)
                else:
                    print(alch_isomers[0][i][0] + 1, alch_isomers[0][i][1] + 1,
                        rel_ene["enantiomers"]["APDFT1"][i], rel_ene["enantiomers"]["APDFT3"][i],
                         rel_ene["enantiomers"]["APDFT5"][i], scf_rel_ene["enantiomers"][i],
                        (rel_ene["enantiomers"]["APDFT1"][i] - scf_rel_ene["enantiomers"][i])*1000,
                        (rel_ene["enantiomers"]["APDFT3"][i] - scf_rel_ene["enantiomers"][i])*1000,
                        (rel_ene["enantiomers"]["APDFT5"][i] - scf_rel_ene["enantiomers"][i])*1000, file=f)

        with open(file_path_diastereomers, 'w') as f:
            for i in range(len(alch_isomers[1])):
                # Shift indexes to start from 1
                if scf_rel_ene is None:
                    print(alch_isomers[1][i][0] + 1, alch_isomers[1][i][1] + 1,
                        rel_ene["diastereomers"]["APDFT1"][i], rel_ene["diastereomers"]["APDFT3"][i],
                        rel_ene["diastereomers"]["APDFT5"][i], file=f)
                else:
                    print(alch_isomers[1][i][0] + 1, alch_isomers[1][i][1] + 1,
                        rel_ene["diastereomers"]["APDFT1"][i], rel_ene["diastereomers"]["APDFT3"][i],
                        rel_ene["diastereomers"]["APDFT5"][i], scf_rel_ene["diastereomers"][i],
                        (rel_ene["diastereomers"]["APDFT1"][i] - scf_rel_ene["diastereomers"][i])*1000,
                        (rel_ene["diastereomers"]["APDFT3"][i] - scf_rel_ene["diastereomers"][i])*1000,
                        (rel_ene["diastereomers"]["APDFT5"][i] - scf_rel_ene["diastereomers"][i])*1000, file=f)

def write_alch_isomers(alch_isomers, output_file_prefix=None):
    if output_file_prefix is None:
        output_file_enantiomers = "alch_isomers_enantiomers.dat"
        output_file_diastereomers = "alch_isomers_diastereomers.dat"
    else:
        output_file_enantiomers = output_file_prefix + "_enantiomers.dat"
        output_file_diastereomers = output_file_prefix + "_diastereomers.dat"

    with open(output_file_enantiomers, 'w') as f:
        for i in range(len(alch_isomers[0])):
            # Shift indexes to start from 1
            print(alch_isomers[0][i][0] + 1, alch_isomers[0][i][1] + 1, file=f)

    with open(output_file_diastereomers, 'w') as f:
        for i in range(len(alch_isomers[1])):
            # Shift indexes to start from 1
            print(alch_isomers[1][i][0] + 1, alch_isomers[1][i][1] + 1, file=f)

def calc_nuclei_repulsion_energy(charges, coordinates, unit_angstrom=True):
    if unit_angstrom:
        angstrom_to_bohr = 1.8897261339212517
    else:
        angstrom_to_bohr = 1.0
    natoms = len(charges)
    ret = 0.0
    for i in range(natoms):
        for j in range(i + 1, natoms):
            d = np.linalg.norm(coordinates[i] - coordinates[j])
            ret += charges[i] * charges[j] / (d * angstrom_to_bohr)
    return ret

def calc_nuclei_repulsion_energies(name_target_molecules_list_file="./target_molecules.inp",
                                   name_coord_file="./target_coord.xyz"):
    coords = read_coordinate_from_xyz_file(name_coord_file)
    target_molecules_list = read_target_molecules(name_target_molecules_list_file)

    nuclear_repulsion_energies = np.zeros(len(target_molecules_list))
    for i, target_molecule in enumerate(target_molecules_list):
        nuclear_repulsion_energies[i] = calc_nuclei_repulsion_energy(target_molecule, coords)
    return nuclear_repulsion_energies

def calc_relative_nuclear_repulsion_energies(alch_isomers, name_coord_file="./target_coord.xyz"):
    coords = read_coordinate_from_xyz_file(name_coord_file)
    target_molecules_list = read_target_molecules()

    nuclear_repulsion_energies = np.zeros(len(target_molecules_list))
    for i, target_molecule in enumerate(target_molecules_list):
        nuclear_repulsion_energies[i] = calc_nuclei_repulsion_energy(target_molecule, coords)

    relative_nuclear_repulsion_energies = [[], []]
    for i, alch_enantiomer in enumerate(alch_isomers[0]):
        relative_nuclear_repulsion_energies[0].append(
            nuclear_repulsion_energies[alch_enantiomer[0] - 1] - nuclear_repulsion_energies[alch_enantiomer[1] - 1])
    for i, alch_diastereomer in enumerate(alch_isomers[1]):
        relative_nuclear_repulsion_energies[1].append(
            nuclear_repulsion_energies[alch_diastereomer[0] - 1] - nuclear_repulsion_energies[alch_diastereomer[1] - 1])

    return relative_nuclear_repulsion_energies

def read_relative_energies_from_outputs(flag_total_energy=False, name_coord_file="./target_coord.xyz",
                                        file_path_enantiomers="rel_ene_enantiomers.dat",
                                        file_path_diastereomers="rel_ene_diastereomers.dat"):
    """ Importantly, obtained indexes of molecules in alch_isomers start from 1, not 0. """
    # TODO: fix the case of only one pair of enantiomers or diastereomers
    #       The temporal solution is to copy the first line.

    # load data from file
    data_rel_ene_enantiomers = np.loadtxt(file_path_enantiomers)
    data_rel_ene_diastereomers = np.loadtxt(file_path_diastereomers)

    if data_rel_ene_enantiomers.size == 0:
        flag_enantiomers = False
    else:
        flag_enantiomers = True

    if data_rel_ene_diastereomers.size == 0:
        flag_diastereomers = False
    else:
        flag_diastereomers = True

    if flag_enantiomers:
        if data_rel_ene_enantiomers.shape[1] == 4:
            flag_scf_rel_ene = False
            flag_APDFT5 = False
        elif data_rel_ene_enantiomers.shape[1] == 5:
            flag_scf_rel_ene = False
            flag_APDFT5 = True
        elif data_rel_ene_enantiomers.shape[1] == 7:
            flag_scf_rel_ene = True
            flag_APDFT5 = False
        elif data_rel_ene_enantiomers.shape[1] == 9:
            flag_scf_rel_ene = True
            flag_APDFT5 = True
        else:
            raise ValueError("The number of columns in rel_ene_enantiomers.dat is not correct.")

    if flag_diastereomers:
        if data_rel_ene_diastereomers.shape[1] == 4:
            flag_scf_rel_ene = False
            flag_APDFT5 = False
        elif data_rel_ene_diastereomers.shape[1] == 5:
            flag_scf_rel_ene = False
            flag_APDFT5 = True
        elif data_rel_ene_diastereomers.shape[1] == 7:
            flag_scf_rel_ene = True
            flag_APDFT5 = False
        elif data_rel_ene_diastereomers.shape[1] == 9:
            flag_scf_rel_ene = True
            flag_APDFT5 = True
        else:
            raise ValueError("The number of columns in rel_ene_diastereomers.dat is not correct.")

    # alch_isomers[0]: enantiomers
    # alch_isomers[1]: diastereomers
    alch_isomers = []

    # Relative energies
    rel_ene = {}

    if not flag_APDFT5:
        if flag_enantiomers:
            alch_isomers.append(data_rel_ene_enantiomers[:, 0:2].astype(int))
            rel_ene["enantiomers"] = {"APDFT1": data_rel_ene_enantiomers[:, 2],
                                    "APDFT3": data_rel_ene_enantiomers[:, 3]}
            if flag_scf_rel_ene:
                rel_ene["enantiomers"]["SCF"] = data_rel_ene_enantiomers[:, 4]
                rel_ene["enantiomers"]["APDFT1-SCF"] = data_rel_ene_enantiomers[:, 5]
                rel_ene["enantiomers"]["APDFT3-SCF"] = data_rel_ene_enantiomers[:, 6]
        else:
            alch_isomers.append([])

        if flag_diastereomers:
            alch_isomers.append(data_rel_ene_diastereomers[:, 0:2].astype(int))
            rel_ene["diastereomers"] = {"APDFT1":data_rel_ene_diastereomers[:, 2],
                                        "APDFT3":data_rel_ene_diastereomers[:, 3]}
            if flag_scf_rel_ene:
                rel_ene["diastereomers"]["SCF"] = data_rel_ene_diastereomers[:, 4]
                rel_ene["diastereomers"]["APDFT1-SCF"] = data_rel_ene_diastereomers[:, 5]
                rel_ene["diastereomers"]["APDFT3-SCF"] = data_rel_ene_diastereomers[:, 6]
        else:
            alch_isomers.append([])
    else:
        if flag_enantiomers:
            alch_isomers.append(data_rel_ene_enantiomers[:, 0:2].astype(int))
            rel_ene["enantiomers"] = {"APDFT1": data_rel_ene_enantiomers[:, 2],
                                    "APDFT3": data_rel_ene_enantiomers[:, 3],
                                    "APDFT5": data_rel_ene_enantiomers[:, 4]}
            if flag_scf_rel_ene:
                rel_ene["enantiomers"]["SCF"] = data_rel_ene_enantiomers[:, 5]
                rel_ene["enantiomers"]["APDFT1-SCF"] = data_rel_ene_enantiomers[:, 6]
                rel_ene["enantiomers"]["APDFT3-SCF"] = data_rel_ene_enantiomers[:, 7]
                rel_ene["enantiomers"]["APDFT5-SCF"] = data_rel_ene_enantiomers[:, 8]
        else:
            alch_isomers.append([])

        if flag_diastereomers:
            alch_isomers.append(data_rel_ene_diastereomers[:, 0:2].astype(int))
            rel_ene["diastereomers"] = {"APDFT1":data_rel_ene_diastereomers[:, 2],
                                        "APDFT3":data_rel_ene_diastereomers[:, 3],
                                        "APDFT5":data_rel_ene_diastereomers[:, 4]}
            if flag_scf_rel_ene:
                rel_ene["diastereomers"]["SCF"] = data_rel_ene_diastereomers[:, 5]
                rel_ene["diastereomers"]["APDFT1-SCF"] = data_rel_ene_diastereomers[:, 6]
                rel_ene["diastereomers"]["APDFT3-SCF"] = data_rel_ene_diastereomers[:, 7]
                rel_ene["diastereomers"]["APDFT5-SCF"] = data_rel_ene_diastereomers[:, 8]
        else:
            alch_isomers.append([])

    if flag_total_energy:
        relative_nuclear_repulsion_energies = calc_relative_nuclear_repulsion_energies(alch_isomers, name_coord_file)

        if flag_enantiomers:
            rel_ene["enantiomers"]["APDFT1"] += relative_nuclear_repulsion_energies[0]
            rel_ene["enantiomers"]["APDFT3"] += relative_nuclear_repulsion_energies[0]
            if flag_APDFT5:
                rel_ene["enantiomers"]["APDFT5"] += relative_nuclear_repulsion_energies[0]
            if flag_scf_rel_ene:
                rel_ene["enantiomers"]["SCF"] += relative_nuclear_repulsion_energies[0]

        if flag_diastereomers:
            rel_ene["diastereomers"]["APDFT1"] += relative_nuclear_repulsion_energies[1]
            rel_ene["diastereomers"]["APDFT3"] += relative_nuclear_repulsion_energies[1]
            if flag_APDFT5:
                rel_ene["diastereomers"]["APDFT5"] += relative_nuclear_repulsion_energies[1]
            if flag_scf_rel_ene:
                rel_ene["diastereomers"]["SCF"] += relative_nuclear_repulsion_energies[1]

    return alch_isomers, rel_ene

def read_alch_isomers(file_path_enantiomers="enantiomers.dat",
                      file_path_diastereomers="diastereomers.dat"):
    """ Importantly, obtained indexes of molecules in alch_isomers start from "0".
        This is different from the above function read_relative_energies_from_outputs.
    """

    # load data from file
    data_rel_ene_enantiomers = np.loadtxt(file_path_enantiomers)
    data_rel_ene_diastereomers = np.loadtxt(file_path_diastereomers)

    if data_rel_ene_enantiomers.size == 0:
        flag_enantiomers = False
    else:
        flag_enantiomers = True

    if data_rel_ene_diastereomers.size == 0:
        flag_diastereomers = False
    else:
        flag_diastereomers = True

    # alch_isomers[0]: enantiomers
    # alch_isomers[1]: diastereomers
    alch_isomers = []

    if flag_enantiomers:
        alch_isomers.append(data_rel_ene_enantiomers[:, 0:2].astype(int) - 1)
    else:
        alch_isomers.append([])

    if flag_diastereomers:
        alch_isomers.append(data_rel_ene_diastereomers[:, 0:2].astype(int) - 1)
    else:
        alch_isomers.append([])

    return alch_isomers

def calc_maes_relative_energies_from_outputs(file_path_enantiomers="rel_ene_enantiomers.dat",
                                             file_path_diastereomers="rel_ene_diastereomers.dat"):
    _, rel_ene = read_relative_energies_from_outputs(file_path_enantiomers=file_path_enantiomers,
                                                     file_path_diastereomers=file_path_diastereomers)

    mae_relative_energies = {}
    if "enantiomers" in rel_ene:
        if "APDFT1-SCF" in rel_ene["enantiomers"]:
            mae_enantiomers_APDFT1 = np.mean(np.abs(rel_ene["enantiomers"]["APDFT1-SCF"]))
            mae_enantiomers_APDFT3 = np.mean(np.abs(rel_ene["enantiomers"]["APDFT3-SCF"]))
            mae_relative_energies["enantiomers"] = {"APDFT1": mae_enantiomers_APDFT1,
                                                    "APDFT3": mae_enantiomers_APDFT3}
            if "APDFT5-SCF" in rel_ene["enantiomers"]:
                mae_enantiomers_APDFT5 = np.mean(np.abs(rel_ene["enantiomers"]["APDFT5-SCF"]))
                mae_relative_energies["enantiomers"]["APDFT5"] = mae_enantiomers_APDFT5

    if "diastereomers" in rel_ene:
        if "APDFT1-SCF" in rel_ene["diastereomers"]:
            mae_diastereomers_APDFT1 = np.mean(np.abs(rel_ene["diastereomers"]["APDFT1-SCF"]))
            mae_diastereomers_APDFT3 = np.mean(np.abs(rel_ene["diastereomers"]["APDFT3-SCF"]))
            mae_relative_energies["diastereomers"] = {"APDFT1": mae_diastereomers_APDFT1,
                                                      "APDFT3": mae_diastereomers_APDFT3}
            if "APDFT5-SCF" in rel_ene["diastereomers"]:
                mae_diastereomers_APDFT5 = np.mean(np.abs(rel_ene["diastereomers"]["APDFT5-SCF"]))
                mae_relative_energies["diastereomers"]["APDFT5"] = mae_diastereomers_APDFT5

    return mae_relative_energies

def save_maes_relative_energies_from_outputs(file_path_enantiomers="rel_ene_enantiomers.dat",
                                             file_path_diastereomers="rel_ene_diastereomers.dat"):
    # Convert the dictionary into DataFrame
    df = pd.DataFrame(calc_maes_relative_energies_from_outputs(file_path_enantiomers,
                                                               file_path_diastereomers))

    # Save the DataFrame to a CSV file
    df.to_csv('maes_relative_energies.csv')

def make_relative_energies_positive(alch_isomers, rel_ene, target_type="SCF", flag_APDFT5=False):
    # Reorder two labels and relative energies so that the SCF relative energies are positive

    # Enantiomers
    if len(alch_isomers[0]) > 0:
        if not flag_APDFT5:
            for i in range(len(alch_isomers[0])):
                if rel_ene["enantiomers"][target_type][i] < 0:
                    # Swap indexes
                    alch_isomers[0][i][0], alch_isomers[0][i][1] = alch_isomers[0][i][1], alch_isomers[0][i][0]
                    rel_ene["enantiomers"]["APDFT1"][i] *= -1.0
                    rel_ene["enantiomers"]["APDFT3"][i] *= -1.0
                    if "SCF" in rel_ene["enantiomers"]:
                        rel_ene["enantiomers"]["SCF"][i] *= -1.0
        else:
            for i in range(len(alch_isomers[0])):
                if rel_ene["enantiomers"][target_type][i] < 0:
                    # Swap indexes
                    alch_isomers[0][i][0], alch_isomers[0][i][1] = alch_isomers[0][i][1], alch_isomers[0][i][0]
                    rel_ene["enantiomers"]["APDFT1"][i] *= -1.0
                    rel_ene["enantiomers"]["APDFT3"][i] *= -1.0
                    rel_ene["enantiomers"]["APDFT5"][i] *= -1.0
                    if "SCF" in rel_ene["enantiomers"]:
                        rel_ene["enantiomers"]["SCF"][i] *= -1.0

    # Diastereomers
    if len(alch_isomers[1]) > 0:
        if not flag_APDFT5:
            for i in range(len(alch_isomers[1])):
                if rel_ene["diastereomers"][target_type][i] < 0:
                    # Swap indexes
                    alch_isomers[1][i][0], alch_isomers[1][i][1] = alch_isomers[1][i][1], alch_isomers[1][i][0]
                    rel_ene["diastereomers"]["APDFT1"][i] *= -1.0
                    rel_ene["diastereomers"]["APDFT3"][i] *= -1.0
                    if "SCF" in rel_ene["diastereomers"]:
                        rel_ene["diastereomers"]["SCF"][i] *= -1.0
        else:
            for i in range(len(alch_isomers[1])):
                if rel_ene["diastereomers"][target_type][i] < 0:
                    # Swap indexes
                    alch_isomers[1][i][0], alch_isomers[1][i][1] = alch_isomers[1][i][1], alch_isomers[1][i][0]
                    rel_ene["diastereomers"]["APDFT1"][i] *= -1.0
                    rel_ene["diastereomers"]["APDFT3"][i] *= -1.0
                    rel_ene["diastereomers"]["APDFT5"][i] *= -1.0
                    if "SCF" in rel_ene["diastereomers"]:
                        rel_ene["diastereomers"]["SCF"][i] *= -1.0

    return alch_isomers, rel_ene

def scatter_plot(data_x, data_y, fig_name="scatter_plot.png", data_labels=None, given_range_x=None, given_range_y=None,
                 x_axis_name="DFT relative energy / Hartree", y_axis_name="APDFT relative energy / Hartree"):
    fig, ax = plt.subplots(figsize=(10, 8))

    plt.rcParams['font.family'] = 'Arial'
    if data_labels is None:
        plt.scatter(data_x, data_y, color='tab:blue', marker='o', s=550, zorder=2,
                    label=data_labels, clip_on=False)
    else:
        # plt.scatter(data_x, data_y, color=['tab:blue', 'tab:orange'], marker=['o', 'v'], s=550, zorder=2, label=data_labels)
        if len(data_x) == 2:
            plt.scatter(data_x[0], data_y[0], color='tab:blue', marker='o', s=550, zorder=2,
                        label=data_labels[0], clip_on=False)
            plt.scatter(data_x[1], data_y[1], color='tab:orange', marker='v', s=550, zorder=2,
                        label=data_labels[1], clip_on=False)
        elif len(data_x) == 3:
            plt.scatter(data_x[0], data_y[0], color='tab:blue', marker='o', s=550, zorder=2,
                        label=data_labels[0], clip_on=False)
            plt.scatter(data_x[1], data_y[1], color='tab:orange', marker='v', s=550, zorder=2,
                        label=data_labels[1], clip_on=False)
            plt.scatter(data_x[2], data_y[2], color='tab:green', marker='s', s=550, zorder=2,
                        label=data_labels[2], clip_on=False)

    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)

    min_value_x = int(np.amin(data_x)) - 1
    max_value_x = round(np.amax(data_x)) + 1
    min_value_y = int(np.amin(data_y)) - 1
    max_value_y = round(np.amax(data_y)) + 1

    def adjust_auto(num, sign):
        if num == 0:
            return 0
        else:
            if sign == "+":
                return round(num, -int(math.floor(math.log10(abs(num))))) + \
                    0.1 ** (-int(math.floor(math.log10(abs(num)))))
            elif sign == "-":
                return round(num, -int(math.floor(math.log10(abs(num))))) - \
                    0.1 ** (-int(math.floor(math.log10(abs(num)))))

    if np.amax(data_x) < 1.0 or np.amax(data_y) < 1.0:
        min_value_x = adjust_auto(np.amin(data_x), sign="-") * 10
        max_value_x = adjust_auto(np.amax(data_x), sign="+") * 10
        min_value_y = adjust_auto(np.amin(data_y), sign="-") * 10
        max_value_y = adjust_auto(np.amax(data_y), sign="+") * 10

    if given_range_x is not None:
        min_value_x = min(given_range_x)
        max_value_x = max(given_range_x)
    if given_range_y is not None:
        min_value_y = min(given_range_y)
        max_value_y = max(given_range_y)

    if max_value_x >= max_value_y:
        max_value_y = max_value_x
    else:
        max_value_x = max_value_y

    if min_value_x >= min_value_y:
        min_value_x = min_value_y
    else:
        min_value_y = min_value_x

    plt.xlim(min_value_x, max_value_x)
    plt.ylim(min_value_y, max_value_y)

    plt.xlabel(x_axis_name, fontsize=30)
    plt.ylabel(y_axis_name, fontsize=30)

    if data_labels is not None:
        ax.legend(fontsize=30)

    # diagonal
    plt.plot([min_value_x, max_value_x], [min_value_y, max_value_y],
             color='black', linewidth=3, zorder=1)

    plt.tick_params(axis='both', which='major', length=18, width=2, direction='in', pad=10, labelsize=26)
    if given_range_x is not None:
        ax.set_xticks(given_range_x)
    if given_range_y is not None:
        ax.set_yticks(given_range_y)

    for spine in plt.gca().spines.values():
        spine.set_linewidth(2)

    plt.tight_layout()
    plt.savefig(fig_name, dpi=600, bbox_inches='tight')

def set_dict_scatter_plot(replace):
    axis_range = {"scatter_plot_enantiomers_APDFT1":None,
                  "scatter_plot_enantiomers_APDFT3":None,
                  "scatter_plot_enantiomers_APDFT1and3":None,
                  "scatter_plot_diastereomers_APDFT1":None,
                  "scatter_plot_diastereomers_APDFT3":None,
                  "scatter_plot_diastereomers_APDFT1and3":None}

    if replace is not None:
        for key in axis_range.keys():
            if key in replace.keys():
                axis_range[key] = replace[key]

    return axis_range

def scatter_plot_relative_energies(alch_isomers=None, rel_ene=None,
                                   given_range_x=None, given_range_y=None,
                                   flag_total_energy=False, name_coord_file="./target_coord.xyz",
                                   file_path_enantiomers="rel_ene_enantiomers.dat",
                                   file_path_diastereomers="rel_ene_diastereomers.dat",
                                   flag_APDFT5=False):
    if alch_isomers is None or rel_ene is None:
        alch_isomers, rel_ene = read_relative_energies_from_outputs(flag_total_energy, name_coord_file,
                                                                    file_path_enantiomers,
                                                                    file_path_diastereomers)

    alch_isomers, rel_ene = make_relative_energies_positive(alch_isomers, rel_ene, flag_APDFT5=flag_APDFT5)

    given_range_x = set_dict_scatter_plot(given_range_x)
    given_range_y = set_dict_scatter_plot(given_range_y)

    def multiply_1000(value):
        if value is not None:
            return value * 1000
        else:
            return value

    given_range_x["scatter_plot_enantiomers_APDFT1"] = multiply_1000(given_range_x["scatter_plot_enantiomers_APDFT1"])
    given_range_y["scatter_plot_enantiomers_APDFT1"] = multiply_1000(given_range_y["scatter_plot_enantiomers_APDFT1"])
    given_range_x["scatter_plot_enantiomers_APDFT3"] = multiply_1000(given_range_x["scatter_plot_enantiomers_APDFT3"])
    given_range_y["scatter_plot_enantiomers_APDFT3"] = multiply_1000(given_range_y["scatter_plot_enantiomers_APDFT3"])
    given_range_x["scatter_plot_enantiomers_APDFT1and3"] = multiply_1000(given_range_x["scatter_plot_enantiomers_APDFT1and3"])
    given_range_y["scatter_plot_enantiomers_APDFT1and3"] = multiply_1000(given_range_y["scatter_plot_enantiomers_APDFT1and3"])

    if len(alch_isomers[0]) > 0:
        scatter_plot(rel_ene["enantiomers"]["SCF"] * 1000, rel_ene["enantiomers"]["APDFT1"] * 1000,
                     fig_name="scatter_plot_enantiomers_APDFT1.png",
                     given_range_x=given_range_x["scatter_plot_enantiomers_APDFT1"],
                     given_range_y=given_range_y["scatter_plot_enantiomers_APDFT1"],
                     x_axis_name="DFT relative energy\n/ milli Hartree",
                     y_axis_name="APDFT relative energy\n/ milli Hartree")
        scatter_plot(rel_ene["enantiomers"]["SCF"] * 1000, rel_ene["enantiomers"]["APDFT3"] * 1000,
                     fig_name="scatter_plot_enantiomers_APDFT3.png",
                     given_range_x=given_range_x["scatter_plot_enantiomers_APDFT3"],
                     given_range_y=given_range_y["scatter_plot_enantiomers_APDFT3"],
                     x_axis_name="DFT relative energy\n/ milli Hartree",
                     y_axis_name="APDFT relative energy\n/ milli Hartree")
        scatter_plot([rel_ene["enantiomers"]["SCF"] * 1000, rel_ene["enantiomers"]["SCF"] * 1000],
                     [rel_ene["enantiomers"]["APDFT1"] * 1000, rel_ene["enantiomers"]["APDFT3"] * 1000],
                     fig_name="scatter_plot_enantiomers_APDFT1and3.png",
                     data_labels=["First derivative", "First plus third derivatives"],
                     given_range_x=given_range_x["scatter_plot_enantiomers_APDFT1and3"],
                     given_range_y=given_range_y["scatter_plot_enantiomers_APDFT1and3"],
                     x_axis_name="DFT relative energy\n/ milli Hartree",
                     y_axis_name="APDFT relative energy\n/ milli Hartree")
        if flag_APDFT5:
            print([rel_ene["enantiomers"]["APDFT1"] * 1000, rel_ene["enantiomers"]["APDFT3"] * 1000, rel_ene["enantiomers"]["APDFT5"] * 1000])
            scatter_plot([rel_ene["enantiomers"]["SCF"] * 1000, rel_ene["enantiomers"]["SCF"] * 1000,
                          rel_ene["enantiomers"]["SCF"] * 1000],
                     [rel_ene["enantiomers"]["APDFT1"] * 1000, rel_ene["enantiomers"]["APDFT3"] * 1000,
                      rel_ene["enantiomers"]["APDFT5"] * 1000],
                     fig_name="scatter_plot_enantiomers_APDFT1and3and5.png",
                     data_labels=["First order", "Third order", "Fifth order"],
                     given_range_x=given_range_x["scatter_plot_enantiomers_APDFT1and3"],
                     given_range_y=given_range_y["scatter_plot_enantiomers_APDFT1and3"],
                     x_axis_name="DFT relative energy\n/ milli Hartree",
                     y_axis_name="APDFT relative energy\n/ milli Hartree")
    if len(alch_isomers[1]) > 0:
        scatter_plot(rel_ene["diastereomers"]["SCF"], rel_ene["diastereomers"]["APDFT1"],
                     fig_name="scatter_plot_diastereomers_APDFT1.png",
                     given_range_x=given_range_x["scatter_plot_diastereomers_APDFT1"],
                     given_range_y=given_range_y["scatter_plot_diastereomers_APDFT1"])
        scatter_plot(rel_ene["diastereomers"]["SCF"], rel_ene["diastereomers"]["APDFT3"],
                     fig_name="scatter_plot_diastereomers_APDFT3.png",
                     given_range_x=given_range_x["scatter_plot_diastereomers_APDFT3"],
                     given_range_y=given_range_y["scatter_plot_diastereomers_APDFT3"])
        scatter_plot([rel_ene["diastereomers"]["SCF"], rel_ene["diastereomers"]["SCF"]],
                     [rel_ene["diastereomers"]["APDFT1"], rel_ene["diastereomers"]["APDFT3"]],
                     fig_name="scatter_plot_diastereomers_APDFT1and3.png",
                     data_labels=["First derivative", "First plus third derivatives"],
                     given_range_x=given_range_x["scatter_plot_diastereomers_APDFT1and3"],
                     given_range_y=given_range_y["scatter_plot_diastereomers_APDFT1and3"])
        if flag_APDFT5:
            scatter_plot([rel_ene["diastereomers"]["SCF"], rel_ene["diastereomers"]["SCF"],
                          rel_ene["diastereomers"]["SCF"]],
                     [rel_ene["diastereomers"]["APDFT1"], rel_ene["diastereomers"]["APDFT3"],
                      rel_ene["diastereomers"]["APDFT5"]],
                     fig_name="scatter_plot_diastereomers_APDFT1and3and5.png",
                     data_labels=["First order", "Third order", "Fifth order"],
                     given_range_x=given_range_x["scatter_plot_diastereomers_APDFT1and3"],
                     given_range_y=given_range_y["scatter_plot_diastereomers_APDFT1and3"])

def count_mutations_from_PAHs_in_target_molecules_inp(filename="target_molecules.inp"):
    # Warning: This function can be only applied to BN-doe PAHs
    num_mutations = []
    with open(filename, 'r') as file:
        for line in file:
            # The number of BN pairs
            count_borons = line.count('5')
            num_mutations.append(count_borons)

    return num_mutations

def classify_num_mutations_alchemical_isomers(num_mutations=None, alch_isomers=None):
    if num_mutations is None:
        # Input: target_molecules.inp
        num_mutations = count_mutations_from_PAHs_in_target_molecules_inp()

    if alch_isomers is None:
        # Input: rel_ene_enantiomers.dat, rel_ene_diastereomers.dat
        alch_isomers, _ = read_relative_energies_from_outputs()

    num_mutations_in_alch_isomers = []
    for alch_enantiomers_or_diastereomers in alch_isomers:
        num_mutations_in_alch_isomer = []
        for alch_isomer in alch_enantiomers_or_diastereomers:
            num_mutations_in_alch_isomer.append(num_mutations[alch_isomer[0] - 1])
        num_mutations_in_alch_isomers.append(num_mutations_in_alch_isomer)

    return num_mutations_in_alch_isomers

def plot_values(data_x, data_y, fig_name="plot_values.png", data_labels=None, given_range_x=None, given_range_y=None,
                num_mutations_in_alch_isomer=None, name_xlabel="Alchemical isomers",
                name_ylabel="APDFT relative energy / Hartree"):
    fig, ax = plt.subplots(figsize=(12, 8))

    if num_mutations_in_alch_isomer is not None:
        # Create a colormap
        colormap = cm.get_cmap('tab20')

        # Iterate over the list and create colored regions
        for i, num in enumerate(num_mutations_in_alch_isomer):
            plt.axvspan(i + 1 - 0.5, i + 1 + 0.5, facecolor=colormap(num - 1), alpha=0.4)

    plt.rcParams['font.family'] = 'Arial'
    if data_labels is None:
        plt.plot(data_x, data_y, color='tab:blue', marker='o', label=data_labels, clip_on=False,
                 markersize=14, linewidth=3)
    else:
        colormap = cm.get_cmap('tab10')
        markers = ['o', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']
        for i in range(len(data_labels)):
            plt.plot(data_x[i], data_y[i], color=colormap(i), marker=markers[i%len(markers)],
                     label=data_labels[i], clip_on=False, markersize=14, linewidth=3)

    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)

    min_value_x = int(np.amin(data_x)) - 1
    max_value_x = round(np.amax(data_x)) + 1
    min_value_y = int(np.amin(data_y)) - 1
    max_value_y = round(np.amax(data_y)) + 1

    def adjust_auto(num, sign):
        if num == 0:
            return 0
        else:
            if sign == "+":
                return round(num, -int(math.floor(math.log10(abs(num))))) + \
                    0.1 ** (-int(math.floor(math.log10(abs(num)))))
            elif sign == "-":
                return round(num, -int(math.floor(math.log10(abs(num))))) - \
                    0.1 ** (-int(math.floor(math.log10(abs(num)))))

    if np.amax(data_y) < 1.0:
        min_value_y = adjust_auto(np.amin(data_y), sign="-") * 10
        max_value_y = adjust_auto(np.amax(data_y), sign="+") * 10

    if given_range_x is not None:
        min_value_x = min(given_range_x)
        max_value_x = max(given_range_x)
    if given_range_y is not None:
        min_value_y = min(given_range_y)
        max_value_y = max(given_range_y)

    plt.xlim(min_value_x, max_value_x)
    plt.ylim(min_value_y, max_value_y)

    plt.xlabel(name_xlabel, fontsize=30)
    plt.ylabel(name_ylabel, fontsize=30)

    if data_labels is not None:
        ax.legend(fontsize=30)

    plt.tick_params(axis='both', which='major', length=18, width=2, direction='in', pad=10, labelsize=26)
    if given_range_x is not None:
        ax.set_xticks(given_range_x)
    if given_range_y is not None:
        ax.set_yticks(given_range_y)

    for spine in plt.gca().spines.values():
        spine.set_linewidth(2)

    plt.tight_layout()
    plt.savefig(fig_name, dpi=600, bbox_inches='tight')

def set_dict_plot_values(replace):
    axis_range = {"plot_enantiomers_APDFT1":None,
                  "plot_enantiomers_APDFT3":None,
                  "plot_enantiomers_APDFT1and3":None,
                  "plot_enantiomers_APDFT1-3":None,
                  "plot_enantiomers_APDFT1and3andSCF":None,
                  "plot_enantiomers_APDFT1and3-SCF":None,
                  "plot_diastereomers_APDFT1":None,
                  "plot_diastereomers_APDFT3":None,
                  "plot_diastereomers_APDFT1and3":None,
                  "plot_diastereomers_APDFT1-3":None,
                  "plot_diastereomers_APDFT1and3andSCF":None,
                  "plot_diastereomers_APDFT1and3-SCF":None}

    if replace is not None:
        for key in axis_range.keys():
            if key in replace.keys():
                axis_range[key] = replace[key]

    return axis_range

def sort_within_groups(num_mutations_in_alch_isomers, *args):
    # Pair each element with its index
    paired = list(enumerate(args[0]))

    # Sort the pairs based on the corresponding value in num_mutations_in_alch_isomers
    paired.sort(key=lambda x: (num_mutations_in_alch_isomers[x[0]], x[1]))

    # Extract the sorted indices
    sorted_indices = [i for i, _ in paired]

    # Sort all lists using the sorted indices
    sorted_arrays = [[lst[i] for i in sorted_indices] for lst in args]
    sorted_arrays = [np.array(lst) for lst in sorted_arrays]

    return sorted_arrays

def plot_relative_energies(alch_isomers=None, rel_ene=None,
                           given_range_x=None, given_range_y=None,
                           flag_total_energy=False, name_coord_file="./target_coord.xyz",
                           file_path_enantiomers="rel_ene_enantiomers.dat",
                           file_path_diastereomers="rel_ene_diastereomers.dat"):
    if alch_isomers is None or rel_ene is None:
        alch_isomers, rel_ene = read_relative_energies_from_outputs(flag_total_energy, name_coord_file,
                                                                    file_path_enantiomers=file_path_enantiomers,
                                                                    file_path_diastereomers=file_path_diastereomers)

    given_range_x = set_dict_plot_values(given_range_x)
    given_range_y = set_dict_plot_values(given_range_y)

    if "enantiomers" in rel_ene:
        if "SCF" in rel_ene["enantiomers"]:
            target_type = "SCF"
        else:
            target_type = "APDFT3"
    elif "diastereomers" in rel_ene:
        if "SCF" in rel_ene["diastereomers"]:
            target_type = "SCF"
        else:
            target_type = "APDFT3"
    else:
        raise ValueError("No relative energies are found.")

    alch_isomers, rel_ene = make_relative_energies_positive(alch_isomers, rel_ene, target_type=target_type)

    num_mutations_in_alch_isomers = classify_num_mutations_alchemical_isomers()

    if "enantiomers" in rel_ene:
        if "SCF" in rel_ene["enantiomers"]:
            rel_ene["enantiomers"]["SCF"], rel_ene["enantiomers"]["APDFT1"], rel_ene["enantiomers"]["APDFT3"] = \
                sort_within_groups(num_mutations_in_alch_isomers[0],
                                   rel_ene["enantiomers"]["SCF"],
                                   rel_ene["enantiomers"]["APDFT1"],
                                   rel_ene["enantiomers"]["APDFT3"])
            rel_ene["enantiomers"]["APDFT3-SCF"], rel_ene["enantiomers"]["APDFT1-SCF"] = \
                sort_within_groups(num_mutations_in_alch_isomers[0],
                                   np.abs(rel_ene["enantiomers"]["APDFT3-SCF"]),
                                   np.abs(rel_ene["enantiomers"]["APDFT1-SCF"]))
        else:
            rel_ene["enantiomers"]["APDFT3"], rel_ene["enantiomers"]["APDFT1"] = \
                sort_within_groups(num_mutations_in_alch_isomers[0],
                                   rel_ene["enantiomers"]["APDFT3"],
                                   rel_ene["enantiomers"]["APDFT1"])

    # Count the number of estimations with errors of the sign
    # error_count = 0
    # print(len(rel_ene["enantiomers"]["APDFT3"]))
    # for i, this_energy in enumerate(rel_ene["enantiomers"]["APDFT3"]):
    #     if this_energy < 0:
    #         print(i + 1, this_energy)
    #         error_count += 1
    # print(error_count)

    if "diastereomers" in rel_ene:
        if "SCF" in rel_ene["diastereomers"]:
            rel_ene["diastereomers"]["SCF"], rel_ene["diastereomers"]["APDFT1"], rel_ene["diastereomers"]["APDFT3"] = \
                sort_within_groups(num_mutations_in_alch_isomers[1],
                                   rel_ene["diastereomers"]["SCF"],
                                   rel_ene["diastereomers"]["APDFT1"],
                                   rel_ene["diastereomers"]["APDFT3"])
            rel_ene["diastereomers"]["APDFT3-SCF"], rel_ene["diastereomers"]["APDFT1-SCF"] = \
                sort_within_groups(num_mutations_in_alch_isomers[1],
                                   np.abs(rel_ene["diastereomers"]["APDFT3-SCF"]),
                                   np.abs(rel_ene["diastereomers"]["APDFT1-SCF"]))
        else:
            rel_ene["diastereomers"]["APDFT3"], rel_ene["diastereomers"]["APDFT1"] = \
                sort_within_groups(num_mutations_in_alch_isomers[1],
                                   rel_ene["diastereomers"]["APDFT3"],
                                   rel_ene["diastereomers"]["APDFT1"])

    def multiply_1000(value):
        if value is not None:
            return value * 1000
        else:
            return value

    given_range_y["plot_enantiomers_APDFT1"] = multiply_1000(given_range_y["plot_enantiomers_APDFT1"])
    given_range_y["plot_enantiomers_APDFT3"] = multiply_1000(given_range_y["plot_enantiomers_APDFT3"])
    given_range_y["plot_enantiomers_APDFT1and3"] = multiply_1000(given_range_y["plot_enantiomers_APDFT1and3"])
    given_range_y["plot_enantiomers_APDFT1-3"] = multiply_1000(given_range_y["plot_enantiomers_APDFT1-3"])
    given_range_y["plot_enantiomers_APDFT1and3andSCF"] = multiply_1000(given_range_y["plot_enantiomers_APDFT1and3andSCF"])
    given_range_y["plot_enantiomers_APDFT1and3-SCF"] = multiply_1000(given_range_y["plot_enantiomers_APDFT1and3-SCF"])
    given_range_y["plot_diastereomers_APDFT1and3-SCF"] = multiply_1000(given_range_y["plot_diastereomers_APDFT1and3-SCF"])
    given_range_y["plot_diastereomers_APDFT1-3"] = multiply_1000(given_range_y["plot_diastereomers_APDFT1-3"])

    if len(alch_isomers[0]) > 0:
        x_axis_indexes = range(1, len(rel_ene["enantiomers"]["APDFT1"]) + 1)
        plot_values(x_axis_indexes, rel_ene["enantiomers"]["APDFT1"] * 1000,
                    fig_name="plot_enantiomers_APDFT1.png",
                    given_range_x=given_range_x["plot_enantiomers_APDFT1"],
                    given_range_y=given_range_y["plot_enantiomers_APDFT1"],
                    num_mutations_in_alch_isomer=num_mutations_in_alch_isomers[0],
                    name_xlabel="Alchemical enantiomer",
                    name_ylabel="Relative energy / milli Hartree")
        plot_values(x_axis_indexes, rel_ene["enantiomers"]["APDFT3"] * 1000,
                    fig_name="plot_enantiomers_APDFT3.png",
                    given_range_x=given_range_x["plot_enantiomers_APDFT3"],
                    given_range_y=given_range_y["plot_enantiomers_APDFT3"],
                    num_mutations_in_alch_isomer=num_mutations_in_alch_isomers[0],
                    name_xlabel="Alchemical enantiomer",
                    name_ylabel="Relative energy / milli Hartree")
        plot_values([x_axis_indexes, x_axis_indexes],
                    [rel_ene["enantiomers"]["APDFT1"] * 1000,
                     rel_ene["enantiomers"]["APDFT3"] * 1000],
                    fig_name="plot_enantiomers_APDFT1and3.png",
                    data_labels=["First derivative", "First plus third derivatives"],
                    given_range_x=given_range_x["plot_enantiomers_APDFT1and3"],
                    given_range_y=given_range_y["plot_enantiomers_APDFT1and3"],
                    num_mutations_in_alch_isomer=num_mutations_in_alch_isomers[0],
                    name_xlabel="Alchemical enantiomer",
                    name_ylabel="Relative energy / milli Hartree")
        plot_values(x_axis_indexes,
                    sort_within_groups(num_mutations_in_alch_isomers[0],
                                       np.abs(rel_ene["enantiomers"]["APDFT3"] - rel_ene["enantiomers"]["APDFT1"]))[0] * 1000,
                    fig_name="plot_enantiomers_APDFT1-3.png",
                    given_range_x=given_range_x["plot_enantiomers_APDFT1-3"],
                    given_range_y=given_range_y["plot_enantiomers_APDFT1-3"],
                    num_mutations_in_alch_isomer=num_mutations_in_alch_isomers[0],
                    name_xlabel="Alchemical enantiomer",
                    name_ylabel="Difference in APDFT\nrelative energies / milli Hartree")
        if "SCF" in rel_ene["enantiomers"]:
            plot_values([x_axis_indexes, x_axis_indexes, x_axis_indexes],
                        [rel_ene["enantiomers"]["APDFT1"] * 1000,
                         rel_ene["enantiomers"]["APDFT3"] * 1000,
                         rel_ene["enantiomers"]["SCF"] * 1000],
                        fig_name="plot_enantiomers_APDFT1and3andSCF.png",
                        data_labels=["First derivative", "First plus third derivatives", "DFT"],
                        given_range_x=given_range_x["plot_enantiomers_APDFT1and3andSCF"],
                        given_range_y=given_range_y["plot_enantiomers_APDFT1and3andSCF"],
                        num_mutations_in_alch_isomer=num_mutations_in_alch_isomers[0],
                        name_xlabel="Alchemical enantiomer",
                        name_ylabel="Relative energy / milli Hartree")
            plot_values([x_axis_indexes, x_axis_indexes],
                        [np.abs(rel_ene["enantiomers"]["APDFT1-SCF"]),
                         np.abs(rel_ene["enantiomers"]["APDFT3-SCF"])],
                        fig_name="plot_enantiomers_APDFT1and3-SCF.png",
                        data_labels=["First derivative", "First plus third derivatives"],
                        given_range_x=given_range_x["plot_enantiomers_APDFT1and3-SCF"],
                        given_range_y=given_range_y["plot_enantiomers_APDFT1and3-SCF"],
                        num_mutations_in_alch_isomer=num_mutations_in_alch_isomers[0],
                        name_xlabel="Alchemical enantiomer",
                        name_ylabel="Error in APDFT relative energy\n/ milli Hartree")

    if len(alch_isomers[1]) > 0:
        x_axis_indexes = range(1, len(rel_ene["diastereomers"]["APDFT1"]) + 1)
        plot_values(x_axis_indexes, rel_ene["diastereomers"]["APDFT1"],
                    fig_name="plot_diastereomers_APDFT1.png",
                    given_range_x=given_range_x["plot_diastereomers_APDFT1"],
                    given_range_y=given_range_y["plot_diastereomers_APDFT1"],
                    num_mutations_in_alch_isomer=num_mutations_in_alch_isomers[1],
                    name_xlabel="Alchemical diastereomer")
        plot_values(x_axis_indexes, rel_ene["diastereomers"]["APDFT3"],
                    fig_name="plot_diastereomers_APDFT3.png",
                    given_range_x=given_range_x["plot_diastereomers_APDFT3"],
                    given_range_y=given_range_y["plot_diastereomers_APDFT3"],
                    num_mutations_in_alch_isomer=num_mutations_in_alch_isomers[1],
                    name_xlabel="Alchemical diastereomer")
        plot_values([x_axis_indexes, x_axis_indexes],
                     [rel_ene["diastereomers"]["APDFT1"], rel_ene["diastereomers"]["APDFT3"]],
                     fig_name="plot_diastereomers_APDFT1and3.png",
                     data_labels=["First derivative", "First plus third derivatives"],
                     given_range_x=given_range_x["plot_diastereomers_APDFT1and3"],
                     given_range_y=given_range_y["plot_diastereomers_APDFT1and3"],
                     num_mutations_in_alch_isomer=num_mutations_in_alch_isomers[1],
                     name_xlabel="Alchemical diastereomer")
        plot_values(x_axis_indexes,
                    sort_within_groups(num_mutations_in_alch_isomers[1],
                                       np.abs(rel_ene["diastereomers"]["APDFT3"] - rel_ene["diastereomers"]["APDFT1"]))[0] * 1000,
                    fig_name="plot_diastereomers_APDFT1-3.png",
                    given_range_x=given_range_x["plot_diastereomers_APDFT1-3"],
                    given_range_y=given_range_y["plot_diastereomers_APDFT1-3"],
                    num_mutations_in_alch_isomer=num_mutations_in_alch_isomers[1],
                    name_xlabel="Alchemical diastereomer",
                    name_ylabel="Difference in APDFT\nrelative energies / milli Hartree")
        if "SCF" in rel_ene["diastereomers"]:
            plot_values([x_axis_indexes, x_axis_indexes, x_axis_indexes],
                        [rel_ene["diastereomers"]["APDFT1"],
                         rel_ene["diastereomers"]["APDFT3"],
                         rel_ene["diastereomers"]["SCF"]],
                        fig_name="plot_diastereomers_APDFT1and3andSCF.png",
                        data_labels=["First derivative", "First plus third derivatives", "DFT"],
                        given_range_x=given_range_x["plot_diastereomers_APDFT1and3andSCF"],
                        given_range_y=given_range_y["plot_diastereomers_APDFT1and3andSCF"],
                        num_mutations_in_alch_isomer=num_mutations_in_alch_isomers[1],
                        name_xlabel="Alchemical diastereomer",
                        name_ylabel="Relative energy / Hartree")
            plot_values([x_axis_indexes, x_axis_indexes],
                        [np.abs(rel_ene["diastereomers"]["APDFT1-SCF"]),
                         np.abs(rel_ene["diastereomers"]["APDFT3-SCF"])],
                        fig_name="plot_diastereomers_APDFT1and3-SCF.png",
                        data_labels=["First derivative", "First plus third derivatives"],
                        given_range_x=given_range_x["plot_diastereomers_APDFT1and3-SCF"],
                        given_range_y=given_range_y["plot_diastereomers_APDFT1and3-SCF"],
                        num_mutations_in_alch_isomer=num_mutations_in_alch_isomers[1],
                        name_xlabel="Alchemical diastereomer",
                        name_ylabel="Error in APDFT relative energy\n/ milli Hartree")
