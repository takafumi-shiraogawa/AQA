import numpy as np
import basis_set_exchange as bse
from pyscf import gto, scf, lib
import pandas as pd
import xarray as xr
import csv
from .mini_qml import representations as amr
from .alch_calc import alchemical_calculator as ac
import multiprocessing as mp
from .utils import read_relative_energies_from_outputs
from .mod_pyscf_scf import RHF, RKS

def judge_alch_isomers_types(alch_isomers):
    try:
        if not alch_isomers[0]:
            flag_enantiomers = False
        else:
            flag_enantiomers = True
    except:
        flag_enantiomers = True

    try:
        if not alch_isomers[1]:
            flag_diastereomers = False
        else:
            flag_diastereomers = True
    except:
        flag_diastereomers = True

    return flag_enantiomers, flag_diastereomers

# Assuming that APDFT relative energies are already calculated for all the molecules
def get_randomly_selected_alch_isomers(num_selected_pairs_enantiomers=100,
                                       num_selected_pairs_diastereomers=100,
                                       random_seed=37):
    alch_isomers, rel_ene = read_relative_energies_from_outputs()

    flag_enantiomers, flag_diastereomers = judge_alch_isomers_types(alch_isomers)

    np.random.seed(random_seed)
    if flag_enantiomers:
        if num_selected_pairs_enantiomers >= len(alch_isomers[0]):
            raise ValueError("num_selected_pairs_enantiomers must be smaller than the number of enantiomers.")

        selected_indices_enantiomers = np.random.choice(len(alch_isomers[0]),
                                                        num_selected_pairs_enantiomers,
                                                        replace=False)
    if flag_diastereomers:
        if num_selected_pairs_diastereomers >= len(alch_isomers[1]):
            raise ValueError("num_selected_pairs_diastereomers must be smaller than the number of diastereomers.")

        selected_indices_diastereomers = np.random.choice(len(alch_isomers[1]),
                                                          num_selected_pairs_diastereomers,
                                                          replace=False)

    # get randomly selected alch_isomers
    selected_alch_isomers = [[], []]
    if flag_enantiomers:
        for idx in selected_indices_enantiomers:
            # -1 accounts for obtained indexes of molecules in alch_isomers start from 1, not 0.
            selected_alch_isomers[0].append(alch_isomers[0][idx] - 1)

    if flag_diastereomers:
        for idx in selected_indices_diastereomers:
            # -1 accounts for obtained indexes of molecules in alch_isomers start from 1, not 0.
            selected_alch_isomers[1].append(alch_isomers[1][idx] - 1)

    return selected_alch_isomers

def get_mols_in_alch_isomers(alch_isomers):
    flag_enantiomers, flag_diastereomers = judge_alch_isomers_types(alch_isomers)

    mol_indexes_in_alch_isomers = []

    if flag_enantiomers:
        for pair in alch_isomers[0]:
            mol_indexes_in_alch_isomers.append(pair[0])
            mol_indexes_in_alch_isomers.append(pair[1])

    if flag_diastereomers:
        for pair in alch_isomers[1]:
            mol_indexes_in_alch_isomers.append(pair[0])
            mol_indexes_in_alch_isomers.append(pair[1])

    mol_indexes_in_alch_isomers = sorted(list(set(mol_indexes_in_alch_isomers)))

    return mol_indexes_in_alch_isomers

def read_randomly_selected_scf_ele_enes_from_output():
    # Get indexes of randomly selected molecules in target_molecules.inp
    with open("randomly_selected_target_molecules_indexes.dat", 'r') as fh:
        data_string = fh.read()
        lines = data_string.split('\n')
        indexes = []
        for line in lines:
            if line.strip():
                indexes.append(int(line))
    # Get APDFT electronic energies
    randomly_selected_scf_ele_enes = read_scf_ele_energies('randomly_selected_scf_ele_ene.dat')

    randomly_selected_scf_ele_enes_dict = {}
    for i, idx in enumerate(indexes):
        randomly_selected_scf_ele_enes_dict[idx] = randomly_selected_scf_ele_enes[i]

    return randomly_selected_scf_ele_enes_dict

# --------------------------- Calculation ---------------------------#

def get_multi_ele_energies(ac_class, nuc_charges_vecs=None, flag_apdft5=False):
    """ Calculate multiple APDFT electronic energies from a list of nuclear charge vectors.

    Args:
        ac_class: An instance of alchemical_calculator class.
        nuc_charges_vecs: A list of nuclear charge vectors.

    Returns:
        A numpy array of shape (nuc_charges_vecs.shape[0], 3) containing the electronic energies for APDFT1, APDFT2, and APDFT3.
    """
    if nuc_charges_vecs is None:
        try:
            # Read nuc_charges_vecs from nuc_charges_list file "target_molecules.inp"
            nuc_charge_list = read_nuc_charges_list()
            nuc_charges_vecs = calc_nuc_charge_vecs(nuc_charge_list)
        except:
            raise ValueError("nuc_charges_vecs must be provided if no nuc_charges_list file is found.")

    if not flag_apdft5:
        ele_enes_apdft = np.zeros((nuc_charges_vecs.shape[0], 3))
    else:
        ele_enes_apdft = np.zeros((nuc_charges_vecs.shape[0], 5))

    for i, nuc_charges_vec in enumerate(nuc_charges_vecs):
        ele_enes_apdft[i, 0] = ac_class.get_elec_APDFT1(nuc_charges_vec[ac_class.ap.sites])
        ele_enes_apdft[i, 1] = ac_class.get_elec_APDFT2(nuc_charges_vec[ac_class.ap.sites])
        ele_enes_apdft[i, 2] = ac_class.get_elec_APDFT3(nuc_charges_vec[ac_class.ap.sites])
        if flag_apdft5:
            ele_enes_apdft[i, 3] = ac_class.get_elec_APDFT4(nuc_charges_vec[ac_class.ap.sites])
            ele_enes_apdft[i, 4] = ac_class.get_elec_APDFT5(nuc_charges_vec[ac_class.ap.sites])

    return ele_enes_apdft

def get_multi_response_properties(ac_class, nuc_charges_vecs=None):
    """ Calculate multiple APDFT response properties from a list of nuclear charge vectors.
        Response properties: electric dipole, polarizabilities

    Args:
        ac_class: An instance of alchemical_calculator class.
        nuc_charges_vecs: A list of nuclear charge vectors.

    Returns:
        A numpy array of shape (nuc_charges_vecs.shape[0], 1, 3) containing the electric dipole moments for APDFT1.
        A numpy array of shape (nuc_charges_vecs.shape[0], 1, 3, 3) containing the electric polarizabilities for APDFT1.
    """
    if not ac_class.flag_response_property:
        raise ValueError("flag_response_property should be True to calculate response properties")

    if nuc_charges_vecs is None:
        try:
            # Read nuc_charges_vecs from nuc_charges_list file "target_molecules.inp"
            nuc_charge_list = read_nuc_charges_list()
            nuc_charges_vecs = calc_nuc_charge_vecs(nuc_charge_list)
        except:
            raise ValueError("nuc_charges_vecs must be provided if no nuc_charges_list file is found.")

    # Electric dipole: (num_mol, APDFT's n (1), Cartesian coordinates)
    electric_dipoles_apdft = np.zeros((nuc_charges_vecs.shape[0], 1, 3))
    # Electric polarizabilities: (num_mol, APDFT's n (1), Cartesian coordinates, Cartesian coordinates)
    electric_pols_apdft = np.zeros((nuc_charges_vecs.shape[0], 1, 3, 3))

    for i, nuc_charges_vec in enumerate(nuc_charges_vecs):
        electric_dipoles_apdft[i, 0] = ac_class.get_electric_dipole_APDFT1(
            nuc_charges_vec[ac_class.ap.sites])
        electric_pols_apdft[i, 0] = ac_class.get_elec_electric_polarizability_APDFT1(
            nuc_charges_vec[ac_class.ap.sites])

    return electric_dipoles_apdft, electric_pols_apdft

# It is different from write_unique_nuclear_numbers_list in gener_chem_space.py
def write_nuclear_numbers_list(nuclear_numbers_list, filename="target_molecules.inp"):
    with open(filename, "w") as f:
        for nuclear_numbers in nuclear_numbers_list:
            f.write(','.join(map(str, list(nuclear_numbers))) + "\n")

def get_selected_unique_nuclear_numbers_list_from_mol_indexes(mol_indexes, nuc_charges_list=None):
    """ mol_indexes should correspond to the indexes of nuclear charge vectors in nuc_charges_list. """

    if nuc_charges_list is None:
        try:
            # Read nuc_charges_list file "target_molecules.inp"
            nuc_charges_list = read_nuc_charges_list()
        except:
            raise ValueError("nuc_charges_list must be provided if no nuc_charges_list file is found.")

    selected_indices = mol_indexes
    selected_indices = sorted(selected_indices)
    selected_nuc_charges_list = nuc_charges_list[selected_indices]
    # selected_nuc_charges_vecs = nuc_charges_vecs[selected_indices]

    # Save the selected nuclear charge vectors
    write_nuclear_numbers_list(selected_nuc_charges_list,
                               filename="randomly_selected_target_molecules.inp")

    # Save indexes of the selected nuclear charge vectors
    np.savetxt("randomly_selected_target_molecules_indexes.dat", selected_indices, fmt="%d")

    selected_nuc_charges_vecs = calc_nuc_charge_vecs(selected_nuc_charges_list, ref_nuc_charges=nuc_charges_list[0])

    return selected_nuc_charges_vecs

def get_randomly_selected_unique_nuclear_numbers_list(nuc_charges_list=None,
                                                      num_selected_mols=100,
                                                      random_seed=542):
    if nuc_charges_list is None:
        try:
            # Read nuc_charges_list file "target_molecules.inp"
            nuc_charges_list = read_nuc_charges_list()
            nuc_charges_vecs = calc_nuc_charge_vecs(nuc_charges_list)
        except:
            raise ValueError("nuc_charges_list must be provided if no nuc_charges_list file is found.")

    if num_selected_mols >= len(nuc_charges_list):
        raise ValueError("num_selected_mols must be larger than the number of molecules.")

    # Select a random subset of molecules
    np.random.seed(random_seed)
    selected_indices = np.random.choice(nuc_charges_list.shape[0],
                                        num_selected_mols, replace=False)
    selected_indices = sorted(selected_indices)
    selected_nuc_charges_list = nuc_charges_list[selected_indices]
    selected_nuc_charges_vecs = nuc_charges_vecs[selected_indices]

    # Save the selected nuclear charge vectors
    write_nuclear_numbers_list(selected_nuc_charges_list,
                               filename="randomly_selected_target_molecules.inp")

    # Save indexes of the selected nuclear charge vectors
    np.savetxt("randomly_selected_target_molecules_indexes.dat", selected_indices, fmt="%d")

    return selected_nuc_charges_vecs

def get_multi_ele_energies_randomly_selected_mols(ac_class, num_selected_mols=100,
                                                  selected_nuc_charges_vecs=None,
                                                  random_seed=542):
    """ Calculate multiple APDFT electronic energies from a list of nuclear charge vectors.

    Args:
        ac_class: An instance of alchemical_calculator class.
        selected_nuc_charges_vecs: A list of nuclear charge vectors.

    Returns:
        A numpy array of shape (nuc_charges_vecs.shape[0], 3) containing the electronic energies for APDFT1, APDFT2, and APDFT3.
    """
    if selected_nuc_charges_vecs is None:
        selected_nuc_charges_vecs = get_randomly_selected_unique_nuclear_numbers_list(nuc_charges_list=None,
                                                                                      num_selected_mols=num_selected_mols,
                                                                                      random_seed=random_seed)

    # Calculate APDFT electronic energies
    ele_enes_apdft = np.zeros((selected_nuc_charges_vecs.shape[0], 3))
    for i, nuc_charges_vec in enumerate(selected_nuc_charges_vecs):
        ele_enes_apdft[i, 0] = ac_class.get_elec_APDFT1(nuc_charges_vec[ac_class.ap.sites])
        ele_enes_apdft[i, 1] = ac_class.get_elec_APDFT2(nuc_charges_vec[ac_class.ap.sites])
        ele_enes_apdft[i, 2] = ac_class.get_elec_APDFT3(nuc_charges_vec[ac_class.ap.sites])

    return ele_enes_apdft

def get_multi_ele_energies_randomly_selected_mols_from_output(file_path=None):
    if file_path is None:
        file_path = 'randomly_selected_target_molecules_indexes.dat'

    with open(file_path, 'r') as file:
        indexes = [int(line.strip()) for line in file.readlines()]

    # Read APDFT electronic energies
    nuclear_charges_list, apdft_energies = read_apdft_energies()

    selected_apdft_energies = np.zeros((len(indexes), 3))
    for i, idx in enumerate(indexes):
        selected_apdft_energies[i, :] = apdft_energies[:, idx]

    return selected_apdft_energies

def calc_scf_ele_energy(data_string, dft_functional="PBE0", name_basis_set="def2-TZVP", num_cpus=None,
                        scf_max_cycle=None, mod_SCF=True, flag_norm_ddm_criterion=False,
                        norm_ddm_criterion=1.e-6, grid_level=3):
    """ Calculate the electronic energy of a molecule using a given DFT functional and basis set.
    """
    if num_cpus is not None:
        lib.num_threads(num_cpus)

    mol = gto.M(atom=data_string, basis=name_basis_set)
    if dft_functional is not None:
        if mod_SCF:
            mf = RKS(mol)
        else:
            mf = scf.RKS(mol)
        mf.xc = dft_functional
        mf.grids.level = grid_level
    else:
        if mod_SCF:
            mf = RHF(mol)
        else:
            mf = scf.RHF(mol)

    if scf_max_cycle is not None:
        mf.max_cycle = scf_max_cycle

    # mf.scf()
    if mod_SCF:
        print("mod_SCF is True.")
        mf.scf(flag_norm_ddm_criterion=flag_norm_ddm_criterion,
               norm_ddm_criterion=norm_ddm_criterion)
    else:
        mf.scf()

    if mf.converged:
        ele_ene = mf.e_tot - mf.energy_nuc()
        del mf
        del mol

        return ele_ene

    else:
        return None

def calc_scf_ele_energies_from_nuc_charge_vecs(data_string, sites, nuc_charges_vecs=None,
                                               dft_functional="PBE0", name_basis_set="def2-TZVP",
                                               bse_off=False, parallel=False, num_parallel=None,
                                               num_cpus=1, scf_max_cycle=None, flag_norm_ddm_criterion=False,
                                               norm_ddm_criterion=1.e-6, grid_level=3):
    """ Calculate the electronic energies of a molecule using a given DFT functional and basis set.
    """
    if nuc_charges_vecs is None:
        try:
            # Read nuc_charges_vecs from nuc_charges_list file "target_molecules.inp"
            nuc_charge_list = read_nuc_charges_list()
            nuc_charges_vecs = calc_nuc_charge_vecs(nuc_charge_list)
        except:
            raise ValueError("nuc_charges_vecs must be provided if no nuc_charges_list file is found.")

    if not bse_off:
        basis_set = bse.get_basis(name_basis_set, fmt="nwchem")
    else:
        basis_set = name_basis_set

    if not parallel:
        ele_enes = np.zeros(len(nuc_charges_vecs))

        for i, nuc_charges_vec in enumerate(nuc_charges_vecs):
            new_data_string = get_data_string_from_nuc_charge_vecs(data_string, nuc_charges_vec, sites)
            ele_enes[i] = calc_scf_ele_energy(new_data_string, dft_functional=dft_functional, name_basis_set=basis_set,
                                              scf_max_cycle=scf_max_cycle, grid_level=grid_level)

    else:
        if num_parallel is None:
            num_parallel = mp.cpu_count()

        # Create a process pool with the number of available CPUs
        pool = mp.Pool(num_parallel)

        # Create a list of parameters for calc_scf_ele_energy
        params = [(get_data_string_from_nuc_charge_vecs(data_string, nuc_charges_vec, sites), dft_functional, basis_set,
                   num_cpus, scf_max_cycle, flag_norm_ddm_criterion, norm_ddm_criterion)
                  for nuc_charges_vec in nuc_charges_vecs]

        # Use the pool's map function to call calc_scf_ele_energy in parallel
        ele_enes = pool.starmap(calc_scf_ele_energy, params)
        ele_enes = np.array(ele_enes)

    return ele_enes

def read_apdft_energies(file_path="./apdft_energies.csv", flag_apdft5=False):
    with open(file_path, "r") as csvfile:
        csvreader = csv.reader(csvfile)
        header = next(csvreader)

        apdft_energies = []
        nuclear_charges_list = []

        for row in csvreader:
            nuclear_charges = row[0]
            apdft1 = float(row[1])
            apdft2 = float(row[2])
            apdft3 = float(row[3])
            if flag_apdft5:
                apdft4 = float(row[4])
                apdft5 = float(row[5])
                apdft_energies.append([apdft1, apdft2, apdft3, apdft4, apdft5])
            else:
                apdft_energies.append([apdft1, apdft2, apdft3])
            nuclear_charges_list.append(nuclear_charges)

    apdft_energies = np.array(apdft_energies).transpose()

    # apdft_energies: (APDFT's n, num_mols)
    return nuclear_charges_list, apdft_energies

def read_apdft_response_properties(file_paths=["./apdft_electric_dipoles.csv", "./apdft_electric_pols.csv"]):
    # Read electric dipole moments
    with open(file_paths[0], "r") as csvfile:
        csvreader = csv.reader(csvfile)
        header = next(csvreader)

        apdft_dipoles = []
        nuclear_charges_list = []

        for row in csvreader:
            # x
            if int(row[1]) == 0:
                dipole = []

            dipole.append(float(row[2]))

            # z
            if int(row[1]) == 2:
                nuclear_charges = row[0]
                nuclear_charges_list.append(nuclear_charges)
                # Only APDFT1
                apdft_dipoles.append([dipole])

    # Read electric polarizabilities
    with open(file_paths[1], "r") as csvfile:
        csvreader = csv.reader(csvfile)
        header = next(csvreader)

        apdft_pols = []

        for row in csvreader:
            # xx
            if (int(row[1]) == 0 and int(row[2]) == 0):
                pol = np.zeros((3, 3))

            pol[int(row[1]), int(row[2])] = float(row[3])

            # zz
            if (int(row[1]) == 2 and int(row[2]) == 2):
                # Only APDFT1
                apdft_pols.append([pol])

    apdft_dipoles = np.array(apdft_dipoles).transpose(1, 0, 2)
    apdft_pols = np.array(apdft_pols).transpose(1, 0, 2, 3)

    # apdft_dipoles: (APDFT's n = 1, num_mols, 3)
    # apdft_pols: (APDFT's n = 1, num_mols, 3, 3)
    return nuclear_charges_list, apdft_dipoles, apdft_pols

def calc_relative_energies_from_output(path_xyz_file, unified_alch_isomers=None, distinguish_isomers=True,
                                       path_output_file="apdft_energies.csv", flag_apdft5=False):
    if unified_alch_isomers is None:
        unified_alch_isomers = get_alch_isomers_indexes(path_xyz_file, distinguish_isomers=True)

    nuclear_charges_list, apdft_energies = read_apdft_energies(file_path=path_output_file, flag_apdft5=flag_apdft5)

    if not flag_apdft5:
        if distinguish_isomers:
            relative_energies = {"enantiomers": {"APDFT1": [], "APDFT3": []}, "diastereomers": {"APDFT1": [], "APDFT3": []}}
            keys_relative_energies = ["enantiomers", "diastereomers"]
        else:
            relative_energies = {"isomers": {"APDFT1": [], "APDFT3": []}}
            keys_relative_energies = ["isomers"]
    else:
        if distinguish_isomers:
            relative_energies = {"enantiomers": {"APDFT1": [], "APDFT3": [], "APDFT5": []},
                                 "diastereomers": {"APDFT1": [], "APDFT3": [], "APDFT5": []}}
            keys_relative_energies = ["enantiomers", "diastereomers"]
        else:
            relative_energies = {"isomers": {"APDFT1": [], "APDFT3": [], "APDFT5": []}}
            keys_relative_energies = ["isomers"]

    for idx_alch_isomers, alch_isomers in enumerate(unified_alch_isomers):
        for alch_isomer in alch_isomers:
            # APDFT1
            relative_energies[keys_relative_energies[idx_alch_isomers]]["APDFT1"].append(
                apdft_energies[0, alch_isomer[0]] - apdft_energies[0, alch_isomer[1]])
            # APDFT3
            relative_energies[keys_relative_energies[idx_alch_isomers]]["APDFT3"].append(
                apdft_energies[2, alch_isomer[0]] - apdft_energies[2, alch_isomer[1]])

            if flag_apdft5:
                # APDFT5
                relative_energies[keys_relative_energies[idx_alch_isomers]]["APDFT5"].append(
                    apdft_energies[4, alch_isomer[0]] - apdft_energies[4, alch_isomer[1]])

    return relative_energies

def read_scf_ele_energies(path_output_file):
    with open(path_output_file, 'r') as fh:
        data_string = fh.read()
        lines = data_string.split('\n')
        scf_ele_energies = []
        for line in lines:
            if line.strip():  # check if line is not empty
                scf_ele_ene = float(line)
                scf_ele_energies.append(scf_ele_ene)
        scf_ele_energies = np.array(scf_ele_energies)

    return scf_ele_energies

def read_scf_electric_dipoles(path_output_file="./ele_dipole.dat"):
    scf_ele_dipoles = []
    with open(path_output_file, "r") as file:
        for line in file:
            values = [float(val) for val in line.split()]
            scf_ele_dipoles.append(values)

    return scf_ele_dipoles

def read_scf_electric_polarizabilities(path_output_files=["./ele_pol.csv", "./ele_hyperpol.csv"]):
    electric_pols_scf = []
    electric_hyper_pols_scf = []

    # Read electric polarizabilities
    with open(path_output_files[0], "r") as csvfile:
        csvreader = csv.reader(csvfile)
        header = next(csvreader)
        count = 0
        for row in csvreader:
            if count == 0:
                pol = np.zeros((3, 3))
            count += 1
            pol[int(row[1]), int(row[2])] = float(row[3])
            if count == 9:
                electric_pols_scf.append(pol)
                count = 0

    # Read electric hyper polarizabilities
    with open(path_output_files[1], "r") as csvfile:
        csvreader = csv.reader(csvfile)
        header = next(csvreader)
        count = 0
        for row in csvreader:
            if count == 0:
                pol = np.zeros((3, 3, 3))
            count += 1
            pol[int(row[1]), int(row[2]), int(row[3])] = float(row[4])
            if count == 27:
                electric_hyper_pols_scf.append(pol)
                count = 0

    return electric_pols_scf, electric_hyper_pols_scf

def read_scf_magnetic_susceptibility(path_output_file="./mag_susp.csv"):
    magnetic_susp_scf = []

    # Read magnetic susceptibility
    with open(path_output_file, "r") as csvfile:
        csvreader = csv.reader(csvfile)
        header = next(csvreader)
        count = 0
        for row in csvreader:
            if count == 0:
                susp = np.zeros((3, 3))
            count += 1
            susp[int(row[1]), int(row[2])] = float(row[3])
            if count == 9:
                magnetic_susp_scf.append(susp)
                count = 0

    return magnetic_susp_scf

def calc_scf_relative_energies_from_output(path_xyz_file, unified_alch_isomers=None, distinguish_isomers=True,
                                           path_output_file="scf_ele_ene.dat", mode='standard'):
    if mode not in ['standard', 'selection']:
        raise ValueError("mode must be either 'standard' or 'selection'.")

    if mode == 'standard':
        if unified_alch_isomers is None:
            alch_isomers = get_alch_isomers_indexes(path_xyz_file)

        scf_ele_energies = read_scf_ele_energies(path_output_file)

        if distinguish_isomers:
            scf_relative_energies = {"enantiomers": [], "diastereomers": []}
            keys_scf_relative_energies = ["enantiomers", "diastereomers"]
        else:
            scf_relative_energies = {"isomers": []}
            keys_scf_relative_energies = ["isomers"]

        for idx_alch_isomers, alch_isomers in enumerate(unified_alch_isomers):
            for alch_isomer in alch_isomers:
                scf_relative_energies[keys_scf_relative_energies[idx_alch_isomers]].append(
                    scf_ele_energies[alch_isomer[0]] - scf_ele_energies[alch_isomer[1]])

    elif mode == 'selection':
        if unified_alch_isomers is None:
            raise ValueError("unified_alch_isomers must be provided if mode is 'selection'.")

        randomly_selected_scf_ele_enes_dict = read_randomly_selected_scf_ele_enes_from_output()

        if distinguish_isomers:
            scf_relative_energies = {"enantiomers": [], "diastereomers": []}
            keys_scf_relative_energies = ["enantiomers", "diastereomers"]
        else:
            scf_relative_energies = {"isomers": []}
            keys_scf_relative_energies = ["isomers"]

        for idx_alch_isomers, alch_isomers in enumerate(unified_alch_isomers):
            for alch_isomer in alch_isomers:
                scf_relative_energies[keys_scf_relative_energies[idx_alch_isomers]].append(
                    randomly_selected_scf_ele_enes_dict[alch_isomer[0]] - \
                        randomly_selected_scf_ele_enes_dict[alch_isomer[1]])

    return scf_relative_energies

def Levy_estimate_relative_energies(ac1, ac2, flag_rev_order=False):
    """ Calculate relative electronic energies between two alchemical diastereomers using Levy's approximation.
    """
    if not flag_rev_order:
        nuc_charges_vecs = ac1.mol.atom_charges() - ac2.mol.atom_charges()
    else:
        nuc_charges_vecs = ac2.mol.atom_charges() - ac1.mol.atom_charges()

    integral1 = ac1.get_Levy_relative_elec_ene_component()
    integral2 = ac2.get_Levy_relative_elec_ene_component()
    rel_ele_ene = np.dot(nuc_charges_vecs, (integral1 + integral2) / 2)

    return rel_ele_ene

def Levy_estimate_relative_energies_pairs_list(path_xyz_file, dft_functional,
                                               name_basis_set, unified_mol_pair_indexes=None,
                                               mutation_sites=None, distinguish_isomers=True):
    """ Levy's estimate of relative energies for given pairs of isoelectronic molecules.
        Need to note that the employed basis set covers all the target atoms.
    Args:
    - dft_functional: A string containing the name of DFT functional, e.g., "pbe0"
    - name_basis_set: A string containing the name of basis set, e.g., "def2-tzvp" or
                      a dictionary of basis set, e.g., {"H":bse.get_basis("pc-2",fmt="nwchem",elements=[1]),'C':bse.get_basis("pcX-2",fmt="nwchem",elements=[6])}
    """
    if unified_mol_pair_indexes is None:
        unified_mol_pair_indexes = get_alch_isomers_indexes(path_xyz_file)

    nuclear_charges, coordinates = read_xyz_file(path_xyz_file)

    # Read nuc_charges_vecs from nuc_charges_list file "target_molecules.inp"
    nuc_charge_list = read_nuc_charges_list()

    if distinguish_isomers:
        rel_ele_ene = {"enantiomers": [], "diastereomers": []}
        keys_rel_ele_ene = ["enantiomers", "diastereomers"]
    else:
        rel_ele_ene = {"isomers": []}
        keys_rel_ele_ene = ["isomers"]

    for idx_mol_pair_indexes, mol_pair_indexes in enumerate(unified_mol_pair_indexes):
        for mol_pair in mol_pair_indexes:
            atom_strings = [bse.lut.element_sym_from_Z(part, normalize=True) for part in nuc_charge_list[mol_pair[0]]]
            target_mol = generate_data_string(atom_strings, coordinates)
            ac1 = ac(target_mol, name_basis_set, dft_functional, sites=mutation_sites, bse_off=False, AP_skip=True)

            atom_strings = [bse.lut.element_sym_from_Z(part, normalize=True) for part in nuc_charge_list[mol_pair[1]]]
            target_mol = generate_data_string(atom_strings, coordinates)
            ac2 = ac(target_mol, name_basis_set, dft_functional, sites=mutation_sites, bse_off=False, AP_skip=True)

            nuc_charges_vec = nuc_charge_list[mol_pair[0]] - nuc_charge_list[mol_pair[1]]

            integral1 = ac1.get_Levy_relative_elec_ene_component()
            integral2 = ac2.get_Levy_relative_elec_ene_component()
            rel_ele_ene[keys_rel_ele_ene[idx_mol_pair_indexes]].append(
                np.dot(nuc_charges_vec, (integral1 + integral2) / 2))

    return rel_ele_ene

# --------------------------- Calculation ---------------------------#


# --------------------------- Analysis ---------------------------#

def read_cubic_from_csv(name_cubic_file="ele_ene_derivatives_3rd.csv"):
    """ Read cubic hessian from a csv file. """
    data = np.genfromtxt(name_cubic_file, delimiter=',', skip_header=1)
    num_sites = int(np.max(data[:, :3])) + 1
    cubic = np.zeros((num_sites, num_sites, num_sites))
    count = -1
    for i in range(num_sites):
        for j in range(num_sites):
            for k in range(num_sites):
                count += 1
                cubic[i, j, k] = data[count, 3]

    return cubic

def decompose_3rd_ele_ene_term(pvec):
    # pvec: nuclear charge vector for selected atom sites
    pvec = np.array(pvec)

    cubic = read_cubic_from_csv()
    ene_cubic = np.zeros_like(cubic)

    e3 = 0.0
    for i in range(len(cubic)):
        for j in range(len(cubic)):
            for k in range(len(cubic)):
                e3 += cubic[i, j, k] * pvec[i] * pvec[j] * pvec[k]
                ene_cubic[i, j, k] = cubic[i, j, k] * pvec[i] * pvec[j] * pvec[k]
                # print(i, j, k, ene_cubic[i, j, k])

    e3 *= 1/6
    # For the relative energy, 1/3 should be multiplied.
    ene_cubic *= 1/6

    return ene_cubic

# def get_alch_diastereomers(symm_atoms=None):
#     """ It is a dangerous function. It is not general. """
#     print("WARNING: get_alch_diastereomers() is not general. It only works for some systems.")
#     nuclear_charges_list = read_nuc_charges_list()
#     nuc_charges_vecs = calc_nuc_charge_vecs(nuclear_charges_list)

#     # For benzene
#     # symm_atoms = [[1, 5], [2, 4]]
#     # symm_atoms = np.array(symm_atoms)

#     alch_diastereomers = []
#     for i, nuc_charges_vec1 in enumerate(nuc_charges_vecs):
#         for j, nuc_charges_vec2 in enumerate(nuc_charges_vecs[i+1:]):
#             if np.all(nuc_charges_vec1 + nuc_charges_vec2 == 0):
#                 alch_diastereomers.append([i, i+1+j])

#             # If symm_atoms is provided, check if the two molecules are diastereomers
#             # by considering symmetry.
#             # The current implementation is not general.
#             # It only works for benzene.
#             elif symm_atoms is not None:
#                 temp10 = nuc_charges_vec1[symm_atoms[0, 0]]
#                 temp11 = nuc_charges_vec1[symm_atoms[0, 1]]
#                 temp20 = nuc_charges_vec1[symm_atoms[1, 0]]
#                 temp21 = nuc_charges_vec1[symm_atoms[1, 1]]
#                 temp_nuc_charges_vec1 = np.copy(nuc_charges_vec1)
#                 temp_nuc_charges_vec1[symm_atoms[0, 0]] = temp11
#                 temp_nuc_charges_vec1[symm_atoms[0, 1]] = temp10
#                 temp_nuc_charges_vec1[symm_atoms[1, 0]] = temp21
#                 temp_nuc_charges_vec1[symm_atoms[1, 1]] = temp20
#                 if np.all(temp_nuc_charges_vec1 + nuc_charges_vec2 == 0):
#                     alch_diastereomers.append([i, i+1+j])

#     alch_diastereomers = np.array(alch_diastereomers)

#     return alch_diastereomers

def get_alch_isomers_indexes(path_xyz_file, mutation_sites=None, expert_mode=False, distinguish_isomers=True):
    # Note that mutation_sites are not used as mutation sites.
    # It is just used to get the number of the Coulomb matrix size.

    # Get nuclear charge vectors
    # Read_nuc_charges_list reads target_molecules.inp
    nuc_charge_list = read_nuc_charges_list()
    nuc_charges_vecs = calc_nuc_charge_vecs(nuc_charge_list)

    if mutation_sites is None:
        mutation_sites = list(range(len(nuc_charge_list[0])))

    classified_atoms = get_atoms_indexes_in_same_chem_env_from_xyz_file(
        path_xyz_file, matrix_size=len(mutation_sites))

    # For avoiding the mistake
    if not expert_mode:
        if np.all(nuc_charge_list[0] != read_xyz_file(path_xyz_file)[0]):
            raise ValueError("The nuclear charges in target_molecules.inp and the xyz file are different.")

    # In alchemical enantiomers or diastereomers (dubbed as "isomers"), the difference
    # between mutated atoms and reference atoms are oposite in the same chemical environment.

    # Grouping all the nuclear charges vectors into the chemical environments
    grouped_nuc_charges_vecs = []
    for nuc_charge_vec in nuc_charges_vecs:
        grouped_nuc_charges_vec = []

        for atom_subgroup in classified_atoms:
            grouped_nuc_charges = []
            for idx_atom in atom_subgroup:
                grouped_nuc_charges.append(nuc_charge_vec[idx_atom])
            grouped_nuc_charges_vec.append(sorted(grouped_nuc_charges))
        grouped_nuc_charges_vecs.append(grouped_nuc_charges_vec)

    # Classify alchemical enantiomers and diastereomers
    unified_grouped_nuc_charges_vecs = []
    idx_nuc_charge_list = []
    if distinguish_isomers:
        idx_nuc_charge_list.append([])
        idx_nuc_charge_list.append([])
        grouped_nuc_charges_vecs_enantiomers = []
        grouped_nuc_charges_vecs_diastereomers = []
        for i, grouped_nuc_charges_vec in enumerate(grouped_nuc_charges_vecs):
            flag_enantiomer = True
            for atom_subgroup in grouped_nuc_charges_vec:
                if sum(atom_subgroup) != 0:
                    flag_enantiomer = False
                    break

            if flag_enantiomer:
                grouped_nuc_charges_vecs_enantiomers.append(grouped_nuc_charges_vec)
                idx_nuc_charge_list[0].append(i)
            else:
                grouped_nuc_charges_vecs_diastereomers.append(grouped_nuc_charges_vec)
                idx_nuc_charge_list[1].append(i)

        unified_grouped_nuc_charges_vecs.append(grouped_nuc_charges_vecs_enantiomers)
        unified_grouped_nuc_charges_vecs.append(grouped_nuc_charges_vecs_diastereomers)
    else:
        idx_nuc_charge_list.append(list(range(len(nuc_charges_vecs))))
        unified_grouped_nuc_charges_vecs.append(grouped_nuc_charges_vecs)

    del grouped_nuc_charges_vecs

    # Obtaining candidate alchemical isomers by classifying the chemical environments
    # of the nuclear charge vectors
    # Pairs of indexes of nuclear charges vectors which are candidates of
    # alchemical isomers.
    unified_candidate_alchemical_isomers_indexes = []
    for idx_isomer_types, grouped_nuc_charges_vecs in enumerate(unified_grouped_nuc_charges_vecs):
        unified_candidate_alchemical_isomers_indexes.append([])
        if grouped_nuc_charges_vecs == []:
            continue

        for i, grouped_nuc_charges_vec1 in enumerate(grouped_nuc_charges_vecs):
            reverse_grouped_nuc_charges_vec1 = []
            for part in grouped_nuc_charges_vec1:
                reverse_grouped_nuc_charges_vec1.append(sorted([-x for x in part]))
            idx1 = idx_nuc_charge_list[idx_isomer_types][i]

            for j, grouped_nuc_charges_vec2 in enumerate(grouped_nuc_charges_vecs[i+1:]):
                idx2 = idx_nuc_charge_list[idx_isomer_types][i + j + 1]

                if reverse_grouped_nuc_charges_vec1 == grouped_nuc_charges_vec2:
                    unified_candidate_alchemical_isomers_indexes[idx_isomer_types].append([idx1, idx2])

    reference_nuc_charges, reference_coordinates = read_xyz_file(path_xyz_file)

    unified_alchemical_isomers_indexes = []
    for idx_isomer_types, candidate_alchemical_isomers_indexes in enumerate(unified_candidate_alchemical_isomers_indexes):
        unified_alchemical_isomers_indexes.append([])
        if candidate_alchemical_isomers_indexes == []:
            continue

        for candidate_alchemical_isomers_index_pair in candidate_alchemical_isomers_indexes:
            charges1 = nuc_charge_list[candidate_alchemical_isomers_index_pair[0]]
            charges2 = reference_nuc_charges - nuc_charges_vecs[candidate_alchemical_isomers_index_pair[1]]
            cm1 = amr.generate_coulomb_matrix(charges1, reference_coordinates, size=len(mutation_sites), sorting="row-norm")
            cm2 = amr.generate_coulomb_matrix(charges2, reference_coordinates, size=len(mutation_sites), sorting="row-norm")
            if np.allclose(cm1, cm2, atol=1e-3):
                unified_alchemical_isomers_indexes[idx_isomer_types].append(candidate_alchemical_isomers_index_pair)

    # check the duplication
    def check_duplicates(lst):
        seen = set()
        for pair in lst:
            for num in pair:
                if num in seen:
                    return True
                seen.add(num)
        return False

    for alchemical_isomers_indexes in unified_alchemical_isomers_indexes:
        flag_duplication = check_duplicates(alchemical_isomers_indexes)
        if flag_duplication:
            raise ValueError("Duplicated pairs of indexes of alchemical isomers.")

    return unified_alchemical_isomers_indexes

def get_atoms_indexes_in_same_chem_env_from_xyz_file(path_xyz_file, matrix_size=None):
    nuclear_charges, coordinates = read_xyz_file(path_xyz_file)
    nuclear_charges = np.array(nuclear_charges)
    coordinates = np.array(coordinates)
    num_atoms = len(nuclear_charges)
    if matrix_size is None:
        matrix_size = num_atoms
    else:
        num_atoms = matrix_size

    # The unsorted matrix is used to maintain the order of the atoms
    lower_cm = amr.generate_coulomb_matrix(nuclear_charges, coordinates, size=matrix_size, sorting="unsorted")

    def get_symmetric_matrix_from_lower_trigonal_part(lower_trigonal_part, dim):
        symmetric_matrix = np.zeros((dim, dim))
        count = 0
        for j in range(dim):
            symmetric_matrix[j, 0:j+1] = lower_trigonal_part[count:count+j+1]
            count += j + 1
        symmetric_matrix = symmetric_matrix + symmetric_matrix.T
        for i in range(dim):
            symmetric_matrix[i, i] /= 2.0

        return symmetric_matrix

    cm = get_symmetric_matrix_from_lower_trigonal_part(lower_cm, num_atoms)

    # row_norm_cm = np.zeros(num_atoms)
    # for i in range(num_atoms):
    #     row_norm_cm[i] = np.linalg.norm(cm[i])

    def create_subgroups(matrix, tolerance=0.0001):
        subgroups = []
        visited = set()

        for i, row in enumerate(matrix):
            if i in visited:
                continue

            sorted_row = np.sort(row)

            subgroup = [i]
            for j, other_row in enumerate(matrix[i+1:], start=i+1):
                sorted_other_row = np.sort(other_row)
                # if j not in visited and np.all(np.abs(sorted_row - sorted_other_row) <= tolerance):
                if j not in visited and np.allclose(sorted_row, sorted_other_row, atol=tolerance):
                    subgroup.append(j)
                    visited.add(j)

            subgroups.append(subgroup)

        return subgroups

    atoms_indexes = create_subgroups(cm)

    # for i, subgroup in enumerate(atoms_indexes):
    #     print(f'Subgroup {i+1}:')
    #     for idx in subgroup:
    #         print(f'  Row {idx+1}')

    return atoms_indexes

# --------------------------- Analysis ---------------------------#


# --------------------------- IO ---------------------------#

def parse_atomic_coordinates(data_string):
    """ Parse atomic coordinates from a string. The string should be in the format of NWChem input file.

    Args:
        data_string: A string containing atomic coordinates.

    Returns:
        atom_types: A list of atom types.
        coordinates: A numpy array of shape (n_atoms, 3) containing atomic coordinates.
    """
    coordinates = []
    atom_types = []

    lines = data_string.strip().split('\n')
    for line in lines:
        parts = line.split()
        try:
            atom_type = int(parts[0])
        except:
            atom_type = parts[0]
        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])

        atom_types.append(atom_type)
        coordinates.append((x, y, z))

    coordinates = np.array(coordinates)

    return atom_types, coordinates

def generate_data_string(atom_types, coordinates):
    """ Generate a string containing atomic coordinates. The string is in the format of NWChem input file.

    Args:
        atom_types: A list of atom types.
        coordinates: A numpy array of shape (n_atoms, 3) containing atomic coordinates.

    Returns:
        data_string: A string containing atomic coordinates.
    """
    data_string = ""

    for i in range(len(coordinates)):
        x, y, z = coordinates[i]
        atom_type = atom_types[i]
        data_string += f"{atom_type:4s}{x:10.6f}{y:10.6f}{z:10.6f}\n"

    return data_string

def read_xyz_file(file_path):
    with open(file_path, 'r') as f:
        num_atoms = int(f.readline().strip())
        f.readline()
        atom_types, coordinates = parse_atomic_coordinates(f.read())
        if all(isinstance(x, int) for x in atom_types):
            nuclear_charges = atom_types
        elif all(isinstance(x, str) for x in atom_types):
            nuclear_charges = [bse.lut.element_Z_from_sym(part) for part in atom_types]
        else:
            raise ValueError("Atom types must be either all integers or all strings.")

        return nuclear_charges, coordinates

def write_gjf(atom_numbers, coords, name):
    natom = len(atom_numbers)

    if natom != len(coords):
        raise ValueError("Generation of a gjf file.")

    with open(str(name) + ".gjf", mode='w') as fh:
        fh.write("%mem=10GB" + "\n")
        fh.write("%nproc=6" + "\n")
        fh.write("# pbe1pbe/def2tzvp scf=maxcycle=500" + "\n")
        fh.write("\n")
        fh.write(" ".join(map(str, atom_numbers)) + "\n")
        fh.write("\n")
        fh.write("0 1" + "\n")

        for i in range(natom):
            fh.write(f"{atom_numbers[i]} {' '.join(map(str, coords[i]))}\n")


        fh.write("\n")

def write_gjf_gen(atom_numbers, coords, name, basis_set_dict):
    natom = len(atom_numbers)

    if natom != len(coords):
        raise ValueError("Generation of a gjf file.")

    basis_set_str = get_basis_set_str_gaussian(atom_numbers, basis_set_dict)

    with open(str(name) + ".gjf", mode='w') as fh:
        fh.write("# pbe1pbe/gen scf=maxcycle=500" + "\n")
        fh.write("\n")
        fh.write(" ".join(map(str, atom_numbers)) + "\n")
        fh.write("\n")
        fh.write("0 1" + "\n")

        for i in range(natom):
            fh.write(f"{atom_numbers[i]} {' '.join(map(str, coords[i]))}\n")

        fh.write("\n")
        fh.write(basis_set_str)
        fh.write("\n")
        fh.write("\n")

def get_basis_set_str_gaussian(nuclear_numbers, basis_set_dict):
    """ Get the basis set string for Gaussian input file from the basis set dictionary.

    The example of basis set dictionary is:
    basis_set_dict = {1:bse.get_basis("pc-2",fmt="gaussian94",elements=[1], header=False),
                  5:bse.get_basis("pcX-2",fmt="gaussian94",elements=[5], header=False),
                  6:bse.get_basis("pcX-2",fmt="gaussian94",elements=[6], header=False),
                  7:bse.get_basis("pcX-2",fmt="gaussian94",elements=[7], header=False)}
    """
    # Get the unique nuclear numbers
    unique_nuclear_numbers = list(set(nuclear_numbers))

    # Get the basis set string for Gaussian input file
    basis_set_str = ""
    for nuclear_number in unique_nuclear_numbers:
        basis_set_str += basis_set_dict[nuclear_number]

    return basis_set_str

def get_data_string_from_nuc_charge_vecs(data_string, nuc_charge_vecs, sites):
    """ Get a new data string from a given data string and a list of nuclear charge vectors.

    Args:
        data_string: A string containing atomic coordinates.
        nuc_charge_vecs: A list of nuclear charge vectors.
        sites: A list of site indices.

    Returns:
        new_data_string: A string containing atomic coordinates.
    """
    atom_types, coordinates = parse_atomic_coordinates(data_string)
    nuc_charges = [bse.lut.element_Z_from_sym(part) for part in atom_types]
    nuc_charges = np.array(nuc_charges)

    for i, site in enumerate(sites):
        nuc_charges[site] += nuc_charge_vecs[i]

    if nuc_charges.min() < 0:
        raise ValueError("Negative nuclear charge detected.")

    atom_strings = [bse.lut.element_sym_from_Z(part, normalize=True) for part in nuc_charges]
    new_data_string = generate_data_string(atom_strings, coordinates)

    return new_data_string

def calc_nuc_charge_vecs(nuc_charges_list, ref_nuc_charges=None):
    """ Calculate the nuclear charge vectors.

    Args:
        nuc_charges_list: A list of nuclear charges of possible molecules.
        ref_nuc_charges: A list of reference nuclear charges.

    Returns:
        nuc_charge_vecs: A numpy array of shape (n_molecules, n_atoms) containing nuclear charge vectors.
    """
    if nuc_charges_list.min() < 0:
        raise ValueError("Negative nuclear charge detected.")

    nuc_charges_list = np.array(nuc_charges_list)
    if ref_nuc_charges is not None:
        ref_nuc_charges = np.array(ref_nuc_charges)

    nuc_charge_vecs = np.zeros_like(nuc_charges_list)

    for i, nuc_charges in enumerate(nuc_charges_list):
        if ref_nuc_charges is None:
            nuc_charge_vecs[i] = nuc_charges - nuc_charges_list[0]
        elif ref_nuc_charges is not None:
            nuc_charge_vecs[i] = nuc_charges - ref_nuc_charges

    return nuc_charge_vecs

def read_nuc_charges_list(nuc_charges_list_file="target_molecules.inp"):
  """ Read nuclear charges of molecules from a file.

  Args:
      nuc_charges_list_file: A file containing nuclear charges of molecules.

  Returns:
      nuc_charges_list: A numpy array of nuclear charges of molecules.
  """
  with open(nuc_charges_list_file, 'r') as fh:
      data_string = fh.read()
      lines = data_string.split('\n')
      nuc_charges_list = []
      for line in lines:
          if line.strip():  # check if line is not empty
              nuc_charges = list(map(int, line.split(',')))
              nuc_charges_list.append(nuc_charges)
      nuc_charges_list = np.array(nuc_charges_list)

  return nuc_charges_list

def write_csv_output_APDFT_energies(apdft_energies, nuc_charges_list=None, output_file="apdft_energies.csv", mode='standard',
                                    flag_apdft5=False):
    """ Write APDFT energies to a CSV file using pandas.

    Args:
        apdft_energies: A numpy array containing the APDFT energies to be written.
        nuc_charges_list: A numpy array containing the nuclear charges of the molecules.
        output_file: The path to the output CSV file.
    """
    if mode not in ['standard', 'random']:
        raise ValueError("mode must be either 'standard' or 'random'.")

    # If nuc_charges_list is not provided, read it from file "target_molecules.inp"
    if nuc_charges_list is None:
        if mode == 'standard':
            nuc_charges_list = read_nuc_charges_list()
        elif mode == 'random':
            nuc_charges_list = read_nuc_charges_list(
                nuc_charges_list_file="randomly_selected_target_molecules.inp")

    # Convert nuclear charges to strings for use as index in DataFrame
    nuc_charges_list_str = [','.join(map(str, nuc_charges)) for nuc_charges in nuc_charges_list]

    # Create DataFrame with APDFT energies and nuclear charges as index
    if not flag_apdft5:
        data = pd.DataFrame(apdft_energies, index=nuc_charges_list_str, columns=['APDFT1', 'APDFT2', 'APDFT3'])
    else:
        data = pd.DataFrame(apdft_energies, index=nuc_charges_list_str, columns=['APDFT1', 'APDFT2', 'APDFT3', 'APDFT4', 'APDFT5'])

    # Write DataFrame to CSV file
    data.to_csv(output_file, index_label='nuclear_charges')

def write_csv_output_APDFT_response_properties(apdft_electric_dipoles, apdft_electric_pols, nuc_charges_list=None,
                                               output_files=["apdft_electric_dipoles.csv", "apdft_electric_pols.csv"],
                                               mode='standard', flag_apdft1_only=False):
    """ Write APDFT response properties to a CSV file using pandas.

    Args:
        apdft_electric_dipoles: A numpy array containing the APDFT electric dipole moments to be written.
        apdft_electric_pols: A numpy array containing the APDFT electric polarizabilities to be written.
        nuc_charges_list: A numpy array containing the nuclear charges of the molecules.
        output_file: The path to the output CSV file.
    """
    if mode not in ['standard', 'random']:
        raise ValueError("mode must be either 'standard' or 'random'.")

    # If nuc_charges_list is not provided, read it from file "target_molecules.inp"
    if nuc_charges_list is None:
        if mode == 'standard':
            nuc_charges_list = read_nuc_charges_list()
        elif mode == 'random':
            nuc_charges_list = read_nuc_charges_list(
                nuc_charges_list_file="randomly_selected_target_molecules.inp")

    # Convert nuclear charges to strings for use as index in DataFrame
    nuc_charges_list_str = [','.join(map(str, nuc_charges)) for nuc_charges in nuc_charges_list]

    # Create DataFrame with APDFT electric dipole moments and nuclear charges as index
    # data_dipoles = pd.DataFrame(apdft_electric_dipoles[:, 0, :], index=nuc_charges_list_str, columns=['x', 'y', 'z'])
    if not flag_apdft1_only:
        ds_dipoles = xr.DataArray(apdft_electric_dipoles[:, 0, :],
                                  dims=('nuclear_charges', 'Cartesian1'),
                                  coords={'nuclear_charges': nuc_charges_list_str})
    else:
        ds_dipoles = xr.DataArray(apdft_electric_dipoles[:, :],
                                  dims=('nuclear_charges', 'Cartesian1'),
                                  coords={'nuclear_charges': nuc_charges_list_str})
    # Create DataFrame with APDFT electric polarizabilities and nuclear charges as index
    # data_pols = pd.DataFrame(apdft_electric_pols[:, 0, :, :], index=nuc_charges_list_str, columns=['APDFT1'])
    if not flag_apdft1_only:
        ds_pols = xr.DataArray(apdft_electric_pols[:, 0, :, :],
                            dims=('nuclear_charges', 'Cartesian1', 'Cartesian2'),
                            coords={'nuclear_charges': nuc_charges_list_str})
    else:
        ds_pols = xr.DataArray(apdft_electric_pols[:, :, :],
                            dims=('nuclear_charges', 'Cartesian1', 'Cartesian2'),
                            coords={'nuclear_charges': nuc_charges_list_str})

    # Write DataFrame to CSV file
    # data_dipoles.to_csv(output_file[0], index_label='nuclear_charges')
    # data_pols.to_csv(output_file[1], index_label='nuclear_charges')
    ds_dipoles.to_dataframe(name='APDFT1_electric_dipoles').to_csv(output_files[0])
    ds_pols.to_dataframe(name='APDFT1_electric_pols').to_csv(output_files[1])

def write_csv_output_ele_ene_derivatives(ac_class, output_file="ele_ene_derivatives.csv"):
    """ Write electronic energy derivatives to a CSV file using pandas.

    Args:
        ac_class: An instance of the alchemical claculator class.
    """
    # First derivatives
    data = pd.DataFrame(ac_class.ap.gradient, index=ac_class.sites, columns=['dE/dZ'])
    data.to_csv(output_file.replace('.csv', '_1st.csv'), index_label='Atom sites')

    # Second derivatives
    data = pd.DataFrame(ac_class.ap.hessian, index=ac_class.sites, columns=ac_class.sites)
    data.to_csv(output_file.replace('.csv', '_2nd.csv'), index_label='Atom sites')

    # Third derivatives
    # Pandas cannot handle 3D data, so we use xarray (xr here) instead.
    ds = xr.DataArray(ac_class.ap.cubic, dims=('atom_sites1', 'atom_sites2', 'atom_sites3'))
    csv_filename = output_file.replace('.csv', '_3rd.csv')
    ds.to_dataframe(name='d3E/dZ3').to_csv(csv_filename)

def write_scf_ele_ene_dat(ele_enes, output_file="scf_ele_ene.dat"):
    """" Write electronic energies calculated by SCF to a dat file. """
    with open(output_file, 'w') as fh:
        for ele_ene in ele_enes:
            fh.write(f'{ele_ene}\n')

def write_csv_output_ele_response_property_derivatives(ac_class,
                                                       output_file="ele_ene_electric_field_derivatives.csv"):
    """ Write electronic energy derivatives for response properties to a CSV file using pandas.

    Args:
        ac_class: An instance of the alchemical claculator class.
    """
    if not ac_class.flag_response_property:
        raise ValueError("flag_response_property should be True to calculate response properties")

    # Second derivatives
    data = pd.DataFrame(ac_class.ap.electric_dipole_gradient, index=ac_class.sites,
                        columns=['x', 'y', 'z'])
    data.to_csv(output_file.replace('.csv', '_2nd.csv'), index_label='Atom sites')

    # Third derivatives
    # Pandas cannot handle 3D data, so we use xarray (xr here) instead.
    ds = xr.DataArray(ac_class.ap.electric_polarizability_gradient,
                      dims=('atom_sites1', 'Cartesian1', 'Cartesian2'))
    csv_filename = output_file.replace('.csv', '_3rd.csv')
    ds.to_dataframe(name='d3E/dZdF2').to_csv(csv_filename)
# --------------------------- IO ---------------------------#
