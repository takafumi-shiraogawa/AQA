from pyscf import gto, scf
import basis_set_exchange as bse
from AP_class import APDFT_perturbator as AP
from FcMole import FcM_like
from mod_pyscf_scf import RHF, RKS

class alchemical_calculator():
    """ Alchemical calculator class """

    def __init__(self, mol_coord, name_basis_set, dft_functional, sites=None,
                left_elec_energy=None, coord_unit="Angstrom", bse_off=False,
                guess=None, grid_level=3, scf_skip=False, AP_skip=False,
                scf_conv=None, scf_max_cycle=None, direct_scf=True,
                max_cycle_cphf=40, conv_tol_cphf=1e-9,
                fractional_charges_calc=False, fractional_charges=None,
                mod_SCF=True, flag_norm_ddm_criterion=False, norm_ddm_criterion=1.e-6,
                flag_response_property=False, charges_for_center=None):
        self.mol_coord  = mol_coord
        if not bse_off:
            self.basis_set = bse.get_basis(name_basis_set, fmt="nwchem")
        else:
            self.basis_set = name_basis_set
        self.dft_functional = dft_functional
        self.sites = sites
        self.mol = gto.M(atom=self.mol_coord, unit=coord_unit, basis=self.basis_set)

        if self.sites is None:
            self.sites = list(range(len(self.mol.atom_charges())))

        if fractional_charges_calc:
            if len(fractional_charges) != len(self.sites):
                raise ValueError("Number of fractional charges must be equal to the number of sites.")

            if guess is None:
                print("Warning: Without guess='1e', SCF may fail.")

            additional_charges = []
            for i in range(len(self.mol.atom_charges())):
                if i in self.sites:
                    additional_charges.append(fractional_charges[self.sites.index(i)])
                else:
                    additional_charges.append(0.0)

            self.mol = FcM_like(self.mol, fcs=additional_charges)

        if self.dft_functional is not None:
            if mod_SCF:
                self.mf = RKS(self.mol)
            else:
                self.mf = scf.RKS(self.mol)
            self.mf.xc = self.dft_functional
            self.mf.grids.level = grid_level
        else:
            print("Warning: DFT functional not specified. Using HF.")
            if mod_SCF:
                self.mf = RHF(self.mol)
            else:
                self.mf = scf.RHF(self.mol)

        if scf_conv is not None:
            # Default is 1e-9
            self.mf.conv_tol = scf_conv
        if scf_max_cycle is not None:
            self.mf.max_cycle = scf_max_cycle
        if not direct_scf:
            self.mf.direct_scf = direct_scf

        if not scf_skip:
            if guess is None:
                if mod_SCF:
                    self.mf.scf(flag_norm_ddm_criterion=flag_norm_ddm_criterion,
                                norm_ddm_criterion=norm_ddm_criterion)
                else:
                    self.mf.scf()
            elif guess == "1e":
                if mod_SCF:
                    self.mf.scf(dm0=self.mf.init_guess_by_1e(),
                                flag_norm_ddm_criterion=flag_norm_ddm_criterion,
                                norm_ddm_criterion=norm_ddm_criterion)
                else:
                    self.mf.scf(dm0=self.mf.init_guess_by_1e())
            else:
                raise ValueError("Unknown guess type: {}".format(guess))

        if not self.mf.converged and not scf_skip:
            raise ValueError("SCF did not converge.")

        self.flag_response_property = flag_response_property

        self.ap = AP(self.mf, sites=self.sites, flag_CP_skip=AP_skip,
                     max_cycle_cphf=max_cycle_cphf, conv_tol_cphf=conv_tol_cphf,
                     flag_response_property=self.flag_response_property,
                     charges_for_center=charges_for_center)

        self.left_elec_energy = left_elec_energy

    def get_elec_energy(self):
        return self.mf.e_tot - self.mf.energy_nuc()

    def calc_all_derivatives(self):
        self.ap.build_elec_gradient()
        self.ap.build_elec_hessian()
        self.ap.build_cubic()
        if self.flag_response_property:
            self.ap.build_elec_electric_dipole_gradient()
            self.ap.build_electric_polarizability_gradient()

    def calc_response_properties_derivatives(self):
        if self.flag_response_property:
            self.ap.build_elec_electric_dipole_gradient()
            self.ap.build_electric_polarizability_gradient()

    def get_elec_APDFT0(self, nuc_charges_vec):
        return self.ap.elec_APDFT0(nuc_charges_vec)

    def get_elec_APDFT1(self, nuc_charges_vec):
        return self.ap.elec_APDFT1(nuc_charges_vec)

    def get_elec_APDFT2(self, nuc_charges_vec):
        return self.ap.elec_APDFT2(nuc_charges_vec)

    def get_elec_APDFT3(self, nuc_charges_vec):
        return self.ap.elec_APDFT3(nuc_charges_vec)

    def get_elec_APDFT4(self, nuc_charges_vec):
        return self.ap.elec_APDFT4(nuc_charges_vec)

    def get_elec_APDFT5(self, nuc_charges_vec):
        return self.ap.elec_APDFT5(nuc_charges_vec)

    def get_relative_elec_APDFT1(self, nuc_charges_vec):
        return self.ap.relative_elec_APDFT1(nuc_charges_vec)

    def get_relative_elec_APDFT3(self, nuc_charges_vec):
        return self.ap.relative_elec_APDFT3(nuc_charges_vec)

    def get_even_elec_APDFT0(self, nuc_charges_vec):
        return self.ap.even_elec_estimate_APDFT0(nuc_charges_vec, self.left_elec_energy)

    def get_even_elec_APDFT2(self, nuc_charges_vec):
        return self.ap.even_elec_estimate_APDFT2(nuc_charges_vec, self.left_elec_energy)

    def get_Levy_relative_elec_ene_component(self):
        return self.ap.build_elec_gradient()

    def get_elec_electric_dipole_APDFT0(self, nuc_charges_vec):
        return self.ap.elec_electric_dipole_APDFT0(nuc_charges_vec)

    def get_electric_dipole_APDFT0(self, nuc_charges_vec):
        return self.ap.electric_dipole_APDFT0(nuc_charges_vec)

    def get_elec_electric_dipole_APDFT1(self, nuc_charges_vec):
        return self.ap.elec_electric_dipole_APDFT1(nuc_charges_vec)

    def get_electric_dipole_APDFT1(self, nuc_charges_vec):
        return self.ap.electric_dipole_APDFT1(nuc_charges_vec)

    def get_elec_electric_polarizability_APDFT0(self, nuc_charges_vec):
        return self.ap.elec_electric_polarizability_APDFT0(nuc_charges_vec)

    def get_elec_electric_polarizability_APDFT1(self, nuc_charges_vec):
        return self.ap.elec_electric_polarizability_APDFT1(nuc_charges_vec)
