from pyscf import gto, scf
import numpy as np
from pyscf import lib
from .aaff import alc_deriv_grad_nuc,aaff_resolv,inefficient_aaff_resolv,aaff_xc_resolv
from .FcMole import FcM_like
from .AP_utils import alias_param,parse_charge,DeltaV,charge2symbol
from .alch_deriv import * # Including alchemy_cphf_deriv and electric_field_cphf_deriv
from .ABSEC import abse_atom
import copy
from . import alch_deriv_ksdft


class APDFT_perturbator(lib.StreamObject):
    @alias_param(param_name="symmetry", param_alias='symm')
    def __init__(self,mf,symmetry=None,sites=None,flag_CP_skip=False,
                 max_cycle_cphf=40, conv_tol_cphf=1e-9, flag_response_property=False,
                 charges_for_center=None):
        self.mf=mf
        mf_name = self.mf.__class__.__name__
        if mf_name not in ['RHF', "SymAdaptedRHF", 'RKS', "SymAdaptedRKS"]:
            raise ValueError(self.mf.__class__.__name__, "should be RHF or SymAdaptedRHF or RKS or  or SymAdaptedRKS")
        self.mol=mf.mol
        self.symm=symmetry
        self.sites=[]
        for site in sites: self.sites.append(site)
        self.DeltaV=DeltaV
        # max_cycle_cphf=40, conv_tol_cphf=1e-9 are default PYSCF params
        self.max_cycle_cphf=max_cycle_cphf
        self.conv_tol_cphf=conv_tol_cphf
        self.alchemy_cphf_deriv=alchemy_cphf_deriv
        self.electric_field_cphf_deriv=electric_field_cphf_deriv
        self.make_dP=make_dP
        self.make_U=make_U
        self.dVs={}
        self.mo1s={}
        self.e1s={}
        self.dPs={}
        self.afs={}
        self.mo1s_electric_field=None
        self.e1s_electric_field=None
        if not flag_CP_skip:
            self.perturb()
        self.flag_response_property = flag_response_property
        if self.flag_response_property:
            self.ao_electric_dipole = self.calc_ao_electric_dipole(charges_for_center=charges_for_center)
            self.ref_elec_electric_dipole_moment = self.calc_elec_electric_dipole_moment()
            self.ref_electric_dipole_moment=self.ref_elec_electric_dipole_moment + \
                self.calc_nuc_dipole_moment(charges_for_center=charges_for_center)
            self.perturb_electric_field()
            self.ref_elec_electric_polarizability = self.calc_elec_electric_polarizability()
        self.cubic=None
        self.hessian=None
        self.gradient=None
        self.electric_dipole_gradient=None
        self.electric_polarizability_gradient=None
        self.xcf=None
        self.flag_calc_U_R=False
        self.afs_xc=None
        try: 
            self.xcf=mf.xc
        except:pass
        # Computed in finite_difference.py
        self.fourth_order_derivatives=None
        self.fifth_order_derivatives=None

    def U(self,atm_idx):
        if atm_idx not in self.sites:
                self.sites.append(atm_idx)
                self.perturb()
        return make_U(self.mo1s[atm_idx])
    def dP(self,atm_idx):
        if atm_idx not in self.sites:
                self.sites.append(atm_idx)
                self.perturb()
        return make_dP(self.mf,self.mo1s[atm_idx])

    def dP_atom(self,atm_idx):
        if atm_idx not in self.sites:
                self.sites.append(atm_idx)
                self.perturb()
        return make_dP(self.mf,self.mo1s[atm_idx])

    def dP_pred(self,pvec):
        pvec=np.asarray(pvec)
        return self.mf.make_rdm1()+np.array([self.dP_atom(i) for i in self.sites]).transpose(1,2,0).dot(pvec)

    def perturb(self):
        for site in self.sites:
            if site in self.mo1s:
                pass
            elif  self.symm and site in self.symm.eqs:
                ref_idx=self.symm.eqs[site]['ref']
                if ref_idx in self.mo1s:
                    self.dVs[site]=DeltaV(self.mol,[[site],[1]])
                    self.mo1s[site],self.e1s[site]=self.symm.rotate_mo1e1(self.mo1s[ref_idx],self.e1s[ref_idx],\
                    site,ref_idx,self.mf.mo_coeff,self.mf.get_ovlp())
                else: continue
            else:
                self.dVs[site]=DeltaV(self.mol,[[site],[1]])
                self.mo1s[site],self.e1s[site]=alchemy_cphf_deriv(self.mf,self.dVs[site],
                                                                  max_cycle_cphf=self.max_cycle_cphf,
                                                                  conv_tol_cphf=self.conv_tol_cphf)
    def perturb_electric_field(self):
        """ Get electric-field perturbed MO coefficients and eigenvalues of CPHF/KS equations """
        if not self.flag_response_property:
            raise ValueError("flag_response_property should be True to calculate response properties")

        self.mo1s_electric_field, self.e1s_electric_field = \
            electric_field_cphf_deriv(self.mf, max_cycle_cphf=self.max_cycle_cphf,
                                      conv_tol_cphf=self.conv_tol_cphf)

    def mo1(self,atm_idx):
        if atm_idx not in self.mo1s:
            if atm_idx not in self.sites:
                self.sites.append(atm_idx)
            self.perturb()
        return self.mo1s[atm_idx]

    def dV(self,atm_idx):
        if atm_idx not in self.dVs:
            if atm_idx not in self.sites:
                self.sites.append(atm_idx)
            self.perturb()
        return self.dVs[atm_idx]

    def e1(self,atm_idx):
        if atm_idx not in self.e1s:
            if atm_idx not in self.sites:
                self.sites.append(atm_idx)
            self.perturb()
        return self.e1s[atm_idx]

    def af(self,atm_idx):
        if atm_idx in self.afs:
            return self.afs[atm_idx]
        elif self.symm and atm_idx in self.symm.eqs:
            ref_idx=self.symm.eqs[atm_idx]['ref']
            afr=self.af(ref_idx)
            self.afs[atm_idx]=self.symm.symm_gradient(afr,atm_idx,ref_idx)
        else:
            if atm_idx not in self.sites:
                self.sites.append(atm_idx)
                self.perturb()
            DZ=[0 for x in range(self.mol.natm)]
            DZ[atm_idx]=1
            af=aaff_resolv(self.mf,DZ,U=self.U(atm_idx),dP=self.dP(atm_idx),e1=self.e1(atm_idx))
            af+=alc_deriv_grad_nuc(self.mol,DZ)
            self.afs[atm_idx]=af
        return self.afs[atm_idx]

    def calc_ao_electric_dipole(self, charges_for_center=None):
        if not self.flag_response_property:
            raise ValueError("flag_response_property should be True to calculate response properties")

        charges = self.mol.atom_charges()
        coords  = self.mol.atom_coords()
        if charges_for_center is None:
            charge_center = np.einsum('i,ix->x', charges, coords) / charges.sum()
        else:
            charges_for_center = np.asarray(charges_for_center)
            charge_center = np.einsum('i,ix->x', charges_for_center, coords) / charges_for_center.sum()

        with self.mol.with_common_orig(charge_center):
            ao_dip = self.mol.intor_symmetric('int1e_r', comp=3)

        return ao_dip

    def first_deriv(self,atm_idx):
        return first_deriv_elec(self.mf,self.dV(atm_idx))+first_deriv_nuc_nuc(self.mol,[[atm_idx],[1]])
    def elec_first_deriv(self,atm_idx):
        return first_deriv_elec(self.mf,self.dV(atm_idx))
    def second_deriv(self,idx_1,idx_2):
        return second_deriv_elec(self.mf,self.dV(idx_1),self.mo1(idx_2)) +second_deriv_nuc_nuc(self.mol,[[idx_1,idx_2],[1,1]])
    def third_deriv(self,pvec):
        pvec=np.asarray(pvec)
        return np.einsum('ijk,i,j,k',self.cubic,pvec,pvec,pvec)

    def calc_elec_electric_dipole_moment(self):
        """ Calculate the electronic part of the electric dipole moment with the charge center.
            Note that in PySCF dip_moment uses (0.0, 0.0, 0.0) for the center.
        """
        if not self.flag_response_property:
            raise ValueError("flag_response_property should be True to calculate response properties")

        dm =self.mf.make_rdm1()
        el_dip = np.einsum('xij,ji->x', self.ao_electric_dipole, dm).real
        el_dip *= -1.0

        return el_dip

    def calc_elec_electric_polarizability(self):
        """ Calculate the electronic part of the electric polarizability with the charge center.
        """
        if not self.flag_response_property:
            raise ValueError("flag_response_property should be True to calculate response properties")

        mo_coeff = self.mf.mo_coeff
        mo_occ = self.mf.mo_occ
        occidx = mo_occ > 0
        orbo = mo_coeff[:, occidx]

        int_r = self.ao_electric_dipole
        int_r = np.einsum('xpq,pi,qj->xij', int_r, mo_coeff.conj(), orbo)
        # *2 for double occupancy
        pol = np.einsum('xij,yij->xy', int_r, self.mo1s_electric_field)
        pol = (pol + pol.T) * 2.0
        pol *= -1.0

        return pol

    def alch_elec_electric_dipole(self, atm_idx, perturb_electron_density="Z"):
        """ Calculate the electronic part of the alchemical electric dipole """
        if not self.flag_response_property:
            raise ValueError("flag_response_property should be True to calculate response properties")

        if perturb_electron_density not in ["Z", "F"]:
            raise ValueError("perturb_electron_density should be 'Z' or 'F'")

        if atm_idx not in self.sites:
            self.sites.append(atm_idx)
            self.perturb()

        mo_coeff = self.mf.mo_coeff
        mo_occ = self.mf.mo_occ
        occidx = mo_occ > 0
        orbo = mo_coeff[:, occidx]

        if perturb_electron_density == "Z":
            # Calculate the electric dipole moment with the derivative of electron density with respect to nuclear charges
            charges = self.mol.atom_charges()
            coords  = self.mol.atom_coords()
            charge_center = np.einsum('i,ix->x', charges, coords) / charges.sum()
            with self.mol.with_common_orig(charge_center):
                int_r = self.mol.intor_symmetric('int1e_r', comp=3)
            int_r = np.einsum('xpq,pi,qj->xij', int_r, mo_coeff.conj(), orbo)
            # *2 for double occupancy
            perturb_dipole = np.einsum('xij,ij->x', int_r, self.mo1(atm_idx))
            perturb_dipole = (perturb_dipole + perturb_dipole.T) * 2.0

        elif perturb_electron_density == "F":
            # Calculate the electric dipole moment with the derivative of electron density with respect to electric field strength
            perturb_dipole = np.einsum('pq,pi,qj->ij', self.dV(atm_idx), mo_coeff.conj(), orbo)
            perturb_dipole = np.einsum('ij,xij->x', perturb_dipole, self.mo1s_electric_field)
            # *2 for double occupancy
            perturb_dipole = (perturb_dipole + perturb_dipole.T) * 2.0

        return perturb_dipole

    def build_inefficient_alchemical_force(self, atm_idx):
        """ Modified af """

        # Calculate first-order response matrix with respect to
        # nuclear coordinates
        if not self.flag_calc_U_R:
            self.mo1s_R = np.array(alch_deriv_ksdft.calc_U_R(self.mf))
            self.dP_R = alch_deriv_ksdft.make_dP_R(self.mf,self.mo1s_R)
            self.flag_calc_U_R = True

        # if atm_idx in self.afs:
        #     return self.afs[atm_idx]
        # elif self.symm and atm_idx in self.symm.eqs:
        #     ref_idx=self.symm.eqs[atm_idx]['ref']
        #     afr=self.af(ref_idx)
        #     self.afs[atm_idx]=self.symm.symm_gradient(afr,atm_idx,ref_idx)
        # else:
        # if atm_idx not in self.afs:

        if atm_idx not in self.sites:
            self.sites.append(atm_idx)
            self.perturb()
        DZ=[0 for x in range(self.mol.natm)]
        DZ[atm_idx]=1
        # af=aaff_resolv(self.mf,DZ,U=self.U(atm_idx),dP=self.dP(atm_idx),e1=self.e1(atm_idx))
        af=inefficient_aaff_resolv(self.mf,DZ,dP=self.dP_R,dV=self.dV(atm_idx))
        af+=alc_deriv_grad_nuc(self.mol,DZ)
        self.afs[atm_idx]=af

        return self.afs[atm_idx]

    def build_alchemical_force(self, atm_idx):
        """ Modified af """

        # if atm_idx in self.afs:
        #     return self.afs[atm_idx]
        # elif self.symm and atm_idx in self.symm.eqs:
        #     ref_idx=self.symm.eqs[atm_idx]['ref']
        #     afr=self.af(ref_idx)
        #     self.afs[atm_idx]=self.symm.symm_gradient(afr,atm_idx,ref_idx)
        # else:
        # if atm_idx not in self.afs:

        if atm_idx not in self.sites:
            self.sites.append(atm_idx)
            self.perturb()
        DZ=[0 for x in range(self.mol.natm)]
        DZ[atm_idx]=1

        mf_name = self.mf.__class__.__name__
        if mf_name in ['RKS', "SymAdaptedRKS"] and self.afs_xc is None:
            # Calculate the exchange-correlation terms
            self.afs_xc = aaff_xc_resolv(self.mf,self.mo1s)

        # Calculate the alchemical force without the exchange-correlation terms
        af=aaff_resolv(self.mf,DZ,U=self.U(atm_idx),dP=self.dP(atm_idx),e1=self.e1(atm_idx))
        if mf_name in ['RKS', "SymAdaptedRKS"]:
            af+=self.afs_xc[atm_idx]
        af+=alc_deriv_grad_nuc(self.mol,DZ)
        self.afs[atm_idx]=af

        return self.afs[atm_idx]

    def build_gradient(self):
        idxs=self.sites
        self.gradient=np.asarray([self.first_deriv(x) for x in idxs])
        return self.gradient
    def build_elec_gradient(self):
        idxs=self.sites
        self.gradient=np.asarray([self.elec_first_deriv(x) for x in idxs])
        return self.gradient
    def build_hessian(self):
        mo1s=[]
        dVs=[]
        for id in self.sites:
            mo1s.append(self.mo1(id))
            dVs.append(self.dV(id))
        mo1s=np.asarray(mo1s)
        dVs=np.asarray(dVs)
        self.hessian=alch_hessian(self.mf,dVs,mo1s) +self.hessian_nuc_nuc(*self.sites)
        return self.hessian

    def build_elec_hessian(self):
        mo1s=[]
        dVs=[]
        for id in self.sites:
            mo1s.append(self.mo1(id))
            dVs.append(self.dV(id))
        mo1s=np.asarray(mo1s)
        dVs=np.asarray(dVs)
        self.hessian=alch_hessian(self.mf,dVs,mo1s)
        return self.hessian

    def hessian_nuc_nuc(self,*args):
            idxs=[]
            for arg in args:
                if isinstance(arg,int):
                    idxs.append(arg)
            hessian=np.zeros((len(idxs),len(idxs)))
            for i in range(len(idxs)):
                for j in range(i,len(idxs)):
                    hessian[i,j]=second_deriv_nuc_nuc(self.mol,[[idxs[i],idxs[j]],[1,1]])/2
            hessian+=hessian.T
            return hessian

    def build_cubic(self):
            idxs=self.sites
            mo1s=np.asarray([self.mo1(x) for x in idxs])
            dVs=np.asarray([self.dV(x) for x in idxs])
            e1s=np.asarray([self.e1(x) for x in idxs])
            self.cubic=alch_cubic(self.mf,dVs,mo1s,e1s)
            return self.cubic

    def build_all(self):
        self.build_gradient()
        self.build_hessian()
        self.build_cubic()

    def build_elec_electric_dipole_gradient(self, perturb_electron_density='Z'):
        """ Calculate the electronic part of the electric dipole moment """
        if not self.flag_response_property:
            raise ValueError("flag_response_property should be True to calculate response properties")

        idxs=self.sites
        self.electric_dipole_gradient=np.asarray(
            [self.alch_elec_electric_dipole(x, perturb_electron_density) for x in idxs])
        return self.electric_dipole_gradient

    def build_electric_polarizability_gradient(self):
        if not self.flag_response_property:
            raise ValueError("flag_response_property should be True to calculate response properties")

        idxs=self.sites
        mo1s=np.asarray([self.mo1(x) for x in idxs])
        dVs=np.asarray([self.dV(x) for x in idxs])
        e1s=np.asarray([self.e1(x) for x in idxs])
        charges = self.mol.atom_charges()
        coords  = self.mol.atom_coords()
        charge_center = np.einsum('i,ix->x', charges, coords) / charges.sum()
        with self.mol.with_common_orig(charge_center):
            dip = self.mol.intor_symmetric('int1e_r', comp=3)
        self.electric_polarizability_gradient=cubic_alch_electric_hessian(
            self.mf, dVs, dip, mo1s, self.mo1s_electric_field, e1s, self.e1s_electric_field)
        return self.electric_polarizability_gradient

    def calc_nuclei_nuclei_repulsion(self):
        charges = self.mol.atom_charges()
        coordinates  = self.mol.atom_coords()
        natoms = len(charges)
        ret = 0.0
        for i in range(natoms):
            for j in range(i + 1, natoms):
                d = np.linalg.norm(coordinates[i] - coordinates[j])
                ret += charges[i] * charges[j] / d
        return ret

    def calc_nuc_dipole_moment(self, charges=None, unit_angstrom=False, charges_for_center=None):
        """ Calculate the nuclear part of the electric dipole moment.
            Note that it is calculated at the charge center of the reference molecule.
        """
        # TODO: move this to a suitable place
        if unit_angstrom:
            angstrom_to_bohr = 1.8897261339212517
        else:
            angstrom_to_bohr = 1.0

        if charges is None:
            charges = self.mol.atom_charges()
        # else:
        #     if charges_for_center is not None:
        #         raise ValueError("charges and charges_for_center cannot be specified at the same time")
        coordinates  = self.mol.atom_coords()
        natoms = len(charges)

        # Calculate charge center
        total_charge = np.sum(charges)
        if charges_for_center is None:
            charge_center = np.einsum('i,ix->x', charges, coordinates) / total_charge
        else:
            charges_for_center = np.asarray(charges_for_center)
            charge_center = np.einsum('i,ix->x', charges_for_center, coordinates) / charges_for_center.sum()

        # Calculate nuclear dipole moment around charge center
        nuclear_dipole_moment = np.zeros(3)
        for i in range(natoms):
            nuclear_dipole_moment += charges[i] * (coordinates[i] - charge_center) * angstrom_to_bohr

        return nuclear_dipole_moment

    def APDFT1(self,pvec):
        pvec=np.asarray(pvec)
        return self.mf.e_tot+pvec.dot(self.gradient)
    def elec_APDFT1(self,pvec):
        pvec=np.asarray(pvec)
        return self.mf.e_tot-self.mf.energy_nuc()+pvec.dot(self.gradient)
    def APDFT2(self,pvec):
        pvec=np.asarray(pvec)
        return self.APDFT1(pvec)+0.5*np.einsum('i,ij,j',pvec,self.hessian,pvec)
    def elec_APDFT2(self,pvec):
        pvec=np.asarray(pvec)
        return self.elec_APDFT1(pvec)+0.5*np.einsum('i,ij,j',pvec,self.hessian,pvec)
    def APDFT3(self,pvec):
        pvec=np.asarray(pvec)
        return self.APDFT2(pvec)+1/6*np.einsum('ijk,i,j,k',self.cubic,pvec,pvec,pvec)
    def elec_APDFT3(self,pvec):
            pvec=np.asarray(pvec)
            return self.elec_APDFT2(pvec)+1/6*np.einsum('ijk,i,j,k',self.cubic,pvec,pvec,pvec)
    def elec_APDFT0(self,pvec):
        pvec=np.asarray(pvec)
        return self.mf.e_tot-self.mf.energy_nuc()

    def elec_APDFT4(self,pvec):
        pvec=np.asarray(pvec)
        return self.elec_APDFT3(pvec)+1/24*np.einsum('ijkl,i,j,k,l',
                                                     self.fourth_order_derivatives,
                                                     pvec,pvec,pvec,pvec)

    def elec_APDFT5(self,pvec):
        pvec=np.asarray(pvec)
        return self.elec_APDFT4(pvec)+1/120*np.einsum('ijklm,i,j,k,l,m',
                                                     self.fifth_order_derivatives,
                                                     pvec,pvec,pvec,pvec,pvec)

    def target_energy_ref_bs(self,pvec):  # with refernce basis set
        tmol=self.target_mol_ref_bs(pvec)
        b2mf=scf.RHF(tmol)
        return b2mf.scf(dm0=b2mf.init_guess_by_1e())

    def target_mol_ref_bs(self,pvec):
        if type(pvec) is list:
            tmol=FcM_like(self.mol,fcs=pvec)
        else:
            tmol=FcM_like(self.mol,fcs=pvec.tolist())
        return tmol

    def get_target_atom_charges(self, pvec):
        target_atom_charges = copy.deepcopy(self.mol.atom_charges())
        for idx in range(len(pvec)):
            target_atom_charges[self.sites[idx]] += int(pvec[idx])
        return target_atom_charges

    def electric_dipole_APDFT0(self,pvec):
        if not self.flag_response_property:
            raise ValueError("flag_response_property should be True to calculate response properties")

        pvec=np.asarray(pvec)
        target_atom_charges = self.get_target_atom_charges(pvec)
        # Charge center of the reference molecule is used for the center of the target molecule
        return self.ref_elec_electric_dipole_moment + self.calc_nuc_dipole_moment(charges=target_atom_charges,
                                                                                  charges_for_center=self.mol.atom_charges())

    def elec_electric_dipole_APDFT0(self,pvec):
        if not self.flag_response_property:
            raise ValueError("flag_response_property should be True to calculate response properties")

        return self.ref_elec_electric_dipole_moment

    def electric_dipole_APDFT1(self,pvec):
        if not self.flag_response_property:
            raise ValueError("flag_response_property should be True to calculate response properties")

        pvec=np.asarray(pvec)
        ref_elec_dipole = self.ref_elec_electric_dipole_moment
        # *-1.0 comes from the definition of the electric dipole moment
        dipole = ref_elec_dipole + np.einsum('ix,i->x', self.electric_dipole_gradient, pvec) * -1.0
        target_atom_charges = self.get_target_atom_charges(pvec)
        # Charge center of the reference molecule is used for the center of the target molecule
        dipole += self.calc_nuc_dipole_moment(charges=target_atom_charges, charges_for_center=self.mol.atom_charges())
        return dipole

    def elec_electric_dipole_APDFT1(self,pvec):
        if not self.flag_response_property:
            raise ValueError("flag_response_property should be True to calculate response properties")

        pvec=np.asarray(pvec)
        ref_elec_dipole = self.ref_elec_electric_dipole_moment
        # *-1.0 comes from the definition of the electric dipole moment
        elec_dipole = ref_elec_dipole + np.einsum('ix,i->x', self.electric_dipole_gradient, pvec) * -1.0
        return elec_dipole

    def elec_electric_polarizability_APDFT0(self,pvec):
        if not self.flag_response_property:
            raise ValueError("flag_response_property should be True to calculate response properties")

        return self.ref_elec_electric_polarizability

    def elec_electric_polarizability_APDFT1(self,pvec):
        if not self.flag_response_property:
            raise ValueError("flag_response_property should be True to calculate response properties")

        pvec=np.asarray(pvec)
        ref_elec_polarizability = self.ref_elec_electric_polarizability
        # *-1.0 comes from the definition of the electric polarizability
        elec_polarizability = ref_elec_polarizability + np.einsum('ixy,i->xy', self.electric_polarizability_gradient, pvec) * -1.0
        return elec_polarizability

    def relative_elec_APDFT1(self,pvec):
        pvec=np.asarray(pvec)
        return 2 * pvec.dot(self.gradient)

    def relative_elec_APDFT3(self,pvec):
        pvec=np.asarray(pvec)
        return self.relative_elec_APDFT1(pvec) + \
                   (1/3) * np.einsum('ijk,i,j,k',self.cubic,pvec,pvec,pvec)

    def relative_elec_APDFT5(self,pvec):
        pvec=np.asarray(pvec)
        return self.relative_elec_APDFT3(pvec) + \
            (1/60) * np.einsum('ijklm,i,j,k,l,m',self.fifth_order_derivatives,pvec,pvec,pvec,pvec,pvec)

    def even_elec_estimate_APDFT0(self,pvec,left_elec_energy):
        # Eq. (2) in von Lilienfeld, Domenichini, arXiv 2023
        pvec=np.asarray(pvec)
        return 2 * (self.mf.e_tot - self.mf.energy_nuc()) - left_elec_energy

    def even_elec_estimate_APDFT2(self,pvec,left_elec_energy):
        # Eq. (2) in von Lilienfeld, Domenichini, arXiv 2023
        pvec=np.asarray(pvec)
        return self.even_elec_estimate_APDFT0(pvec,left_elec_energy) + \
                   np.einsum('i,ij,j',pvec,self.hessian,pvec)

    def target_mol(self,pvec):
        splitted=(self.mol.atom.split())
        refchgs=copy.deepcopy(self.mol.atom_charges())
        for idx in range(len(pvec)):
            refchgs[self.sites[idx]]+=int(pvec[idx])
        for idx in range(len(pvec)):
            splitted[self.sites[idx]*4]=charge2symbol[refchgs[self.sites[idx]]]
        atomstr=" ".join(splitted)
        tmol=gto.M(atom=atomstr,unit=self.mol.unit,basis=self.mol.basis,charge=self.mol.charge+sum(pvec))
        return tmol

    def target_energy(self,pvec):
        tmf=scf.RHF(self.target_mol(pvec))
        return tmf.scf()

    def ap_bsec(self,pvec):
        ral=[charge2symbol[i] for i in self.mol.atom_charges()]
        tal=[charge2symbol[i] for i in self.target_mol(pvec).atom_charges()]
        if len(ral) != len(tal):
            print(ral,tal,"reference and target lengths do not match!", sys.exc_info()[0])
            raise 
        bsecorr=0
        for i in range(len(ral)):
            bsecorr+=abse_atom(ral[i],tal[i],self.mf.__class__,self.xcf, bs=self.mol.basis)
        return bsecorr

def parse_to_array(natm,dL):
    arr=np.zeros(natm)
    for i in range(len(dL[0])):
        arr[dL[0][i]]=dL[1][i]
    return arr
