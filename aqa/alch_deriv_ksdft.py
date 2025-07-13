import numpy as np
from . import mod_hessian_rhf

def proc_hessian_(mf_hess, mo_energy=None, mo_coeff=None, mo_occ=None, h1ao_grad=None):
    """
    Process hessian evaluation for alchemical force
    Modified proc_hessian_ in pyscf/prop/infrared/rhf
    """
    mf = mf_hess.base

    mo_energy = mo_energy if mo_energy else mf.mo_energy
    mo_coeff = mo_coeff if mo_coeff else mf.mo_coeff
    mo_occ = mo_occ if mo_occ else mf.mo_occ
    h1ao_grad = h1ao_grad if h1ao_grad else mf_hess.make_h1(mo_coeff, mo_occ)

    # moao1_grad, mo_e1_grad = mf_hess.solve_mo1(mo_energy, mo_coeff, mo_occ, h1ao_grad)
    mo1, mo_e1_grad = mf_hess.get_response_matrix(mo_energy, mo_coeff, mo_occ, h1ao_grad)

    # hess_elec = mf_hess.hess_elec(mo1=moao1_grad, mo_e1=mo_e1_grad, h1ao=h1ao_grad)
    # hess_nuc = mf_hess.hess_nuc()
    # mf_hess.de = hess_elec + hess_nuc
    # mo1_grad = lib.einsum("up, uv, Axvi -> Axpi", mo_coeff, mf.get_ovlp(), moao1_grad)
    # return mf_hess, h1ao_grad, mo1_grad, mo_e1_grad

    return mo1

def calc_U_R(mf):
    """ Calculate the first-order response matrix with respect to nuclear coordinates """

    mf.mf_hess = mod_hessian_rhf.Hessian(mf)
    mo1 = proc_hessian_(mf.mf_hess)
    return mo1

def make_dP_R(mf,mo1_R):
    mol=mf.mol
    nao=mol.nao
    nocc=mf.mol.nelec[0]
    C=mf.mo_coeff
    # dP=np.zeros_like(C)
    num_atom = np.shape(mo1_R)[0]
    dP = np.zeros((num_atom, 3, nao, nao))
    for idx_atom in range(num_atom):
        for idx_ncoord in range(3):
            dP[idx_atom,idx_ncoord,:,:]=2*np.einsum(
                'ij,jk,lk->il',C,mo1_R[idx_atom,idx_ncoord,:,:],C[:,:nocc])
            dP[idx_atom,idx_ncoord]=dP[idx_atom,idx_ncoord]+dP[idx_atom,idx_ncoord].T
    return dP

# def make_U_R(mo1_R):
#     # mo1_R's shape is (3, num_MO, num_occ_MO)
#     U_R=np.zeros((3, mo1_R.shape[1],mo1_R.shape[1]))
#     # inefficient alchemical force does not need the occupied-virtual block
#     for i in range(3):
#         U_R[i,:,:mo1_R.shape[2]]=mo1_R[i]
#         U_R[i]=U_R[i]-U_R[i].T
#     return U_R