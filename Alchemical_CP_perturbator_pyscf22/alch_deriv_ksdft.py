import mod_hessian_rhf

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

def make_U_R(mf):
    """ Calculate the first-order response matrix with respect to nuclear coordinates """

    mf.mf_hess = mod_hessian_rhf.Hessian(mf)
    mo1 = proc_hessian_(mf.mf_hess)
    return mo1