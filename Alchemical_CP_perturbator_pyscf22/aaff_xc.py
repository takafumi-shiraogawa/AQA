import numpy
from pyscf import lib
from pyscf.lib import logger
# from pyscf.hessian import rhf as rhf_hess
# from pyscf.grad import rks as rks_grad
# from pyscf.dft import numint
from pyscf.hessian import rks as rks_hess

def make_h1(hessobj, mo_coeff, mo_occ, chkfile=None, atmlst=None, verbose=None):
    """ Modified make_h1 in hessian/rks.py
    Only vxc and fxc terms are evaluated.
    """

    mol = hessobj.mol
    if atmlst is None:
        atmlst = range(mol.natm)

    # nao, nmo = mo_coeff.shape
    # mocc = mo_coeff[:,mo_occ>0]
    # dm0 = numpy.dot(mocc, mocc.T) * 2
    # hcore_deriv = hessobj.base.nuc_grad_method().hcore_generator(mol)

    mf = hessobj.base
    ni = mf._numint
    ni.libxc.test_deriv_order(mf.xc, 2, raise_error=True)
    # omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)
    # hybrid = ni.libxc.is_hybrid_xc(mf.xc)

    mem_now = lib.current_memory()[0]
    max_memory = max(2000, mf.max_memory*.9-mem_now)
    h1ao = rks_hess._get_vxc_deriv1(hessobj, mo_coeff, mo_occ, max_memory)
    # aoslices = mol.aoslice_by_atom()
    # for i0, ia in enumerate(atmlst):
    #     shl0, shl1, p0, p1 = aoslices[ia]
    #     shls_slice = (shl0, shl1) + (0, mol.nbas)*3
    #     if hybrid:
    #         vj1, vj2, vk1, vk2 = \
    #                 rhf_hess._get_jk(mol, 'int2e_ip1', 3, 's2kl',
    #                                  ['ji->s2kl', -dm0[:,p0:p1],  # vj1
    #                                   'lk->s1ij', -dm0         ,  # vj2
    #                                   'li->s1kj', -dm0[:,p0:p1],  # vk1
    #                                   'jk->s1il', -dm0         ], # vk2
    #                                  shls_slice=shls_slice)
    #         veff = vj1 - hyb * .5 * vk1
    #         veff[:,p0:p1] += vj2 - hyb * .5 * vk2
    #         if omega != 0:
    #             with mol.with_range_coulomb(omega):
    #                 vk1, vk2 = \
    #                     rhf_hess._get_jk(mol, 'int2e_ip1', 3, 's2kl',
    #                                      ['li->s1kj', -dm0[:,p0:p1],  # vk1
    #                                       'jk->s1il', -dm0         ], # vk2
    #                                      shls_slice=shls_slice)
    #             veff -= (alpha-hyb) * .5 * vk1
    #             veff[:,p0:p1] -= (alpha-hyb) * .5 * vk2
    #     else:
    #         vj1, vj2 = rhf_hess._get_jk(mol, 'int2e_ip1', 3, 's2kl',
    #                                     ['ji->s2kl', -dm0[:,p0:p1],  # vj1
    #                                      'lk->s1ij', -dm0         ], # vj2
    #                                     shls_slice=shls_slice)
    #         veff = vj1
    #         veff[:,p0:p1] += vj2

    #     h1ao[ia] += veff + veff.transpose(0,2,1)
    #     h1ao[ia] += hcore_deriv(ia)

    if chkfile is None:
        return h1ao
    else:
        for ia in atmlst:
            lib.chkfile.save(chkfile, 'scf_f1ao/%d'%ia, h1ao[ia])
        return chkfile


def hess_elec(hessobj, mo_energy=None, mo_coeff=None, mo_occ=None,
              mo1=None, mo_e1=None, h1ao=None,
              atmlst=None, max_memory=4000, verbose=None):
    """ Modified make_h1 in hessian/rhf.py
    Only (vxc + fxc) * X * del_R X * P terms are evaluated.

    args
        mo1: The rotated molecular orbitals with the dimension of (num_AO, num_MO)
             for nuclear charges of all atoms.
    """
    log = logger.new_logger(hessobj, verbose)
    time0 = t1 = (logger.process_clock(), logger.perf_counter())

    mol = hessobj.mol
    mf = hessobj.base
    if mo_energy is None: mo_energy = mf.mo_energy
    if mo_occ is None:    mo_occ = mf.mo_occ
    if mo_coeff is None:  mo_coeff = mf.mo_coeff
    if atmlst is None: atmlst = range(mol.natm)

    # de2 = hessobj.partial_hess_elec(mo_energy, mo_coeff, mo_occ, atmlst,
    #                                 max_memory, log)

    de2 = numpy.zeros((mol.natm, mol.natm, 3))

    if h1ao is None:
        h1ao = hessobj.make_h1(mo_coeff, mo_occ, hessobj.chkfile, atmlst, log)
        # t1 = log.timer_debug1('making H1', *time0)
    # if mo1 is None or mo_e1 is None:
    #     mo1, mo_e1 = hessobj.solve_mo1(mo_energy, mo_coeff, mo_occ, h1ao,
    #                                    None, atmlst, max_memory, log)
    #     t1 = log.timer_debug1('solving MO1', *t1)

    if isinstance(h1ao, str):
        h1ao = lib.chkfile.load(h1ao, 'scf_f1ao')
        h1ao = dict([(int(k), h1ao[k]) for k in h1ao])
    # if isinstance(mo1, str):
    #     mo1 = lib.chkfile.load(mo1, 'scf_mo1')
    #     mo1 = dict([(int(k), mo1[k]) for k in mo1])

    # nao, nmo = mo_coeff.shape
    mocc = mo_coeff[:,mo_occ>0]
    # s1a = -mol.intor('int1e_ipovlp', comp=3)

    # aoslices = mol.aoslice_by_atom()
    for i0, ia in enumerate(atmlst):
        # shl0, shl1, p0, p1 = aoslices[ia]
        # s1ao = numpy.zeros((3,nao,nao))
        # s1ao[:,p0:p1] += s1a[:,p0:p1]
        # s1ao[:,:,p0:p1] += s1a[:,p0:p1].transpose(0,2,1)
        # s1oo = numpy.einsum('xpq,pi,qj->xij', s1ao, mocc, mocc)

        # for j0 in range(i0+1):
        #     ja = atmlst[j0]
        for j0, ja in enumerate(atmlst):
            # q0, q1 = aoslices[ja][2:]
# *2 for double occupancy, *2 for +c.c.
            # dm1 = numpy.einsum('ypi,qi->ypq', mo1[ja], mocc)
            dm1 = numpy.einsum('pi,qi->pq', mo1[ia], mocc)
            # de2[i0,j0] += numpy.einsum('xpq,ypq->xy', h1ao[ia], dm1) * 4
            de2[i0,j0] += numpy.einsum('xpq,pq->x', h1ao[ja], dm1) * 4
            # dm1 = numpy.einsum('ypi,qi,i->ypq', mo1[ja], mocc, mo_energy[mo_occ>0])
            # de2[i0,j0] -= numpy.einsum('xpq,ypq->xy', s1ao, dm1) * 4
            # de2[i0,j0] -= numpy.einsum('xpq,ypq->xy', s1oo, mo_e1[ja]) * 2

        # for j0 in range(i0):
        #     de2[j0,i0] = de2[i0,j0].T

    # log.timer('RHF hessian', *time0)
    return de2


def kernel(self, mo_energy=None, mo_coeff=None, mo_occ=None, atmlst=None):
        if mo_energy is None: mo_energy = self.base.mo_energy
        if mo_coeff is None: mo_coeff = self.base.mo_coeff
        if mo_occ is None: mo_occ = self.base.mo_occ
        if atmlst is None:
            atmlst = self.atmlst
        else:
            self.atmlst = atmlst

        # de = self.hess_elec(mo_energy, mo_coeff, mo_occ, atmlst=atmlst)
        self.de = self.hess_elec(mo_energy, mo_coeff, mo_occ, mo1=self.mo1_z,
                                 atmlst=atmlst)
        # self.de = de + self.hess_nuc(self.mol, atmlst=atmlst)
        return self.de


class Hessian(rks_hess.Hessian):
    '''Non-relativistic RKS hessian'''
    def __init__(self, mf, mo1_z):
        rks_hess.Hessian.__init__(self, mf)
        self.mo1_z = mo1_z
        self.de = numpy.zeros((0,0,3))  # (A,B,dR_B)
        self._keys = set(self.__dict__.keys())

    hess_elec = hess_elec
    make_h1 = make_h1
    kernel = kernel