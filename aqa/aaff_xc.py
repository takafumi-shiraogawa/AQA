import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.hessian import rhf as rhf_hess
from pyscf.hessian import rks as rks_hess
from pyscf.dft import numint

from pyscf.grad import rks

def make_h1(hessobj, mo_coeff, mo_occ, chkfile=None, atmlst=None, verbose=None):
    """ Modified make_h1 in hessian/rks.py
    Only vxc and fxc terms are evaluated.
    """

    mol = hessobj.mol
    if atmlst is None:
        atmlst = range(mol.natm)

    nao, nmo = mo_coeff.shape
    mocc = mo_coeff[:,mo_occ>0]
    dm0 = numpy.dot(mocc, mocc.T) * 2
    # hcore_deriv = hessobj.base.nuc_grad_method().hcore_generator(mol)

    mf = hessobj.base
    ni = mf._numint
    ni.libxc.test_deriv_order(mf.xc, 2, raise_error=True)
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)
    hybrid = ni.libxc.is_hybrid_xc(mf.xc)

    mem_now = lib.current_memory()[0]
    max_memory = max(2000, mf.max_memory*.9-mem_now)
    h1ao = rks_hess._get_vxc_deriv1(hessobj, mo_coeff, mo_occ, max_memory)
    aoslices = mol.aoslice_by_atom()
    for i0, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = aoslices[ia]
        shls_slice = (shl0, shl1) + (0, mol.nbas)*3
        if hybrid:
            vj1, vj2, vk1, vk2 = \
                    rhf_hess._get_jk(mol, 'int2e_ip1', 3, 's2kl',
                                     ['ji->s2kl', -dm0[:,p0:p1],  # vj1
                                      'lk->s1ij', -dm0         ,  # vj2
                                      'li->s1kj', -dm0[:,p0:p1],  # vk1
                                      'jk->s1il', -dm0         ], # vk2
                                     shls_slice=shls_slice)
            veff = vj1 - hyb * .5 * vk1
            veff[:,p0:p1] += vj2 - hyb * .5 * vk2
            if omega != 0:
                with mol.with_range_coulomb(omega):
                    vk1, vk2 = \
                        rhf_hess._get_jk(mol, 'int2e_ip1', 3, 's2kl',
                                         ['li->s1kj', -dm0[:,p0:p1],  # vk1
                                          'jk->s1il', -dm0         ], # vk2
                                         shls_slice=shls_slice)
                veff -= (alpha-hyb) * .5 * vk1
                veff[:,p0:p1] -= (alpha-hyb) * .5 * vk2
        else:
            vj1, vj2 = rhf_hess._get_jk(mol, 'int2e_ip1', 3, 's2kl',
                                        ['ji->s2kl', -dm0[:,p0:p1],  # vj1
                                         'lk->s1ij', -dm0         ], # vj2
                                        shls_slice=shls_slice)
            veff = vj1
            veff[:,p0:p1] += vj2
        h1ao[ia] += veff + veff.transpose(0,2,1)
        # h1ao[ia] += hcore_deriv(ia)

    # if chkfile is None:
    #     return h1ao
    # else:
    #     for ia in atmlst:
    #         lib.chkfile.save(chkfile, 'scf_f1ao/%d'%ia, h1ao[ia])
    #     return chkfile
    return h1ao


# TODO: use the specified atoms in atmlst
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

    # TODO: len(atmlst) should be the number of atoms to be considered.
    de2 = numpy.zeros((mol.natm, mol.natm, 3))

    if h1ao is None:
        h1ao = hessobj.make_h1(mo_coeff, mo_occ, hessobj.chkfile, atmlst, log)
        # t1 = log.timer_debug1('making H1', *time0)
    # if mo1 is None or mo_e1 is None:
    #     mo1, mo_e1 = hessobj.solve_mo1(mo_energy, mo_coeff, mo_occ, h1ao,
    #                                    None, atmlst, max_memory, log)
    #     t1 = log.timer_debug1('solving MO1', *t1)

    # if isinstance(h1ao, str):
    #     h1ao = lib.chkfile.load(h1ao, 'scf_f1ao')
    #     h1ao = dict([(int(k), h1ao[k]) for k in h1ao])
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
            dm1 = numpy.einsum('ij,jk,lk->il', mo_coeff, mo1[ia], mocc)
            # de2[i0,j0] += numpy.einsum('xpq,ypq->xy', h1ao[ia], dm1) * 4
            de2[i0,j0] += numpy.einsum('xpq,pq->x', h1ao[ja], dm1) * 4
            # dm1 = numpy.einsum('ypi,qi,i->ypq', mo1[ja], mocc, mo_energy[mo_occ>0])
            # de2[i0,j0] -= numpy.einsum('xpq,ypq->xy', s1ao, dm1) * 4
            # de2[i0,j0] -= numpy.einsum('xpq,ypq->xy', s1oo, mo_e1[ja]) * 2

        # for j0 in range(i0):
        #     de2[j0,i0] = de2[i0,j0].T

    # log.timer('RHF hessian', *time0)
    return de2


# Modified pyscf.hessian.rks._get_vxc_deriv1
# TODO: _get_kxc should be moved into a new file since aaff implies alchemical force.
#       Otherwise, aaff_xc.py should be renamed.
def _get_kxc(hessobj, mo_coeff, mo_occ, mo1s, max_memory):
    # Note that here mo1s is dC = UC.
    # mo1s: (natom, nao, noccmo)

    mol = hessobj.mol
    mf = hessobj.base
    if hessobj.grids is not None:
        grids = hessobj.grids
    else:
        grids = mf.grids
    if grids.coords is None:
        grids.build(with_non0tab=True)

    nao, nmo = mo_coeff.shape
    ni = mf._numint
    xctype = ni._xc_type(mf.xc)

    aoslices = mol.aoslice_by_atom()
    # (natom, start-shell-id, stop-shell-id, start-AO-id, stop-AO-id)

    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()

    # dm0 = mf.make_rdm1(mo_coeff, mo_occ)
    dm1 = numpy.zeros((mo1s.shape[0], nao, nao))

    nocc=mf.mol.nelec[0]
    # dm1: (natom, nao, nao)
    dm1 = numpy.einsum('xik,lk->xil',mo1s, mo_coeff[:,:nocc])
    dm1 = dm1 + dm1.transpose(0,2,1)

    ntargetatm = mo1s.shape[0]
    vmat = numpy.zeros((ntargetatm,ntargetatm,nao,nao))
    max_memory = max(2000, max_memory-vmat.size*8/1e6)

    # Note: the structure is similar to function nr_rks of numint.py
    if xctype == 'LDA':
        ao_deriv = 0

        # block_loop defines a macro to loop over grids by blocks.
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            # weight: (ngrid)
            # rho: (ngrids)
            rho = ni.eval_rho2(mol, ao, mo_coeff, mo_occ, mask, xctype)
            kxc = ni.eval_xc_eff(mf.xc, rho, deriv=3, xctype=xctype)[3]

            # wk: (ngrid)
            wk = weight * kxc[0, 0, 0]

            for ia in range(ntargetatm):
                # ao_dm1_ia: (ngrid, nao)
                ao_dm1_ia = numint._dot_ao_dm(mol, ao, dm1[ia], mask, shls_slice, ao_loc)

                for aos_ia in range(mol.natm):
                    p0, p1 = aoslices[aos_ia][2:]

                    # AO * AO * DM
                    # p: grids
                    # i: AO index
                    # ao_dm0: (ngrids, nao)
                    # rho1_ia: (ngrids)
                    rho1_ia = numpy.einsum('pi,pi->p', ao[:,p0:p1], ao_dm1_ia[:,p0:p1])

                    for jb in range(ntargetatm):
                        ao_dm1_jb = numint._dot_ao_dm(mol, ao, dm1[jb], mask, shls_slice, ao_loc)

                        for aos_jb in range(mol.natm):
                            q0, q1 = aoslices[aos_jb][2:]

                            # rho1_jb: (ngrids)
                            rho1_jb = numpy.einsum('pi,pi->p', ao[:,q0:q1], ao_dm1_jb[:,q0:q1])

                            # w * kxc * (AO * AO * DM)
                            # wk: (ngrids)
                            # wv: (ngrids)
                            # wv = wk * rho1 * rho2
                            wv = wk * rho1_ia * rho1_jb

                            # aow: (ngrids, nao)
                            aow = numint._scale_ao(ao, wv)

                            # ao: (ngrids, nao)
                            # aow: (ngrids, nao)
                            shls_slice = (0, mol.nbas)
                            vmat[ia,jb] += numint._dot_ao_ao(mol, aow, ao, mask, shls_slice, ao_loc)

            aow = None
            ao_dm1_ia = None
            ao_dm1_jb = None

    elif xctype == 'GGA':
        ao_deriv = 1
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            # ao: (4, ngrid, nao)
            # weight: ngrid

            rho = ni.eval_rho2(mol, ao[:4], mo_coeff, mo_occ, mask, xctype)

            # kxc: (4, 4, 4, ngrid)
            kxc = ni.eval_xc_eff(mf.xc, rho, 3, xctype=xctype)[3]

            # wk: (4, 4, 4, ngrid)
            wk = weight * kxc

            for ia in range(ntargetatm):
                # ao_dm1_ia: (4, ngrid, nao)
                ao_dm1_ia = [numint._dot_ao_dm(mol, ao[i], dm1[ia], mask, shls_slice, ao_loc)
                            for i in range(4)]
                ao_dm1_ia = numpy.array(ao_dm1_ia)

                for aos_ia in range(mol.natm):
                    p0, p1 = aoslices[aos_ia][2:]

                    # ao: (4, ngrid, nao)
                    # ao_dm1_ia: (4, ngrid, nao)
                    # rho1_ia: (4, ngrid)
                    rho1_ia = numpy.einsum('gi,xgi->xg', ao[0, :, p0:p1], ao_dm1_ia[:, :, p0:p1])
                    rho1_ia[1:4] *= 2

                    # wk: (4, 4, 4, ngrid)
                    # rho1_ia: (4, ngrid)
                    # wv_ia: (4, 4, ngrid)
                    wv_ia = numpy.einsum('xg,xyzg->yzg', rho1_ia, wk)

                    for jb in range(ntargetatm):
                        # ao_dm1_ia: (4, ngrid, nao)
                        ao_dm1_jb = [numint._dot_ao_dm(mol, ao[i], dm1[jb], mask, shls_slice, ao_loc)
                                    for i in range(4)]
                        ao_dm1_jb = numpy.array(ao_dm1_jb)

                        for aos_jb in range(mol.natm):
                            q0, q1 = aoslices[aos_jb][2:]

                            rho1_jb = numpy.einsum('gi,ygi->yg', ao[0, :, q0:q1], ao_dm1_jb[:, :, q0:q1])
                            rho1_jb[1:4] *= 2

                            # wk: (4, 4, 4, ngrid)
                            # rho1_ia, rho1_jb: (, ngrid)
                            # wv: (4, ngrid)
                            wv = numpy.einsum('yg,yzg->zg', rho1_jb, wv_ia)
                            wv[0] *= .5

                            # aow: (ngrid, nao)
                            aow = numint._scale_ao(ao, wv)
                            shls_slice = (0, mol.nbas)
                            vmat[ia,jb] += numint._dot_ao_ao(mol, aow, ao[0], mask, shls_slice, ao_loc)

            ao_dm1_ia = None
            ao_dm1_jb = None
            aow = None

        vmat[:,:] = vmat[:,:] + vmat[:,:].transpose(0,1,3,2)

    elif xctype == 'MGGA':
        raise NotImplementedError('MGGA is not implemented in 3rd derivatives of XC energy')
    #     if grids.level < 5:
    #         logger.warn(mol, 'MGGA Hessian is sensitive to dft grids.')
    #     ao_deriv = 2
    #     for ao, mask, weight, coords \
    #             in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
    #         rho = ni.eval_rho2(mol, ao[:10], mo_coeff, mo_occ, mask, xctype)
    #         vxc, fxc = ni.eval_xc_eff(mf.xc, rho, 2, xctype=xctype)[1:3]
    #         wv = weight * vxc
    #         wv[0] *= .5
    #         wv[4] *= .5  # for the factor 1/2 in tau
    #         rks_grad._gga_grad_sum_(v_ip, mol, ao, wv, mask, ao_loc)
    #         rks_grad._tau_grad_dot_(v_ip, mol, ao, wv[4], mask, ao_loc, True)

    #         ao_dm0 = [numint._dot_ao_dm(mol, ao[i], dm0, mask, shls_slice, ao_loc) for i in range(4)]
    #         wf = weight * fxc
    #         for ia in range(mol.natm):
    #             dR_rho1 = _make_dR_rho1(ao, ao_dm0, ia, aoslices, xctype)
    #             wv = numpy.einsum('xyg,sxg->syg', wf, dR_rho1)
    #             wv[:,0] *= .5
    #             wv[:,4] *= .25
    #             aow = [numint._scale_ao(ao[:4], wv[i,:4]) for i in range(3)]
    #             rks_grad._d1_dot_(vmat[ia], mol, aow, ao[0], mask, ao_loc, True)

    #             for j in range(1, 4):
    #                 aow = [numint._scale_ao(ao[j], wv[i,4]) for i in range(3)]
    #                 rks_grad._d1_dot_(vmat[ia], mol, aow, ao[j], mask, ao_loc, True)
    #         ao_dm0 = aow = None

    return vmat


def make_h1_kxc(hessobj, mo_coeff, mo_occ, mo1s, chkfile=None, atmlst=None, verbose=None):
    """ Modified make_h1 in hessian/rks.py """
    nao, nmo = mo_coeff.shape
    mocc = mo_coeff[:,mo_occ>0]

    mf = hessobj.base

    mem_now = lib.current_memory()[0]
    max_memory = max(2000, mf.max_memory*.9-mem_now)
    h1ao = _get_kxc(hessobj, mo_coeff, mo_occ, mo1s, max_memory)

    return h1ao


def hess_elec_kxc(hessobj, mo_energy=None, mo_coeff=None, mo_occ=None,
              mo1s=None, mo_e1=None, h1ao=None,
              atmlst=None, max_memory=4000, verbose=None):
    """ Modified make_h1 in hessian/rhf.py """
    # mo1s: (natom, nmo, noccmo)
    log = logger.new_logger(hessobj, verbose)
    time0 = t1 = (logger.process_clock(), logger.perf_counter())

    mol = hessobj.mol
    mf = hessobj.base
    if mo_energy is None: mo_energy = mf.mo_energy
    if mo_occ is None:    mo_occ = mf.mo_occ
    if mo_coeff is None:  mo_coeff = mf.mo_coeff
    if atmlst is None: atmlst = range(mol.natm)

    ntargetatm = mo1s.shape[0]
    de3 = numpy.zeros((ntargetatm, ntargetatm, ntargetatm))

    if h1ao is None:
        h1ao = hessobj.make_h1_kxc(mo_coeff, mo_occ, mo1s, hessobj.chkfile, atmlst, log)

    mocc = mo_coeff[:,mo_occ>0]

    # atmlst is not a true atom list.
    atmlst = range(ntargetatm)

    for i0, ia in enumerate(atmlst):
        # for j0, ja in enumerate(atmlst):
        for j0 in range(i0+1):
            ja = atmlst[j0]
            # for k0, ka in enumerate(atmlst):
            for k0 in range(j0+1):
                ka = atmlst[k0]
                dm1 = numpy.einsum('ik,lk->il', mo1s[ia], mocc)
                dm1 = dm1 + dm1.T
                de3[i0, j0, k0] = numpy.einsum('pq,pq->', h1ao[ja,ka], dm1) * 8
                de3[i0, k0, j0] = de3[i0, j0, k0]
                de3[j0, k0, i0] = de3[i0, j0, k0]
                de3[j0, i0, k0] = de3[i0, j0, k0]
                de3[k0, i0, j0] = de3[i0, j0, k0]
                de3[k0, j0, i0] = de3[i0, j0, k0]

    # for i0, ia in enumerate(atmlst):
    #     for j0, ja in enumerate(atmlst):
    #         for k0, ka in enumerate(atmlst):
    #             dm1 = numpy.einsum('ik,lk->il', mo1s[ia], mocc)
    #             dm1 = dm1 + dm1.T
    #             de3[i0, j0, k0] = numpy.einsum('pq,pq->', h1ao[ja,ka], dm1) * 8

    # log.timer('RHF hessian', *time0)
    return de3

def hess_elec_kxc_electric_field(hessobj, mo_energy=None, mo_coeff=None, mo_occ=None,
              mo1s=None, mo1s_electric_field=None, mo_e1=None, mo_e1_electric_field=None,
              h1ao=None, h1ao_electric_field=None,
              atmlst=None, max_memory=4000, verbose=None):
    """ Modified make_h1 in hessian/rhf.py.
        This function is used for electric field perturbation: d3Exc / dZ dF dF """

    # mo1s: (natom, nmo, noccmo)
    # mo1s_electric_field: (3, nmo, noccmo)

    log = logger.new_logger(hessobj, verbose)
    time0 = t1 = (logger.process_clock(), logger.perf_counter())

    mol = hessobj.mol
    mf = hessobj.base
    if mo_energy is None: mo_energy = mf.mo_energy
    if mo_occ is None:    mo_occ = mf.mo_occ
    if mo_coeff is None:  mo_coeff = mf.mo_coeff
    if atmlst is None: atmlst = range(mol.natm)

    ntargetatm = mo1s.shape[0]
    de3 = numpy.zeros((ntargetatm, 3, 3))

    if h1ao_electric_field is None:
        h1ao_electric_field = hessobj.make_h1_kxc(mo_coeff, mo_occ, mo1s_electric_field,
                                                  hessobj.chkfile, atmlst, log)

    mocc = mo_coeff[:,mo_occ>0]

    # atmlst is not a true atom list.
    atmlst = range(ntargetatm)

    for i0, ia in enumerate(atmlst):
        for ja in range(3):
            for ka in range(ja+1):
                j0 = ja
                k0 = ka
                # DM for the perturbed nuclear charge
                dm1 = numpy.einsum('ik,lk->il', mo1s[ia], mocc)
                dm1 = dm1 + dm1.T

                # Z, F, F
                de3[i0, j0, k0] = numpy.einsum('pq,pq->', h1ao_electric_field[ja,ka], dm1) * 8
                de3[i0, k0, j0] = de3[i0, j0, k0]
                # de3[j0, k0, i0] = de3[i0, j0, k0]
                # de3[j0, i0, k0] = de3[i0, j0, k0]
                # de3[k0, i0, j0] = de3[i0, j0, k0]
                # de3[k0, j0, i0] = de3[i0, j0, k0]

    return de3


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


def kernel_kxc(self, mo_energy=None, mo_coeff=None, mo_occ=None, atmlst=None):
    if mo_energy is None: mo_energy = self.base.mo_energy
    if mo_coeff is None: mo_coeff = self.base.mo_coeff
    if mo_occ is None: mo_occ = self.base.mo_occ
    if atmlst is None:
        atmlst = self.atmlst
    else:
        self.atmlst = atmlst

    self.de_kxc = self.hess_elec_kxc(mo_energy, mo_coeff, mo_occ, mo1s=self.mo1_z,
                                        atmlst=atmlst)
    return self.de_kxc


def kernel_kxc_electric_field(self, mo_energy=None, mo_coeff=None, mo_occ=None, atmlst=None):
    if mo_energy is None: mo_energy = self.base.mo_energy
    if mo_coeff is None: mo_coeff = self.base.mo_coeff
    if mo_occ is None: mo_occ = self.base.mo_occ
    if atmlst is None:
        atmlst = self.atmlst
    else:
        self.atmlst = atmlst

    self.de_kxc_electric_field = self.hess_elec_kxc_electric_field(mo_energy, mo_coeff, mo_occ, mo1s=self.mo1_z,
                                                                    mo1s_electric_field=self.mo1_electric_field,
                                                                    atmlst=atmlst)
    return self.de_kxc_electric_field


class Hessian(rks_hess.Hessian):
    '''Non-relativistic RKS hessian'''
    def __init__(self, mf, mo1_z, mo1_electric_field=None):
        rks_hess.Hessian.__init__(self, mf)
        self.mo1_z = mo1_z
        self.mo1_electric_field = mo1_electric_field
        self.de = numpy.zeros((0,0,3))  # (A,B,dR_B)
        self.de_kxc = numpy.zeros((0,0,0))
        self.de_kxc_electric_field = numpy.zeros((0,0,0))
        self._keys = set(self.__dict__.keys())

    hess_elec = hess_elec
    make_h1 = make_h1
    kernel = kernel

    hess_elec_kxc = hess_elec_kxc
    hess_elec_kxc_electric_field = hess_elec_kxc_electric_field
    make_h1_kxc = make_h1_kxc
    kernel_kxc = kernel_kxc
    kernel_kxc_electric_field = kernel_kxc_electric_field
