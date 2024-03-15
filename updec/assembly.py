import jax
import jax.numpy as jnp
from jax.tree_util import Partial

from functools import cache, partial

from updec.config import DIM
from updec.utils import make_nodal_rbf, make_monomial, compute_nb_monomials, make_all_monomials
from updec.cloud import Cloud


def assemble_Phi(cloud:Cloud, rbf:callable=None):
    """ Assemble the matrix Phi (see equation 5) from Shahane """
    ## TODO: Make this matrix sparse. Only consider local supports
    ## rbf could be a string instead

    N, Ni, Nd = cloud.N, cloud.Ni, cloud.Nd
    Phi = jnp.zeros((N, N))
    # nodal_rbf = jax.jit(partial(make_nodal_rbf, rbf=rbf))         
    # nodal_rbf = Partial(make_nodal_rbf, rbf=rbf)                    ## TODO Use the prexisting nodal_rbf func
    rbf_vec = jax.vmap(rbf, in_axes=(None, 0), out_axes=0)
    grad_rbf = jax.grad(rbf)
    grad_rbf_vec = jax.vmap(grad_rbf, in_axes=(None, 0), out_axes=0)

    nodes = cloud.sorted_nodes

    # for i in range(N):      ## TODO only from 0 to Ni+Nd
    # # for i in range(Ni+Nd):
    #     # for j in cloud.local_supports[i]:
    #     #     Phi = Phi.at[i, j].set(nodal_rbf(cloud.nodes[i], cloud.nodes[j]))

    #     support_ids = jnp.array(cloud.local_supports[i])
    #     Phi = Phi.at[i, support_ids].set(rbf_vec(nodes[i], nodes[support_ids]))

    def rbf_body_func(i, Phi):
        support_ids = cloud.sorted_local_supports[i]
        return Phi.at[i, support_ids].set(rbf_vec(nodes[i], nodes[support_ids]))

    Phi = jax.lax.fori_loop(0, N, rbf_body_func, Phi)


    # for i in range(Ni+Nd, N):
    #     assert cloud.node_types[i] in ["n"], "not a neumann boundary node"    ## Internal nod

    #     support_n = jnp.array(cloud.local_supports[i])
    #     grads = jnp.nan_to_num(grad_rbf_vec(nodes[i], nodes[support_n]), neginf=0., posinf=0.)
    #     Phi = Phi.at[i, support_n].set(jnp.dot(grads, cloud.outward_normals[i]))

    # print("Finiteness Phi:", jnp.all(jnp.isfinite(Phi)))
    # print("Last column Phi all zero?", jnp.allclose(Phi[:,-1], 0.))
    # print("matrix Phi:\n", Phi)     ## Indicates singularity
    # print("Determinant of matrix Phi:", jnp.linalg.det(Phi))     ## Indicates singularity

    return Phi


def assemble_P(cloud:Cloud, nb_monomials:int):
    """ See (6) from Shanane """
    N = cloud.N
    M = nb_monomials
    P = jnp.zeros((N, M))
    nodes = cloud.sorted_nodes

    for j in range(M):
        # monomial = jax.jit(Partial(make_monomial, id=j))      ## IS
        monomial = Partial(make_monomial, id=j)
        monomial_vec = jax.vmap(monomial, in_axes=(0,), out_axes=0)
        P = P.at[:, j].set(monomial_vec(nodes))

        # for i in range(N):
        #     P = P.at[i, j].set(monomial(cloud.nodes[i]))
    ###  print("Finiteness P:", jnp.all(jnp.isfinite(P)))

    # def monomial_body_func(j, P):
    #     monomial = Partial(make_monomial, id=j)
    #     monomial_vec = jax.vmap(monomial, in_axes=(0,), out_axes=0)
    #     return P.at[:, j].set(monomial_vec(nodes))
    # P = jax.lax.fori_loop(0, M, monomial_body_func, P)

    return P


# @cache        ## TODO Make caching work with jax.jit ! Should be OK with jitting the whole PDE solver
def assemble_A(cloud, rbf, nb_monomials=2):
    """ Assemble matrix A, see (4) from Shanane """

    Phi = assemble_Phi(cloud, rbf)
    P = assemble_P(cloud, nb_monomials)

    N, M = Phi.shape[1], P.shape[1]

    A = jnp.zeros((N+M, N+M))
    A = A.at[:N, :N].set(Phi)
    A = A.at[:N, N:].set(P)
    A = A.at[N:, :N].set(P.T)

    return A

# @cache          ## Turn this into assemble and LU decompose
def assemble_invert_A(cloud, rbf, nb_monomials):
    A = assemble_A(cloud, rbf, nb_monomials)
    ### print("Determinant of matrix A:", jnp.linalg.det(A))     ## Indicates singularity
    ### print("Invert:", jnp.linalg.inv(A))
    return jnp.linalg.inv(A)


def assemble_op_Phi_P(operator:callable, cloud:Cloud, rbf:callable, nb_monomials:int, args:list):
    """ Assembles upper op(Phi): the collocation matrix to which a differential operator is applied """
    ## Only the internal nodes (M, N)

    # operator = jax.jit(operator, static_argnums=2)

    N = cloud.N
    Ni = cloud.Ni
    M = nb_monomials
    opPhi = jnp.zeros((Ni, N))
    opP = jnp.zeros((Ni, M))

    nodes = cloud.sorted_nodes
    # if len(args) > 0:
    #     # fields = jnp.stack(args, axis=-1)
    #     fields = jnp.stack(args, axis=-1)
    # else:
    #     fields = jnp.ones((N,1))     ## TODO Won't be used tho. FIx this !
    fields = jnp.stack(args, axis=-1) if args else jnp.ones((N,1))      ## TODO Find a better way. Will never be used

    # operator_rbf = partial(operator, monomial=None)
    # @jax.jit
    def operator_rbf(x, center=None, args=None):
        return operator(x, center, rbf, None, args)
    operator_rbf_vec = jax.jit(jax.vmap(operator_rbf, in_axes=(None, 0, None), out_axes=0))

    # operator_mon = Partial(operator, node=None)
    def operator_mon(x, args=None, monomial=None):
        return operator(x, None, rbf, monomial, args)
    monomials = make_all_monomials(M)

    # coords = cloud.sorted_nodes
    internal_ids = jnp.arange(Ni)

    # for i in range(Ni):
    #     assert cloud.node_types[i] == "i", "not an internal node"    ## Internal node

    #     support_ids = jnp.array(cloud.local_supports[i])
    #     opPhi = opPhi.at[i, support_ids].set(operator_rbf_vec(nodes[i], nodes[support_ids], fields[i]))

    def rbf_body_func(i, opPhi):
        # assert cloud.node_types[i] == "i", "not an internal node"    ## Internal node
        support_ids = cloud.sorted_local_supports[i]
        return opPhi.at[i, support_ids].set(operator_rbf_vec(nodes[i], nodes[support_ids], fields[i]))

    opPhi = jax.lax.fori_loop(0, Ni, rbf_body_func, opPhi)

    # opPhi = jax.lax.fori_loop(0, Ni, lambda i, Phi: rbf_body_func(i, opPhi, cloud, nodes, fields, operator_rbf_vec), opPhi)


    for j in range(M):
        operator_mon_func = Partial(operator_mon, monomial=monomials[j])
        operator_mon_vec = jax.vmap(operator_mon_func, in_axes=(0, 0), out_axes=0)
        opP = opP.at[internal_ids, j].set(operator_mon_vec(nodes[internal_ids], fields[internal_ids]))

    # def mon_body_func(j, opP):
    #     operator_mon_func = Partial(operator_mon, monomial=monomials[j])
    #     operator_mon_vec = jax.vmap(operator_mon_func, in_axes=(0, 0), out_axes=0)
    #     opP = opP.at[internal_ids, j].set(operator_mon_vec(nodes[internal_ids], fields[internal_ids]))

    # opP = jax.lax.fori_loop(0, M, mon_body_func, opP)

    ### print("Finiteness op Phi and P:", jnp.all(jnp.isfinite(opPhi)), jnp.all(jnp.isfinite(opP)))
    return opPhi, opP



def assemble_bd_Phi_P(cloud:Cloud, rbf:callable, nb_monomials:int, robin_coeffs:dict=None):

    """ Assembles upper op(Phi): the collocation matrix to which a differential operator is applied """
    ## Only the internal nodes (M, N)

    # operator = jax.jit(operator, static_argnums=2)

    N, Ni = cloud.N, cloud.Ni
    Nd, Nn, Nr = cloud.Nd, cloud.Nn, cloud.Nr
    Np = cloud.Np
    M = nb_monomials
    bdPhi = jnp.zeros((Nd+Nn+Nr+sum(Np), N))
    bdP = jnp.zeros((Nd+Nn+Nr+sum(Np), M))
    # print("SHapes:", Nd, Nn, Nr, sum(Np), bdPhi.shape, bdP.shape)

    # rbf = Partial(make_rbf, rbf=rbf)                    ## TODO JIT THIS, and Use the prexisting rbf func
    grad_rbf = jax.grad(rbf)
    # grad_rbf = jax.jit(jax.grad(nodal_rbf))

    rbf_vec = jax.vmap(rbf, in_axes=(None, 0), out_axes=0)
    grad_rbf_vec = jax.vmap(grad_rbf, in_axes=(None, 0), out_axes=0)

    nodes = cloud.sorted_nodes


    # ## Fill Matrix Phi with vectorisation from axis=1 ###

    # for i in range(Ni, N):
    #     ii = i-Ni        ## Actual index in the matrices
    #     assert cloud.node_types[i] in ["d", "n", "r"], "not a boundary node"    ## Internal node

    #     support = jnp.array(cloud.local_supports[i])
    #     vals = rbf_vec(nodes[i], nodes[support])
    #     grads = jnp.nan_to_num(grad_rbf_vec(nodes[i], nodes[support]), neginf=0., posinf=0.)

    #     if cloud.node_types[i] == "d":
    #         bdPhi = bdPhi.at[ii, support].set(vals)

    #     elif cloud.node_types[i] == "n":    ## Neumann node
    #         bdPhi = bdPhi.at[ii, support].set(jnp.dot(grads, cloud.outward_normals[i]))

    #     elif cloud.node_types[i] == "r":    ## Robin node
    #         betas_js = robin_coeffs[i]*jnp.ones(support.shape[0])
    #         bdPhi = bdPhi.at[ii, support].set(betas_js*vals + jnp.dot(grads, cloud.outward_normals[i]))



    def bdPhi_d_body_func(i, bdPhi):
        # assert cloud.node_types[i] in ["d", "n", "r"], "not a boundary node"    ## Internal node

        support = cloud.sorted_local_supports[i]
        vals = rbf_vec(nodes[i], nodes[support])

        return bdPhi.at[i-Ni, support].set(vals)

    bdPhi = jax.lax.fori_loop(Ni, Ni+Nd, bdPhi_d_body_func, bdPhi)

    def bdPhi_n_body_func(i, bdPhi):
        # assert cloud.node_types[i] in ["d", "n", "r"], "not a boundary node"    ## Internal node

        support = cloud.sorted_local_supports[i]
        grads = jnp.nan_to_num(grad_rbf_vec(nodes[i], nodes[support]), neginf=0., posinf=0.)

        if hasattr(cloud, "sorted_outward_normals"):
            normals = cloud.sorted_outward_normals[i-Ni-Nd]
        else:
            normals = jnp.zeros((DIM,))

        ## PROBLEM: index i must be shifted when outward normals becomes an array. Also, what if no outward normal are defined?
        return bdPhi.at[i-Ni, support].set(jnp.dot(grads, normals))

    bdPhi = jax.lax.fori_loop(Ni+Nd, Ni+Nd+Nn, bdPhi_n_body_func, bdPhi)

    ## TODO: SOrt thisfirst
    sorted_robin_coeffs = cloud.sort_dict_by_keys(robin_coeffs) if len(robin_coeffs) > 0 else None   ## For JIT. Since their ids are contiguous by construction. TODO write a full test for this

    def bdPhi_r_body_func(i, bdPhi):
        # assert cloud.node_types[i] in ["d", "n", "r"], "not a boundary node"    ## Internal node

        support = cloud.sorted_local_supports[i]
        vals = rbf_vec(nodes[i], nodes[support])
        grads = jnp.nan_to_num(grad_rbf_vec(nodes[i], nodes[support]), neginf=0., posinf=0.)

        if sorted_robin_coeffs is not None:       ## Compile time const
            betas_js = sorted_robin_coeffs[i-Ni-Nd-Nn]*jnp.ones(support.shape[0])
        else:
            betas_js = jnp.zeros(support.shape[0])

        if hasattr(cloud, "sorted_outward_normals"):
            normals = cloud.sorted_outward_normals[i-Ni-Nd-Nn]
        else:
            normals = jnp.zeros((DIM,)) ## This brach shoudl never be taken TODO

        return bdPhi.at[i-Ni, support].set(betas_js*vals + jnp.dot(grads, normals))

    bdPhi = jax.lax.fori_loop(Ni+Nd+Nn, Ni+Nd+Nn+Nr, bdPhi_r_body_func, bdPhi)


    # print("BdPhi:", bdPhi)

    def bdPhi_pd_body_func(i, vals):
        bdPhi, nb_conds, jump_points = vals

        support1 = cloud.sorted_local_supports[i]
        support2 = cloud.sorted_local_supports[i+nb_conds]
        support = jnp.concatenate((support1, support2))

        vals1 = rbf_vec(nodes[i], nodes[support])
        vals2 = rbf_vec(nodes[i+nb_conds], nodes[support])
        # jax.debug.print("Nodes 1 and 2: \n {} {} \n", nodes[i], nodes[i+nb_conds])

        return bdPhi.at[i-Ni-jump_points, support].set(vals1-vals2), nb_conds, jump_points

    Np_ = Ni+Nd+Nn+Nr
    jump_points = 0
    for nb_p_points in Np:
        nb_conds = nb_p_points//2
        bdPhi, _, _ = jax.lax.fori_loop(Np_, Np_+nb_conds, bdPhi_pd_body_func, (bdPhi, nb_conds, jump_points))
        Np_ += nb_p_points
        jump_points += nb_conds

    # print("BdPhi:", bdPhi)



    def bdPhi_pn_body_func(i, vals):
        bdPhi, nb_conds, jump_points = vals

        support1 = cloud.sorted_local_supports[i]
        support2 = cloud.sorted_local_supports[i+nb_conds]
        support = jnp.concatenate((support1, support2))

        grads1 = jnp.nan_to_num(grad_rbf_vec(nodes[i], nodes[support]), neginf=0., posinf=0.)
        grads2 = jnp.nan_to_num(grad_rbf_vec(nodes[i+nb_conds], nodes[support]), neginf=0., posinf=0.)

        # jax.debug.print("Nodes 1 and 2: \n {} {} \n", nodes[i+jump_normals-sum(Np)//2], nodes[i+nb_conds+jump_normals-sum(Np)//2])

        # grads1 = grad_rbf_vec(nodes[i], nodes[support])
        # grads2 = grad_rbf_vec(nodes[i+nb_conds], nodes[support])

        if hasattr(cloud, "sorted_outward_normals"):
            normals1 = cloud.sorted_outward_normals[i-Ni-Nd]
            normals2 = cloud.sorted_outward_normals[i-Ni-Nd+nb_conds]
            # normals1 = cloud.outward_normals[i]
            # normals2 = cloud.outward_normals[i+nb_conds]
            # jax.debug.print("Normals 1 and 2: {} {} \n", normals1, normals2)
        else:
            normals1 = jnp.zeros((DIM,))
            normals2 = jnp.zeros((DIM,))

        # print("Shapes before dot:", grads1.shape, normals1.shape, grads2.shape, normals2.shape)
        diff_grads = jnp.dot(grads1, normals1) - jnp.dot(grads2, -normals2)
        # jax.debug.print("Dots 1 and 2: {}\n {} \n", i-Ni-jump_points+sum(Np)//2, jnp.dot(grads1, normals1)-jnp.dot(grads2, -normals2))

        # print("SHupport shape", support.shape)
        # jax.debug.print("Support: \n {} \n", support)

        # diff_grads = jnp.zeros_like(jnp.dot(grads1, normals1))
        # diff_grads = diff_grads / 10.
        # diff_grads = jnp.clip(diff_grads, -2., 2.)
        # jax.debug.print("Current positions to fill: {} \n {} \n", i-Ni-jump_normals+sum(Np)//2)
        return bdPhi.at[i-Ni-jump_points+sum(Np)//2, support].set(diff_grads), nb_conds, jump_points


    Np_ = Ni+Nd+Nn+Nr
    jump_points = 0
    for nb_p_points in Np:
        nb_conds = nb_p_points//2
        bdPhi, _, _ = jax.lax.fori_loop(Np_, Np_+nb_conds, bdPhi_pn_body_func, (bdPhi, nb_conds, jump_points))
        Np_ += nb_p_points
        jump_points += nb_conds
    # print("Final value of Np_:", Np_)
        # print("BdPhi:", bdPhi)



    # def bdPhi_pn_body_func(i, vals):
    #     bdPhi, nb_conds, jump_normals = vals

    #     support1 = cloud.sorted_local_supports[i+jump_normals-sum(Np)//2]
    #     support2 = cloud.sorted_local_supports[i+nb_conds+jump_normals-sum(Np)//2]
    #     support = jnp.concatenate((support1, support2))

    #     grads1 = jnp.nan_to_num(grad_rbf_vec(nodes[i+jump_normals-sum(Np)//2], nodes[support]), neginf=0., posinf=0.)
    #     grads2 = jnp.nan_to_num(grad_rbf_vec(nodes[i+nb_conds+jump_normals-sum(Np)//2], nodes[support]), neginf=0., posinf=0.)

    #     # jax.debug.print("Nodes 1 and 2: \n {} {} \n", nodes[i+jump_normals-sum(Np)//2], nodes[i+nb_conds+jump_normals-sum(Np)//2])

    #     # grads1 = grad_rbf_vec(nodes[i], nodes[support])
    #     # grads2 = grad_rbf_vec(nodes[i+nb_conds], nodes[support])

    #     if hasattr(cloud, "sorted_outward_normals"):
    #         normals1 = cloud.sorted_outward_normals[i-Ni-Nd+jump_normals-sum(Np)//2]
    #         normals2 = cloud.sorted_outward_normals[i-Ni-Nd+jump_normals-sum(Np)//2+nb_conds]
    #         # normals1 = cloud.outward_normals[i]
    #         # normals2 = cloud.outward_normals[i+nb_conds]
    #     else:
    #         normals1 = jnp.zeros((DIM,))
    #         normals2 = jnp.zeros((DIM,))

    #     diff_grads = jnp.dot(grads1, normals1) - jnp.dot(grads2, -normals2)
    #     # diff_grads = jnp.zeros_like(jnp.dot(grads1, normals1))
    #     # diff_grads = diff_grads / 10.
    #     # diff_grads = jnp.clip(diff_grads, -2., 2.)
    #     return bdPhi.at[i-Ni, support].set(diff_grads), nb_conds, jump_normals

    # Np_ = Ni+Nd+Nn+Nr + sum(Np)//2
    # jump_normals = 0
    # for nb_p_points in Np:
    #     nb_conds = nb_p_points//2
    #     bdPhi, _, _ = jax.lax.fori_loop(Np_, Np_+nb_conds, bdPhi_pn_body_func, (bdPhi, nb_conds, jump_normals))
    #     Np_ += nb_conds
    #     jump_normals += nb_p_points
    # # print("Final value of Np_:", Np_)
    # # print("BdPhi:", bdPhi)







    ### Fill Matrix P with vectorisation from axis=0 ###
    node_ids_d = [k for k,v in cloud.node_types.items() if v == "d"]
    node_ids_n = [k for k,v in cloud.node_types.items() if v == "n"]
    node_ids_r = [k for k,v in cloud.node_types.items() if v == "r"]
    betas_is = [robin_coeffs[k] for k,v in cloud.node_types.items() if v == "r"]

    # node_ids_p = [[ k for k,v in cloud.node_types.items() if v[0] == "p" ] for _ in range(len(Np))]

    monomials = make_all_monomials(M)
    if len(node_ids_d) > 0:
        node_ids_d = jnp.array(node_ids_d)
        for j in range(M):
            monomial_vec = jax.vmap(monomials[j], in_axes=(0,), out_axes=0)
            bdP = bdP.at[node_ids_d-Ni, j].set(monomial_vec(nodes[node_ids_d]))

        # def dirichlet_body_func(j, bdP):
        #     monomial_vec = jax.vmap(monomials[j], in_axes=(0,), out_axes=0)
        #     return bdP.at[node_ids_d-Ni, j].set(monomial_vec(nodes[node_ids_d]))
        # bdP = jax.lax.fori_loop(0, M, dirichlet_body_func, bdP)

    if len(node_ids_n) > 0:
        normals_n = jnp.stack([cloud.outward_normals[i] for i in node_ids_n], axis=0)
        node_ids_n = jnp.array(node_ids_n)

        dot_vec = jax.vmap(jnp.dot, in_axes=(0,0), out_axes=0)
        for j in range(M):
            grad_monomial = jax.grad(monomials[j])
            grad_monomial_vec = jax.vmap(grad_monomial, in_axes=(0,), out_axes=0)
            grads = grad_monomial_vec(nodes[node_ids_n])
            bdP = bdP.at[node_ids_n-Ni, j].set(dot_vec(grads, normals_n))

        # def neumann_body_func(j, bdP):
        #     grad_monomial = jax.grad(monomials[j])
        #     grad_monomial_vec = jax.vmap(grad_monomial, in_axes=(0,), out_axes=0)
        #     grads = grad_monomial_vec(nodes[node_ids_n])
        #     return bdP.at[node_ids_n-Ni, j].set(dot_vec(grads, normals_n))
        # bdP = jax.lax.fori_loop(0, M, neumann_body_func, bdP)

    if len(node_ids_r) > 0:
        normals_r = jnp.stack([cloud.outward_normals[i] for i in node_ids_r], axis=0)
        node_ids_r = jnp.array(node_ids_r)

        dot_vec = jax.vmap(jnp.dot, in_axes=(0,0), out_axes=0)
        for j in range(M):
            monomial_vec = jax.vmap(monomials[j], in_axes=(0,), out_axes=0)
            vals = monomial_vec(nodes[node_ids_r])

            grad_monomial = jax.grad(monomials[j])
            grad_monomial_vec = jax.vmap(grad_monomial, in_axes=(0,), out_axes=0)
            grads = grad_monomial_vec(nodes[node_ids_r])

            bdP = bdP.at[node_ids_r-Ni, j].set(jnp.array(betas_is)*vals + dot_vec(grads, normals_r))

        # def robin_body_func(j, bdP):
        #     monomial_vec = jax.vmap(monomials[j], in_axes=(0,), out_axes=0)
        #     vals = monomial_vec(nodes[node_ids_r])

        #     grad_monomial = jax.grad(monomials[j])
        #     grad_monomial_vec = jax.vmap(grad_monomial, in_axes=(0,), out_axes=0)
        #     grads = grad_monomial_vec(nodes[node_ids_r])

        #     return bdP.at[node_ids_r-Ni, j].set(jnp.array(betas_is)*vals + dot_vec(grads, normals_r))
        # bdP = jax.lax.fori_loop(0, M, robin_body_func, bdP)

    ### print("Finiteness bd Phi and P:", jnp.all(jnp.isfinite(bdPhi)), jnp.all(jnp.isfinite(bdP)))

    # print("BdP:\n", bdP)

    # print("All boundary nodes:", nodes[16:, :])






    # if len(Np) > 0:
    #     Np_ = Ni+Nd+Nn+Nr
    #     for nb_p_points in Np:

    #         # node_ids_pd1 = [k for k,v in cloud.node_types.items() if v[:-1] == n_type]

    #         nb_conds = nb_p_points//2
    #         node_ids_pd1 = jnp.arange(Np_, Np_+nb_conds)
    #         node_ids_pd2 = jnp.arange(Np_+nb_conds, Np_+nb_p_points)

    #         # print("These are the nodes 1:", nodes[node_ids_pd1])
    #         # print("These are the nodes 2:", nodes[node_ids_pd2])

    #         for j in range(M):
    #             monomial_vec = jax.vmap(monomials[j], in_axes=(0,), out_axes=0)
    #             diff = monomial_vec(nodes[node_ids_pd1]) - monomial_vec(nodes[node_ids_pd2])
    #             bdP = bdP.at[node_ids_pd1-Ni, j].set(diff)
    #             # print("BdP:\n", bdP)

    #         Np_ += nb_p_points

    #     jax.debug.print("BdP: \n {} \n", bdP)

    #     half_Np = sum(Np)//2
    #     Np_ = Ni+Nd+Nn+Nr + half_Np
    #     for nb_p_points in Np:
    #         nb_conds = nb_p_points//2
    #         node_ids_pn1 = range(Np_-half_Np, Np_+nb_conds-half_Np)
    #         node_ids_pn2 = range(Np_+nb_conds-half_Np, Np_+nb_p_points-half_Np)

    #         # print("These are the nodes 1:", nodes[jnp.array(node_ids_pn1)])
    #         # print("These are the nodes 2:", nodes[jnp.array(node_ids_pn2)])

    #         normals_pn1 = jnp.stack([cloud.outward_normals[i] for i in node_ids_pn1], axis=0)
    #         normals_pn2 = jnp.stack([cloud.outward_normals[i] for i in node_ids_pn2], axis=0)

    #         node_ids_pn1 = jnp.array(node_ids_pn1)
    #         node_ids_pn2 = jnp.array(node_ids_pn2)

    #         dot_vec = jax.vmap(jnp.dot, in_axes=(0,0), out_axes=0)
    #         for j in range(M):
    #             grad_monomial = jax.grad(monomials[j])
    #             grad_monomial_vec = jax.vmap(grad_monomial, in_axes=(0,), out_axes=0)
    #             grads1 = grad_monomial_vec(nodes[node_ids_pn1])
    #             grads2 = grad_monomial_vec(nodes[node_ids_pn2])
    #             diff_grads = dot_vec(grads1, normals_pn1) - dot_vec(grads2, -normals_pn2)
    #             bdP = bdP.at[node_ids_pn1-Ni+nb_conds, j].set(diff_grads)

    #         Np_ += nb_p_points
    # # print("BdP:\n", bdP)

    # # print("Numper of nodes of each type:", Nd, Nn, Nr, Np)
    # jax.debug.print("BdP: \n {} \n", bdP)






    if len(Np) > 0:
        jump_points = Ni+Nd+Nn+Nr
        Np_ = Ni+Nd+Nn+Nr
        for nb_p_points in Np:

            # node_ids_pd1 = [k for k,v in cloud.node_types.items() if v[:-1] == n_type]

            nb_conds = nb_p_points//2
            node_ids_pd1 = jnp.arange(jump_points, jump_points+nb_conds)
            node_ids_pd2 = jnp.arange(jump_points+nb_conds, jump_points+nb_p_points)

            # print("These are the nodes 1:", nodes[node_ids_pd1])
            # print("These are the nodes 2:", nodes[node_ids_pd2])

            for j in range(M):
                monomial_vec = jax.vmap(monomials[j], in_axes=(0,), out_axes=0)
                diff = monomial_vec(nodes[node_ids_pd1]) - monomial_vec(nodes[node_ids_pd2])
                bdP = bdP.at[jnp.arange(Np_,Np_+nb_conds)-Ni, j].set(diff)
                # print("BdP:\n", bdP)

            Np_ += nb_conds
            jump_points += nb_p_points

        # jax.debug.print("BdP: \n {} \n", bdP)

        half_Np = sum(Np)//2
        jump_points = Ni+Nd+Nn+Nr
        Np_ = Ni+Nd+Nn+Nr
        for nb_p_points in Np:
            nb_conds = nb_p_points//2
            node_ids_pn1 = range(jump_points, jump_points+nb_conds)
            node_ids_pn2 = range(jump_points+nb_conds, jump_points+nb_p_points)

            # print("These are the nodes 1:", nodes[jnp.array(node_ids_pn1)])
            # print("These are the nodes 2:", nodes[jnp.array(node_ids_pn2)])

            normals_pn1 = jnp.stack([cloud.outward_normals[i] for i in node_ids_pn1], axis=0)
            normals_pn2 = jnp.stack([cloud.outward_normals[i] for i in node_ids_pn2], axis=0)

            node_ids_pn1 = jnp.array(node_ids_pn1)
            node_ids_pn2 = jnp.array(node_ids_pn2)

            dot_vec = jax.vmap(jnp.dot, in_axes=(0,0), out_axes=0)
            for j in range(M):
                grad_monomial = jax.grad(monomials[j])
                grad_monomial_vec = jax.vmap(grad_monomial, in_axes=(0,), out_axes=0)
                grads1 = grad_monomial_vec(nodes[node_ids_pn1])
                grads2 = grad_monomial_vec(nodes[node_ids_pn2])
                diff_grads = dot_vec(grads1, normals_pn1) - dot_vec(grads2, -normals_pn2)
                bdP = bdP.at[jnp.arange(Np_,Np_+nb_conds)-Ni+half_Np, j].set(diff_grads)

            Np_ += nb_conds
            jump_points += nb_p_points
    # print("BdP:\n", bdP)

    # print("Numper of nodes of each type:", Nd, Nn, Nr, Np)
    # jax.debug.print("BdP again: \n {} \n", bdP)






    return bdPhi, bdP



def assemble_B(operator:callable, cloud:Cloud, rbf:callable, nb_monomials:int, diff_args:list, robin_coeffs:dict):
    """ Assemble B using opPhi, P, and A """

    N, Ni = cloud.N, cloud.Ni
    # M = compute_nb_monomials(max_degree, 2)
    M = nb_monomials

    # Phi, P = assemble_Phi(cloud, rbf), assemble_P(cloud, M)
    # rbf = Partial(make_rbf, rbf=rbf)

    ## Compute coefficients here

    opPhi, opP = assemble_op_Phi_P(operator, cloud, rbf, M, diff_args)
    bdPhi, bdP = assemble_bd_Phi_P(cloud, rbf, M, robin_coeffs)

    full_opPhi = jnp.zeros((N, N))
    full_opP = jnp.zeros((N, M))

    full_opPhi = full_opPhi.at[:Ni, :].set(opPhi[:, :])
    full_opP = full_opP.at[:Ni, :].set(opP[:, :])

    # print("Shapes:", Ni, full_opPhi.shape, bdPhi.shape)
    full_opPhi = full_opPhi.at[Ni:, :].set(bdPhi[:, :])
    full_opP = full_opP.at[Ni:, :].set(bdP[:, :])

    diffMat = jnp.concatenate((full_opPhi, full_opP), axis=-1)

    # A = assemble_A(cloud, nodal_rbf, M)       ## TODO make this work for nodal_rbf
    # A = assemble_A(cloud, rbf, M)

    inv_A = assemble_invert_A(cloud, rbf, M)
    B = diffMat @ inv_A

    return B[:, :N]


def new_compute_coefficients(field:jnp.ndarray, cloud:Cloud, rbf:callable, nb_monomials:int):
    """ Find nodal and polynomial coefficients for scalar field """ 

    rhs = jnp.concatenate((field, jnp.zeros((nb_monomials))))
    inv_A = assemble_invert_A(cloud, rbf, nb_monomials)

    return inv_A@rhs



def assemble_q(operator:callable, boundary_conditions:dict, cloud:Cloud, rbf:callable, nb_monomials:int, rhs_args:list):
    """ Assemble the right hand side q using the operator """
    ### Boundary conditions should match all the types of boundaries

    N = cloud.N
    Ni = cloud.Ni
    M = nb_monomials
    q = jnp.zeros((N,))

    ## Compute coefficients here
    if rhs_args != None:
        fields_coeffs = [new_compute_coefficients(field, cloud, rbf, M) for field in rhs_args]
        fields_coeffs = jnp.stack(fields_coeffs, axis=-1)
    else:
        fields_coeffs = None

    ## Internal node
    operator_vec = jax.vmap(operator, in_axes=(0, None, None, None), out_axes=(0))
    nodes = cloud.sorted_nodes
    internal_ids = jnp.arange(Ni)
    q = q.at[internal_ids].set(operator_vec(nodes[internal_ids], nodes, rbf, fields_coeffs))


    ## Facet nodes
    for f_id in cloud.facet_types.keys():
        assert f_id in boundary_conditions.keys(), "facets and boundary functions don't match ids"

        bd_op = boundary_conditions[f_id]
        bd_node_ids = jnp.array(cloud.facet_nodes[f_id])

        if callable(bd_op):      ## Is a (jitted) function 
            bd_op_vec = jax.vmap(bd_op, in_axes=(0,), out_axes=0)
            q = q.at[bd_node_ids].set(bd_op_vec(nodes[bd_node_ids]))
        else:                   ## Must be a jax array then
            q = q.at[bd_node_ids].set(bd_op)

    # keys = list(cloud.facet_types.keys())
    # def rhs_body_func(i, q):
    #     f_id = keys[i]
    #     assert f_id in boundary_conditions.keys(), "facets and boundary functions don't match ids"

    #     bd_op = boundary_conditions[f_id]
    #     bd_node_ids = jnp.array(cloud.facet_nodes[f_id])

    #     if callable(bd_op):      ## Is a (jitted) function 
    #         bd_op_vec = jax.vmap(bd_op, in_axes=(0,), out_axes=0)
    #         q = q.at[bd_node_ids].set(bd_op_vec(nodes[bd_node_ids]))
    #     else:                   ## Must be a jax array then
    #         q = q.at[bd_node_ids].set(bd_op)

    #     return q
    # q = jax.lax.fori_loop(0, len(keys), rhs_body_func, q)

    return q
