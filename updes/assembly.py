import jax
import jax.numpy as jnp
from jax.tree_util import Partial

from functools import cache, partial

from updes.config import DIM
from updes.utils import make_nodal_rbf, make_monomial, compute_nb_monomials, make_all_monomials
from updes.cloud import Cloud


def assemble_Phi(cloud:Cloud, rbf:callable):
    """ Assemble the collocation matrix Phi (see equation 5) from Shahane et al.

    Args:
        cloud (Cloud): The points to use, along with all required information
        rbf (callable, optional): The radial basis function to use

    Returns:
        Float[Array, "nb_nodes, nb_nodes"]: The collocation matrix Phi
    """
    ## TODO: Make this matrix sparse. Only consider local supports

    N, Ni, Nd = cloud.N, cloud.Ni, cloud.Nd
    Phi = jnp.zeros((N, N))
    rbf_vec = jax.vmap(rbf, in_axes=(None, 0), out_axes=0)
    grad_rbf = jax.grad(rbf)
    grad_rbf_vec = jax.vmap(grad_rbf, in_axes=(None, 0), out_axes=0)

    nodes = cloud.sorted_nodes

    def rbf_body_func(i, Phi):
        support_ids = cloud.sorted_local_supports[i]
        return Phi.at[i, support_ids].set(rbf_vec(nodes[i], nodes[support_ids]))

    Phi = jax.lax.fori_loop(0, N, rbf_body_func, Phi)

    return Phi


def assemble_P(cloud:Cloud, nb_monomials:int):
    """Assemble the polynomial matrix P (see equation 6 from Shahane et al.)

    Args:
        cloud (Cloud): The cloud of points to use, along with all required information
        nb_monomials (int): The number of monomials to use, with increasing degree

    Returns:
        Float[Array, "nb_nodes, nb_monomials"]: The matrix P
    """
    N = cloud.N
    M = nb_monomials
    P = jnp.zeros((N, M))
    nodes = cloud.sorted_nodes

    for j in range(M):
        monomial = Partial(make_monomial, id=j)
        monomial_vec = jax.vmap(monomial, in_axes=(0,), out_axes=0)
        P = P.at[:, j].set(monomial_vec(nodes))

    return P


def assemble_A(cloud, rbf, nb_monomials=2):
    """Assemble the matrix A (see equation 4 from Shahane et al.)

    Args:
        cloud (Cloud): The cloud of points to use, along with all required information
        rbf (callable, optional): The radial basis function to use
        nb_monomials (int): The number of monomials to use, with increasing degree. Defaults to 2.

    Returns:
        Float[Array, "nb_nodes+nb_monomials, nb_nodes+nb_monomials"]: The matrix A
    """
    ## TODO Make caching work with jax.jit ! Should be OK with jitting the whole PDE solver

    Phi = assemble_Phi(cloud, rbf)
    P = assemble_P(cloud, nb_monomials)

    N, M = Phi.shape[1], P.shape[1]

    A = jnp.zeros((N+M, N+M))
    A = A.at[:N, :N].set(Phi)
    A = A.at[:N, N:].set(P)
    A = A.at[N:, :N].set(P.T)

    return A

def assemble_invert_A(cloud, rbf, nb_monomials):
    """ Assembles the inverts of the matrix A """
    A = assemble_A(cloud, rbf, nb_monomials)
    return jnp.linalg.inv(A)


def assemble_op_Phi_P(operator:callable, cloud:Cloud, rbf:callable, nb_monomials:int, args:list):
    """Assembles op(Phi) and op(P), the collocation and polynomial matrices after a differential operator is applied to internal nodes

    Args:
        operator (callable): the differential operator to apply
        cloud (Cloud): the cloud of points to use
        rbf (callable): the radial basis function to use
        nb_monomials (int): the number of monomials to use
        args (list): the fields to use in the aplication of the operator. Can be either the field values themselves or their coefficients

    Returns:
        tuple(Float[Array, "nb_internal_nodes nb_nodes"], Float[Array, "nb_internal_nodes nb_monomials"]): The internal operator matrices
    """

    N = cloud.N
    Ni = cloud.Ni
    M = nb_monomials
    opPhi = jnp.zeros((Ni, N))
    opP = jnp.zeros((Ni, M))

    nodes = cloud.sorted_nodes
    fields = jnp.stack(args, axis=-1) if args else jnp.ones((N,1))      ## TODO Find a better way rather than ones (although should never be used in practice)

    def operator_rbf(x, center=None, args=None):
        return operator(x, center, rbf, None, args)
    operator_rbf_vec = jax.jit(jax.vmap(operator_rbf, in_axes=(None, 0, None), out_axes=0))

    def operator_mon(x, args=None, monomial=None):
        return operator(x, None, rbf, monomial, args)
    monomials = make_all_monomials(M)

    internal_ids = jnp.arange(Ni)

    def rbf_body_func(i, opPhi):
        support_ids = cloud.sorted_local_supports[i]
        return opPhi.at[i, support_ids].set(operator_rbf_vec(nodes[i], nodes[support_ids], fields[i]))

    opPhi = jax.lax.fori_loop(0, Ni, rbf_body_func, opPhi)

    for j in range(M):
        operator_mon_func = Partial(operator_mon, monomial=monomials[j])
        operator_mon_vec = jax.vmap(operator_mon_func, in_axes=(0, 0), out_axes=0)
        opP = opP.at[internal_ids, j].set(operator_mon_vec(nodes[internal_ids], fields[internal_ids]))

    return opPhi, opP



def assemble_bd_Phi_P(cloud:Cloud, rbf:callable, nb_monomials:int, robin_coeffs:dict=None):
    """Assembles bd(Phi) and bd(P), the collocation and polynomial matrices after boundary conditions are applied to boundary nodes

    Args:
        cloud (Cloud): the cloud of points to use
        rbf (callable): the radial basis function to use
        nb_monomials (int): the number of monomials to use
        robin_coeffs (dict): the coefficients for the Robin boundary conditions if needed

    Returns:
        tuple(Float[Array, "nb_boundary_nodes nb_nodes"], Float[Array, "nb_boundary_nodes nb_monomials"]): The boundary operator matrices
    """
    N, Ni = cloud.N, cloud.Ni
    Nd, Nn, Nr = cloud.Nd, cloud.Nn, cloud.Nr
    Np = cloud.Np
    M = nb_monomials
    bdPhi = jnp.zeros((Nd+Nn+Nr+sum(Np), N))
    bdP = jnp.zeros((Nd+Nn+Nr+sum(Np), M))

    grad_rbf = jax.grad(rbf)

    rbf_vec = jax.vmap(rbf, in_axes=(None, 0), out_axes=0)
    grad_rbf_vec = jax.vmap(grad_rbf, in_axes=(None, 0), out_axes=0)

    nodes = cloud.sorted_nodes



    def bdPhi_d_body_func(i, bdPhi):
        """Utility function for JAX's fori_loop on Dirichlet nodes"""
        support = cloud.sorted_local_supports[i]
        vals = rbf_vec(nodes[i], nodes[support])

        return bdPhi.at[i-Ni, support].set(vals)

    bdPhi = jax.lax.fori_loop(Ni, Ni+Nd, bdPhi_d_body_func, bdPhi)

    def bdPhi_n_body_func(i, bdPhi):
        """Utility function for JAX's fori_loop on Neumann nodes"""
        support = cloud.sorted_local_supports[i]
        grads = jnp.nan_to_num(grad_rbf_vec(nodes[i], nodes[support]), neginf=0., posinf=0.)

        if hasattr(cloud, "sorted_outward_normals"):
            normals = cloud.sorted_outward_normals[i-Ni-Nd]
        else:
            normals = jnp.zeros((DIM,))     ## TODO: This branch should never be taken

        return bdPhi.at[i-Ni, support].set(jnp.dot(grads, normals))

    bdPhi = jax.lax.fori_loop(Ni+Nd, Ni+Nd+Nn, bdPhi_n_body_func, bdPhi)

    sorted_robin_coeffs = cloud.sort_dict_by_keys(robin_coeffs) if len(robin_coeffs) > 0 else None   ## For JIT purposes. This is possible since their ids are contiguous by construction. TODO: write a full test for this !

    def bdPhi_r_body_func(i, bdPhi):
        """Utility function for JAX's fori_loop on Robin nodes"""
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


    def bdPhi_pd_body_func(i, vals):
        """Utility function for JAX's fori_loop on Periodic nodes with Dirichlet conditions"""
        bdPhi, nb_conds, jump_points = vals

        support1 = cloud.sorted_local_supports[i]
        support2 = cloud.sorted_local_supports[i+nb_conds]
        support = jnp.concatenate((support1, support2))

        vals1 = rbf_vec(nodes[i], nodes[support])
        vals2 = rbf_vec(nodes[i+nb_conds], nodes[support])

        return bdPhi.at[i-Ni-jump_points, support].set(vals1-vals2), nb_conds, jump_points

    Np_ = Ni+Nd+Nn+Nr
    jump_points = 0
    for nb_p_points in Np:
        nb_conds = nb_p_points//2
        bdPhi, _, _ = jax.lax.fori_loop(Np_, Np_+nb_conds, bdPhi_pd_body_func, (bdPhi, nb_conds, jump_points))
        Np_ += nb_p_points
        jump_points += nb_conds



    def bdPhi_pn_body_func(i, vals):
        """Utility function for JAX's fori_loop on Periodic nodes with Neumann conditions (same nodes as above, just an extra set of conditions to satisfy well-posedness)"""
        bdPhi, nb_conds, jump_points = vals

        support1 = cloud.sorted_local_supports[i]
        support2 = cloud.sorted_local_supports[i+nb_conds]
        support = jnp.concatenate((support1, support2))

        grads1 = jnp.nan_to_num(grad_rbf_vec(nodes[i], nodes[support]), neginf=0., posinf=0.)
        grads2 = jnp.nan_to_num(grad_rbf_vec(nodes[i+nb_conds], nodes[support]), neginf=0., posinf=0.)

        if hasattr(cloud, "sorted_outward_normals"):
            normals1 = cloud.sorted_outward_normals[i-Ni-Nd]
            normals2 = cloud.sorted_outward_normals[i-Ni-Nd+nb_conds]
        else:
            normals1 = jnp.zeros((DIM,))    ## TODO: raise warning if this is taken
            normals2 = jnp.zeros((DIM,))

        diff_grads = jnp.dot(grads1, normals1) - jnp.dot(grads2, -normals2)

        return bdPhi.at[i-Ni-jump_points+sum(Np)//2, support].set(diff_grads), nb_conds, jump_points


    Np_ = Ni+Nd+Nn+Nr
    jump_points = 0
    for nb_p_points in Np:
        nb_conds = nb_p_points//2
        bdPhi, _, _ = jax.lax.fori_loop(Np_, Np_+nb_conds, bdPhi_pn_body_func, (bdPhi, nb_conds, jump_points))
        Np_ += nb_p_points
        jump_points += nb_conds






    ### Fill Matrix the polynomial bd(P) by dealing with each monomial in turn for each type of bc ###
    node_ids_d = [k for k,v in cloud.node_types.items() if v == "d"]
    node_ids_n = [k for k,v in cloud.node_types.items() if v == "n"]
    node_ids_r = [k for k,v in cloud.node_types.items() if v == "r"]
    betas_is = [robin_coeffs[k] for k,v in cloud.node_types.items() if v == "r"]

    monomials = make_all_monomials(M)

    ## Dirichlet nodes
    if len(node_ids_d) > 0:
        node_ids_d = jnp.array(node_ids_d)
        for j in range(M):
            monomial_vec = jax.vmap(monomials[j], in_axes=(0,), out_axes=0)
            bdP = bdP.at[node_ids_d-Ni, j].set(monomial_vec(nodes[node_ids_d]))

    ## Neumann nodes
    if len(node_ids_n) > 0:
        normals_n = jnp.stack([cloud.outward_normals[i] for i in node_ids_n], axis=0)
        node_ids_n = jnp.array(node_ids_n)

        dot_vec = jax.vmap(jnp.dot, in_axes=(0,0), out_axes=0)
        for j in range(M):
            grad_monomial = jax.grad(monomials[j])
            grad_monomial_vec = jax.vmap(grad_monomial, in_axes=(0,), out_axes=0)
            grads = grad_monomial_vec(nodes[node_ids_n])
            bdP = bdP.at[node_ids_n-Ni, j].set(dot_vec(grads, normals_n))

    ## Robin nodes
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

    ## Periodic nodes. TODO: add more comments to explain what I was doing below !
    if len(Np) > 0:
        jump_points = Ni+Nd+Nn+Nr
        Np_ = Ni+Nd+Nn+Nr
        for nb_p_points in Np:

            nb_conds = nb_p_points//2
            node_ids_pd1 = jnp.arange(jump_points, jump_points+nb_conds)
            node_ids_pd2 = jnp.arange(jump_points+nb_conds, jump_points+nb_p_points)

            for j in range(M):
                monomial_vec = jax.vmap(monomials[j], in_axes=(0,), out_axes=0)
                diff = monomial_vec(nodes[node_ids_pd1]) - monomial_vec(nodes[node_ids_pd2])
                bdP = bdP.at[jnp.arange(Np_,Np_+nb_conds)-Ni, j].set(diff)

            Np_ += nb_conds
            jump_points += nb_p_points

        half_Np = sum(Np)//2
        jump_points = Ni+Nd+Nn+Nr
        Np_ = Ni+Nd+Nn+Nr
        for nb_p_points in Np:
            nb_conds = nb_p_points//2
            node_ids_pn1 = range(jump_points, jump_points+nb_conds)
            node_ids_pn2 = range(jump_points+nb_conds, jump_points+nb_p_points)

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


    return bdPhi, bdP



def assemble_B(operator:callable, cloud:Cloud, rbf:callable, nb_monomials:int, diff_args:list, robin_coeffs:dict):
    """Assemble matrix B using opPhi, bdPhi, opP, bdP, and A, see equation (*) from Shahane et al.

    Args:
        operator (callable): the differential operator to apply
        cloud (Cloud): the cloud of points to use
        rbf (callable): the radial basis function to use
        nb_monomials (int): the number of monomials to use
        diff_args (list): the fields to use in the aplication of the operator. Can be either the field values themselves or their coefficients
        robin_coeffs (dict): the coefficients for the Robin boundary conditions if needed

    Returns:
        Float[Array, "dim1 nb_nodes"]: The matrix B to use in the PDE solver
    """

    N, Ni = cloud.N, cloud.Ni
    M = nb_monomials

    opPhi, opP = assemble_op_Phi_P(operator, cloud, rbf, M, diff_args)
    bdPhi, bdP = assemble_bd_Phi_P(cloud, rbf, M, robin_coeffs)

    full_opPhi = jnp.zeros((N, N))
    full_opP = jnp.zeros((N, M))

    full_opPhi = full_opPhi.at[:Ni, :].set(opPhi[:, :])
    full_opP = full_opP.at[:Ni, :].set(opP[:, :])

    full_opPhi = full_opPhi.at[Ni:, :].set(bdPhi[:, :])
    full_opP = full_opP.at[Ni:, :].set(bdP[:, :])

    diffMat = jnp.concatenate((full_opPhi, full_opP), axis=-1)

    inv_A = assemble_invert_A(cloud, rbf, M)
    B = diffMat @ inv_A

    return B[:, :N]


def core_compute_coefficients(field:jnp.ndarray, cloud:Cloud, rbf:callable, nb_monomials:int):
    """ Find nodal and polynomial coefficients for scalar field directly from the number of monomials """ 

    rhs = jnp.concatenate((field, jnp.zeros((nb_monomials))))   ## Question, does it make sense to append zeros to the field values. TODO: verify this analytically
    inv_A = assemble_invert_A(cloud, rbf, nb_monomials)

    return inv_A@rhs


def compute_coefficients(field:jnp.ndarray, cloud:Cloud, rbf:callable, max_degree:int):
    """ Find nodal and polynomial coefficients for scalar field """
    N = cloud.N
    M = compute_nb_monomials(max_degree, cloud.dim)     ## Carefull with the problem dimension: 2 by default

    return core_compute_coefficients(field, cloud, rbf, M)




def get_field_coefficients(field:jnp.ndarray, cloud:Cloud, rbf:callable, max_degree:int):
    """ Find nodal and polynomial coefficients for scalar field. Alias of compute_coefficients()""" 
    nb_monomials = compute_nb_monomials(max_degree, cloud.dim)

    rhs = jnp.concatenate((field, jnp.zeros((nb_monomials))))
    inv_A = assemble_invert_A(cloud, rbf, nb_monomials)

    return inv_A@rhs



def assemble_q(operator:callable, boundary_conditions:dict, cloud:Cloud, rbf:callable, nb_monomials:int, rhs_args:list):
    """Assemble the right hand side q using the given operator (See equation * from Shahane et al.)

    Args:
        operator (callable): the rhs operator to apply
        boundary_conditions (dict): the boundary conditions to use
        cloud (Cloud): the cloud of points to use
        rbf (callable): the radial basis function to use
        nb_monomials (int): the number of monomials to use
        rhs_args (list): the fields to use in the application of the operator. Can be either the field values themselves or their coefficients

    Returns:
        Float[Array, "nb_nodes"]: The vector q
    """

    N = cloud.N
    Ni = cloud.Ni
    M = nb_monomials
    q = jnp.zeros((N,))

    ## Make sure we use coefficients of the field here
    if rhs_args != None:
        fields_coeffs = []
        for field in rhs_args:
            if field.shape[0] == N:
                fields_coeffs.append(core_compute_coefficients(field, cloud, rbf, M))
            else:
                fields_coeffs.append(field)
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

    return q
