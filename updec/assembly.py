import jax
import jax.numpy as jnp
from jax.tree_util import Partial

from functools import cache

from updec.utils import make_nodal_rbf, make_monomial, compute_nb_monomials
from updec.cloud import Cloud


def assemble_Phi(cloud:Cloud, nodal_rbf:callable=None):
    """ Assemble the matrix Phi (see equation 5) from Shahane """
    ## TODO: Make this matrix sparse. Only consider local supports
    ## rbf could be a string instead

    N = cloud.N
    Phi = jnp.zeros((N, N))
    # nodal_rbf = jax.jit(partial(make_nodal_rbf, rbf=rbf))         
    # nodal_rbf = Partial(make_nodal_rbf, rbf=rbf)                    ## TODO Use the prexisting nodal_rbf func
    nodal_rbf_vec = jax.vmap(nodal_rbf, in_axes=(None, 0), out_axes=0)
    nodes = cloud.sort_nodes_jnp()

    for i in range(N):
        # for j in cloud.local_supports[i]:
        #     Phi = Phi.at[i, j].set(nodal_rbf(cloud.nodes[i], cloud.nodes[j]))

        support_ids = jnp.array(cloud.local_supports[i])
        Phi = Phi.at[i, support_ids].set(nodal_rbf_vec(cloud.nodes[i], nodes[support_ids]))

    return Phi


def assemble_P(cloud:Cloud, nb_monomials:int):
    """ See (6) from Shanane """
    N = cloud.N
    M = nb_monomials
    P = jnp.zeros((N, M))
    nodes = cloud.sort_nodes_jnp()

    for j in range(M):
        # monomial = jax.jit(Partial(make_monomial, id=j))      ## IS
        monomial = Partial(make_monomial, id=j)
        monomial_vec = jax.vmap(monomial, in_axes=(0,), out_axes=0)
        P = P.at[:, j].set(monomial_vec(nodes))

        # for i in range(N):
        #     P = P.at[i, j].set(monomial(cloud.nodes[i]))

    return P


## Cache the results for future calls to this function
@cache
def assemble_A(cloud, nodal_rbf, nb_monomials=2):
    """ Assemble matrix A, see (4) from Shanane """

    Phi = assemble_Phi(cloud, nodal_rbf)
    P = assemble_P(cloud, nb_monomials)

    N, M = Phi.shape[1], P.shape[1]

    A = jnp.zeros((N+M, N+M))
    A = A.at[:N, :N].set(Phi)
    A = A.at[:N, N:].set(P)
    A = A.at[N:, :N].set(P.T)

    return A

@cache          ## Turn this into assemble and LU decompose
def assemble_invert_A(cloud, nodal_rbf, nb_monomials):
    A = assemble_A(cloud, nodal_rbf, nb_monomials)
    return jnp.linalg.inv(A)


def assemble_op_Phi_P(operator:callable, cloud:Cloud, nb_monomials:int, nodal_rbf:callable, *args):
    """ Assembles upper op(Phi): the collocation matrix to which a differential operator is applied """
    ## Only the internal nodes (M, N)

    # operator = jax.jit(operator, static_argnums=2)

    N = cloud.N
    Ni = cloud.Ni
    M = nb_monomials
    opPhi = jnp.zeros((Ni, N))
    opP = jnp.zeros((Ni, M))

    nodes = cloud.sort_nodes_jnp()
    if len(args) > 0:
        fields = jnp.stack(args, axis=-1)
    else:
        fields = jnp.ones((N,1))     ## TODO Won't be used tho. FIx this !

    # operator_rbf = Partial(operator, monomial=None)
    def operator_rbf(x, node=None, args=None): 
        return operator(x, node, None, nodal_rbf, args)
    operator_rbf_vec = jax.vmap(operator_rbf, in_axes=(None, 0, None), out_axes=0)

    # operator_mon = Partial(operator, node=None)
    def operator_mon(x, args=None, monomial=None):
        return operator(x, None, monomial, nodal_rbf, args)

    coords = cloud.sort_nodes_jnp()
    internal_ids = jnp.arange(Ni)

    for i in range(Ni):
        assert cloud.node_boundary_types[i] == "i", "not an internal node"    ## Internal node

        support_ids = jnp.array(cloud.local_supports[i])
        opPhi = opPhi.at[i, support_ids].set(operator_rbf_vec(cloud.nodes[i], nodes[support_ids], fields[i]))

    for j in range(M):
        monomial_func = Partial(make_monomial, id=j)
        operator_mon_func = Partial(operator_mon, monomial=monomial_func)
        operator_mon_vec = jax.vmap(operator_mon_func, in_axes=(0, 0), out_axes=0)
        opP = opP.at[internal_ids, j].set(operator_mon_vec(coords[internal_ids], fields[internal_ids]))

    return opPhi, opP



def assemble_bd_Phi_P(cloud:Cloud, nb_monomials:int, nodal_rbf:callable, *args):
    """ Assembles upper op(Phi): the collocation matrix to which a differential operator is applied """
    ## Only the internal nodes (M, N)

    # operator = jax.jit(operator, static_argnums=2)

    N, Ni = cloud.N, cloud.Ni
    Nd, Nn, Nr = cloud.Nd, cloud.Nn, cloud.Nr
    M = nb_monomials
    bdPhi = jnp.zeros((Nd+Nn+Nr, N))
    bdP = jnp.zeros((Nd+Nn+Nr, M))

    # nodal_rbf = Partial(make_nodal_rbf, rbf=rbf)                    ## TODO JIT THIS, and Use the prexisting nodal_rbf func
    grad_rbf = jax.grad(nodal_rbf)
    # grad_rbf = jax.jit(jax.grad(nodal_rbf))

    nodal_rbf_vec = jax.vmap(nodal_rbf, in_axes=(None, 0), out_axes=0)
    grad_rbf_vec = jax.vmap(grad_rbf, in_axes=(None, 0), out_axes=0)

    nodes = cloud.sort_nodes_jnp()


    ### Fill Matrix Phi with vectorisation from axis=1 ###

    for i in range(Ni, N):
        ii = i-Ni        ## Actual index in the matrices
        assert cloud.node_boundary_types[i] in ["d", "n"], "not a boundary node"    ## Internal node

        if cloud.node_boundary_types[i] == "d":
            support_d = jnp.array([j for j in cloud.local_supports[i]])
            bdPhi = bdPhi.at[ii, support_d].set(nodal_rbf_vec(cloud.nodes[i], nodes[support_d]))

        elif cloud.node_boundary_types[i] == "n":    ## Neumann node
            support_n = jnp.array([j for j in cloud.local_supports[i]])
            grads = grad_rbf_vec(cloud.nodes[i], nodes[support_n])
            bdPhi = bdPhi.at[ii, support_n].set(jnp.dot(grads, cloud.outward_normals[i]))


    ### Fill Matrix P with vectorisation from axis=0 ###
    node_ids_d = jnp.array([k for k,v in cloud.node_boundary_types.items() if v == "d"])
    node_ids_n = jnp.array([k for k,v in cloud.node_boundary_types.items() if v == "n"])

    if node_ids_d.shape[0] > 0:
        for j in range(M):
            monomial = Partial(make_monomial, id=j)
            monomial_vec = jax.vmap(monomial, in_axes=(0,), out_axes=0)
            bdP = bdP.at[node_ids_d-Ni, j].set(monomial_vec(nodes[node_ids_d]))

    if node_ids_n.shape[0] > 0:
        normals_n = jnp.stack([cloud.outward_normals[i] for i in node_ids_n.tolist()], axis=0)
        dot_vec = jax.vmap(jnp.dot, in_axes=(0,0), out_axes=0)
        for j in range(M):
            monomial = Partial(make_monomial, id=j)
            grad_monomial = jax.grad(monomial)
            grad_monomial_vec = jax.vmap(grad_monomial, in_axes=(0,), out_axes=0)
            grads = grad_monomial_vec(nodes[node_ids_n])
            bdP = bdP.at[node_ids_n-Ni, j].set(dot_vec(grads, normals_n))

    return bdPhi, bdP



def assemble_B(operator:callable, cloud:Cloud, rbf:callable, max_degree:int, *args):
    """ Assemble B using opPhi, P, and A """

    N, Ni = cloud.N, cloud.Ni
    M = compute_nb_monomials(max_degree, 2)

    # Phi, P = assemble_Phi(cloud, rbf), assemble_P(cloud, M)
    nodal_rbf = Partial(make_nodal_rbf, rbf=rbf)

    opPhi, opP = assemble_op_Phi_P(operator, cloud, M, nodal_rbf, *args)
    bdPhi, bdP = assemble_bd_Phi_P(cloud, M, nodal_rbf, *args)

    full_opPhi = jnp.zeros((N, N))
    full_opP = jnp.zeros((N, M))

    full_opPhi = full_opPhi.at[:Ni, :].set(opPhi[:, :])
    full_opP = full_opP.at[:Ni, :].set(opP[:, :])

    full_opPhi = full_opPhi.at[Ni:, :].set(bdPhi[:, :])
    full_opP = full_opP.at[Ni:, :].set(bdP[:, :])

    diffMat = jnp.concatenate((full_opPhi, full_opP), axis=-1)

    # A = assemble_A(cloud, nodal_rbf, M)       ## TODO make this work for nodal_rbf
    # A = assemble_A(cloud, rbf, M)

    inv_A = assemble_invert_A(cloud, nodal_rbf, M)

    B = diffMat @ inv_A

    B1 = B[:, :N]
    return B1


def assemble_q(operator:callable, cloud:Cloud, boundary_functions:dict):
    """ Assemble the right hand side q using the operator """
    ### Boundary conditions should match all the types of boundaries

    N = cloud.N
    Ni = cloud.Ni
    q = jnp.zeros((N,))

    ## Internal node
    operator_vec = jax.vmap(operator, in_axes=(0,), out_axes=(0))
    nodes = cloud.sort_nodes_jnp()
    internal_ids = jnp.arange(Ni)
    q = q.at[internal_ids].set(operator_vec(nodes[internal_ids]))


    ## Facet nodes
    for f_id in cloud.facet_types.keys():
        assert f_id in boundary_functions.keys(), "facets and boundary functions don't match ids"

        bd_op = boundary_functions[f_id]
        bd_op_vec = jax.vmap(bd_op, in_axes=(0,), out_axes=0)
        bd_node_ids = jnp.array(cloud.facet_nodes[f_id])
        q = q.at[bd_node_ids].set(bd_op_vec(nodes[bd_node_ids]))

    return q
