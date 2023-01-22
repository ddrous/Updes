import jax
import jax.numpy as jnp

import math
from functools import partial

from updec.cloud import Cloud
from updec.utils import make_nodal_rbf, make_monomial



def assemble_Phi(cloud:Cloud, rbf:callable=None):
    """ Assemble the matrix Phi (see equation 5) from Shahane """
    ## TODO: Make this matrix sparse. Only consider local supports
    ## rbf could be a string instead

    N = cloud.N
    Phi = jnp.zeros((N, N), dtype=jnp.float32)
    nodal_rbf = partial(make_nodal_rbf, rbf=rbf)
    grad_rbf = jax.grad(nodal_rbf)

    for i in range(N):
        node_i = cloud.nodes[i]
        for j in range(N):
            node_j = cloud.nodes[j]

            if cloud.boundaries[i] == 0 or cloud.boundaries[i] == 1:    ## Internal or Dirichlet node
                Phi.at[i, j].set(nodal_rbf(node_i, node_j))

            elif cloud.boundaries[i] == 2:    ## Neumann node
                grad = grad_rbf(node_i, node_j)
                normal = cloud.outward_normals[i]
                Phi.at[i, j].set(jnp.dot(grad, normal))

    return Phi


def assemble_P(cloud:Cloud, nb_monomials:int):
    """ See (6) from Shanane """
    N = cloud.N
    m = nb_monomials
    P = jnp.zeros((N, m), dtype=jnp.float32)

    for j in range(m):
        monomial = partial(make_monomial, id=j)
        grad_monomial = jax.grad(monomial)
        for i in range(N):

            if cloud.boundaries[i] == 0 or cloud.boundaries[i] == 1:    ## Internal or Dirichlet node
                P.at[i, j].set(monomial(cloud.nodes[i]))

            elif cloud.boundaries[i] == 2:    ## Neumann node
                grad = grad_monomial(cloud.nodes[i])
                normal = cloud.outward_normals[i]
                P.at[i, j].set(jnp.dot(grad, normal))

    return P


## Cache the results for future calls to this function
def assemble_A(cloud, rbf, nb_monomials=2):
    """ Assemble matrix A, see (4) from Shanane """

    Phi = assemble_Phi(cloud, rbf)
    P = assemble_P(cloud, nb_monomials)

    N, m = Phi.shape[1], P.shape[1]
    
    A = jnp.zeros((N+m, N+m), dtype=jnp.float32)
    A = A.at[:N, :N].set(Phi)
    A = A.at[:N, N:].set(P)
    A = A.at[N:, :N].set(P.T)

    return A



def assemble_diff_Phi_P(operator:callable, cloud:Cloud, nb_monomials:int):
    """ Assembles upper op(Phi): the collocation matrix to which a differential operator is applied """
    ## Only the internal nodes (M, N)

    N = cloud.N
    M = cloud.M
    m = nb_monomials
    opPhi = jnp.zeros((M, N), dtype=jnp.float32)
    opP = jnp.zeros((M, m), dtype=jnp.float32)

    for i in range(M):
        assert cloud.boundaries[i] == 0, "not an internal node"    ## Internal node
        node_i = cloud.nodes[i]

        for j in range(N):
            node_j = cloud.nodes[j]
            opPhi.at[i, j].set(operator(node_i, node_j=node_j))

        for j in range(m):
            opP.at[i, j].set(operator(node_i, monomial_j=j))

    return opPhi, opP


def assemble_B(operator:callable, cloud:Cloud, rbf:str, max_degree:int):
    """ Assemble B using opPhi, P, and A """

    N, M = cloud.N, cloud.M
    m = math.comb(max_degree+2, max_degree)

    Phi, P = assemble_Phi(cloud, rbf), assemble_P(cloud, m)
    opPhi, opP = assemble_diff_Phi_P(operator, cloud, m)

    full_opPhi = jnp.zeros((N, N), dtype=jnp.float32)
    full_opP = jnp.zeros((N, m), dtype=jnp.float32)

    full_opPhi = full_opPhi.at[:M, :].set(opPhi[:, :])
    full_opP = full_opP.at[:M, :].set(opP[:, :])

    full_opPhi = full_opPhi.at[M:, :].set(Phi[M:, :])
    full_opP = full_opP.at[M:, :].set(P[M:, :])

    diffMat = jnp.concatenate((full_opPhi, full_opP), axis=-1)

    A = assemble_A(cloud, rbf, m)
    inv_A = jnp.linalg.inv(A)
    B = diffMat @ inv_A

    B1 = B[:, :N]
    return B1


def assemble_q(operator:callable, cloud:Cloud, boundary_functions:dict):
    """ Assemble the right hand side q using the operator """
    ### Boundary conditions should match all the types of boundaries
    
    N = cloud.N
    M = cloud.M
    rhs = jnp.zeros((N,))

    for i in range(M):
        assert cloud.boundaries[i]==0, "not an internal node"
        rhs.at[i].set(operator(cloud.node[i]))

    for i in range(M, N):
        assert cloud.boundaries[i]==1, "not a dirichlet nor neumann node"
        bd_op = boundary_functions[cloud.surfaces[i]]
        rhs.at[i].set(bd_op(cloud.node[i]))
