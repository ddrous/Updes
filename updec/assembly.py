import jax
import jax.numpy as jnp

from functools import partial

from updec.utils import compute_nb_monomials
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
        for j in range(N):          ## TODO: Fix this with only local support
            if i != j:               ## Non-differentiability. Won't have this problem with local support

                if cloud.boundaries[i] == 0 or cloud.boundaries[i] == 1:    ## Internal or Dirichlet node
                    Phi = Phi.at[i, j].set(nodal_rbf(cloud.nodes[i], cloud.nodes[j]))

                elif cloud.boundaries[i] == 2:    ## Neumann node
                    grad = grad_rbf(cloud.nodes[i], cloud.nodes[j])
                    normal = cloud.outward_normals[i]
                    Phi = Phi.at[i, j].set(jnp.dot(grad, normal))

    return Phi


def assemble_P(cloud:Cloud, nb_monomials:int):
    """ See (6) from Shanane """
    N = cloud.N
    M = nb_monomials
    P = jnp.zeros((N, M), dtype=jnp.float32)

    for j in range(M):
        monomial = partial(make_monomial, id=j)
        grad_monomial = jax.grad(monomial)
        for i in range(N):

            if cloud.boundaries[i] == 0 or cloud.boundaries[i] == 1:    ## Internal or Dirichlet node
                P = P.at[i, j].set(monomial(cloud.nodes[i]))

            elif cloud.boundaries[i] == 2:    ## Neumann node
                grad = grad_monomial(cloud.nodes[i])
                normal = cloud.outward_normals[i]
                P = P.at[i, j].set(jnp.dot(grad, normal))

    return P


## Cache the results for future calls to this function
def assemble_A(cloud, rbf, nb_monomials=2):
    """ Assemble matrix A, see (4) from Shanane """

    Phi = assemble_Phi(cloud, rbf)
    P = assemble_P(cloud, nb_monomials)

    N, M = Phi.shape[1], P.shape[1]
    
    A = jnp.zeros((N+M, N+M), dtype=jnp.float32)
    A = A.at[:N, :N].set(Phi)
    A = A.at[:N, N:].set(P)
    A = A.at[N:, :N].set(P.T)

    return A



def assemble_op_Phi_P(operator:callable, cloud:Cloud, nb_monomials:int):
    """ Assembles upper op(Phi): the collocation matrix to which a differential operator is applied """
    ## Only the internal nodes (M, N)

    N = cloud.N
    Ni = cloud.Ni
    M = nb_monomials
    opPhi = jnp.zeros((Ni, N), dtype=jnp.float32)
    opP = jnp.zeros((Ni, M), dtype=jnp.float32)

    for i in range(Ni):
        assert cloud.boundaries[i] == 0, "not an internal node"    ## Internal node

        for j in range(N):  ## TODO: Fix this with only Local support
            if i != j:      ## Only go through the local support because of non-differentiability at distance r=0.
                opPhi = opPhi.at[i, j].set(operator(cloud.nodes[i], node=cloud.nodes[j]))

        for j in range(M):
            opP = opP.at[i, j].set(operator(cloud.nodes[i], monomial=j))

    return opPhi, opP


def assemble_B(operator:callable, cloud:Cloud, rbf:callable, max_degree:int):
    """ Assemble B using opPhi, P, and A """

    N, Ni = cloud.N, cloud.Ni
    M = compute_nb_monomials(max_degree, 2)

    Phi, P = assemble_Phi(cloud, rbf), assemble_P(cloud, M)
    opPhi, opP = assemble_op_Phi_P(operator, cloud, M)

    full_opPhi = jnp.zeros((N, N), dtype=jnp.float32)
    full_opP = jnp.zeros((N, M), dtype=jnp.float32)

    full_opPhi = full_opPhi.at[:Ni, :].set(opPhi[:, :])
    full_opP = full_opP.at[:Ni, :].set(opP[:, :])

    full_opPhi = full_opPhi.at[Ni:, :].set(Phi[Ni:, :])
    full_opP = full_opP.at[Ni:, :].set(P[Ni:, :])

    diffMat = jnp.concatenate((full_opPhi, full_opP), axis=-1)

    A = assemble_A(cloud, rbf, M)
    inv_A = jnp.linalg.inv(A)
    B = diffMat @ inv_A

    B1 = B[:, :N]
    return B1


def assemble_q(operator:callable, cloud:Cloud, boundary_functions:dict):
    """ Assemble the right hand side q using the operator """
    ### Boundary conditions should match all the types of boundaries
    
    N = cloud.N
    Ni = cloud.Ni
    q = jnp.zeros((N,))

    for i in range(Ni):
        assert cloud.boundaries[i]==0, "not an internal node"
        q = q.at[i].set(operator(cloud.nodes[i]))

    for i in range(Ni, N):
        assert cloud.boundaries[i]==1 or cloud.boundaries[i]==2, "not a dirichlet nor neumann node"
        bd_op = boundary_functions[cloud.surfaces[i]]
        q = q.at[i].set(bd_op(cloud.nodes[i]))

    return q
