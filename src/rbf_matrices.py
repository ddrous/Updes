import jax
import jax.numpy as jnp
import math

from cloud import Cloud
from rbf_functions import *


def assemble_Phi(cloud:Cloud, rbf:callable):
    """ Assemble the matrix Phi (see equation 5) from Shahane """
    ## TODO: Make this matrix sparse. Only consider local supports

    N = cloud.N
    Phi = jnp.zeros((N, N), dtype=jnp.float32)
    grad_rbf = jax.grad(rbf)

    for i in range(N):
        node_i = cloud.nodes[i]
        for j in range(N):
            node_j = cloud.nodes[j]

            if cloud.boundaries[i] == 0 or cloud.boundaries[i] == 1:    ## Internal or Dirichlet node
                Phi.at[i, j].set(rbf(node_i, node_j))

            elif cloud.boundaries[i] == 2:    ## Neumann node
                grad = grad_rbf(node_i, node_j)
                normal = cloud.outward_normals[i]
                Phi.at[i, j].set(jnp.dot(grad, normal))

    return Phi

def assemble_P(cloud:Cloud, max_degrees:int):
    """ See (6) from Shanane """
    N = cloud.N
    m = math.comb(max_degrees+2, max_degrees)       ## 2 is the dimension of the problem
    P = jnp.zeros((N, m), dtype=jnp.float32)

    for j in range(m):
        monomial = polynomial(j)
        grad_monomial = jax.grad(monomial)
        for i in range(N):

            if cloud.boundaries[i] == 0 or cloud.boundaries[i] == 1:    ## Internal or Dirichlet node
                P.at[i, j].set(monomial(cloud.nodes[i]))

            elif cloud.boundaries[i] == 2:    ## Neumann node
                grad = grad_monomial(cloud.nodes[i])
                normal = cloud.outward_normals[i]
                P.at[i, j].set(jnp.dot(grad, normal))

    return P


def assemble_A(cloud, rbf=polyharmonic, max_degrees=2):
    """ Assemble matrix A, see (4) from Shanane"""

    Phi = assemble_Phi(cloud, rbf)
    P = assemble_P(cloud, max_degrees)

    N, m = Phi.shape[1], P.shape[1]
    
    A = jnp.zeros((N+m, N+m), dtype=jnp.float32)
    A = A.at[:N, :N].set(Phi)
    A = A.at[:N, N:].set(P)
    A = A.at[N:, :N].set(P.T)

    return A
