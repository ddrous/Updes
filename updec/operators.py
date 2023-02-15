import jax
import jax.numpy as jnp
from jax.tree_util import Partial

from functools import cache, lru_cache

# from updec.config import RBF, MAX_DEGREE, DIM
import updec.config as UPDEC
from updec.utils import make_nodal_rbf, make_monomial, compute_nb_monomials, SteadySol, polyharmonic, gaussian
from updec.cloud import Cloud
from updec.assembly import assemble_A, assemble_invert_A, assemble_B, assemble_q, new_compute_coefficients


@Partial(jax.jit, static_argnums=[2,3])
def nodal_value(x, center=None, rbf=None, monomial=None):
    """ Computes the rbf and polynomial functions """
    """ x: gradient at position x 
        node: the node defining the rbf function (if for a monomial)
        rbf: rbf function to use
        monomial: the id of the monomial (if for a monomial)
    """
    ## Only one of node_j or monomial_j can be given
    if center != None:
        # rbf = Partial(make_rbf, rbf=rbf)
        return rbf(x, center)
    elif monomial != None:
        # monomial = Partial(make_monomial, id=monomial)
        return monomial(x)


@cache
def core_gradient_rbf(rbf):
    return jax.grad(rbf)

## LRU cache this
# @lru_cache(maxsize=32)
@cache
def core_gradient_mon(monomial):
    # monomial = Partial(make_monomial, id=monomial)
    return jax.grad(monomial)

## Does calling this all the time cause problems ?
def nodal_gradient(x, center=None, rbf=None, monomial=None):
    """ Computes the gradients of the RBF and polynomial functions """
    """ x: gradient at position x 
        node: the node defining the rbf function (if for a monomial)
        rbf: rbf function to use
        monomial: the id of the monomial (if for a monomial)
    """
    # cfg.RBFtest = gaussian
    ## Only one of node_j or monomial_j can be given
    if center != None:
        # nodal_rbf = Partial(make_nodal_rbf, rbf=rbf)
        # return jax.grad(nodal_rbf)(x, node)
        # return jnp.where(jnp.all(x==node), jax.grad(nodal_rbf)(x, node), jnp.array([0., 0.]))
        return core_gradient_rbf(rbf)(x, center)
    elif monomial != None:
        # monomial = Partial(make_monomial, id=monomial)
        # return jax.grad(monomial)(x)
        return core_gradient_mon(monomial)(x)

### N.B: """ No divergence for RBF and Polynomial functions, because they are scalars """

## LRU cache this
# @lru_cache(maxsize=32)
@cache
def core_laplacian_rbf(rbf):
    return jax.jacfwd(jax.grad(rbf))

## LRU cache this
# @lru_cache(maxsize=32)
@cache
def core_laplacian_mon(monomial):
    # monomial = Partial(make_monomial, id=monomial)
    return jax.jacfwd(jax.grad(monomial))

def nodal_laplacian(x, center=None, rbf=None, monomial=None):     ## TODO Jitt through this efficiently
    """ Computes the lapalcian of the RBF and polynomial functions """
    if center != None:
        # nodal_rbf = Partial(make_nodal_rbf, rbf=rbf)
        # nodal_rbf = rbf
        return jnp.trace(core_laplacian_rbf(rbf)(x, center))      ## TODO: try reverse mode
    elif monomial != None:
        # monomial = Partial(make_monomial, id=monomial)
        # monomial = monomial
        return jnp.trace(core_laplacian_mon(monomial)(x))



def compute_coefficients(field:jnp.DeviceArray, cloud:Cloud, rbf:callable, max_degree:int):
    """ Find nodal and polynomial coefficients for scaar field s """ 
    N = cloud.N
    M = compute_nb_monomials(max_degree, 2)     ## Carefull with the problem dimension: 2

    ##TODO solve the linear system quicker (store and cache LU decomp) 
    # nodal_rbf = Partial(make_nodal_rbf, rbf=rbf)

    rhs = jnp.concatenate((field, jnp.zeros((M))))
    # A = assemble_A(cloud, nodal_rbf, M)
    inv_A = assemble_invert_A(cloud, rbf, M)
    # coefficients = jnp.linalg.solve(A, rhs)
    coefficients = inv_A@rhs

    lambdas = coefficients[:N]
    gammas = coefficients[N:]

    return lambdas[:, jnp.newaxis], gammas[:, jnp.newaxis]


# def gradient(x, field, cloud, rbf=None, max_degree=2):
#     """ Computes the gradient of field quantity s at position x """
#     ## Find coefficients for s on the cloud
#     lambdas, gammas = compute_coefficients(field, cloud, rbf, max_degree)

#     ## Now, compute the gradient of the field
#     final_grad = jnp.array([0.,0.])

#     ################################        ATTEMPT TO VECTORIZE ... TODO Only rbf works, nor monom. and still, with nans
#     # sorted_nodes = cloud.sorted_nodes
#     # monomial_ids = jnp.arange(gammas.shape[0])

#     # nodal_grad_vec_rbf = jax.vmap(nodal_gradient, in_axes=(None, 0, None, None), out_axes=0)
#     # nodal_grad_vec_mon = jax.vmap(nodal_gradient, in_axes=(None, None, 0, None), out_axes=0)

#     # nodal_grads = nodal_grad_vec_rbf(x, sorted_nodes, None, rbf)
#     # monom_grads = nodal_grad_vec_mon(x, None, monomial_ids, None)

#     # final_grad = final_grad.at[:].add(jnp.sum(lambdas * nodal_grads, axis=0))
#     # final_grad = final_grad.at[:].add(jnp.sum(gammas * monom_grads, axis=0))
#     ################################

#     for j in range(lambdas.shape[0]):                                           ## TODO: Awwfull !! Vectorize this please !
#         node_j = cloud.nodes[j]
#         # if jnp.all(x==node_j):      ## Non-differentiable case
#         #     continue
#         # rbf_grad = nodal_gradient(x, node=node_j, rbf=rbf)
#         rbf_grad = jnp.where(jnp.all(x==node_j), 0., nodal_gradient(x, node=node_j, rbf=rbf))       ## TODO To account for non-differentiable case
#         final_grad = final_grad.at[:].add(lambdas[j] * rbf_grad)

#     for j in range(gammas.shape[0]):                                                   ### TODO: Use VMAP to vectorise this too !!
#         monomial = Partial(make_monomial, id=j)
#         polynomial_grad = nodal_gradient(x, monomial=monomial)
#         final_grad = final_grad.at[:].add(gammas[j] * polynomial_grad)

#     return final_grad




def gradient(x, field, centers, rbf=None):
    """ Computes the gradient of field quantity s at position x """
    ## Find coefficients for s on the cloud

    ## Now, compute the gradient of the field
    N = centers.shape[0]
    lambdas, gammas = field[:N], field[N:]

    final_grad = jnp.array([0.,0.])
    for j in range(lambdas.shape[0]):                                           ## TODO: Awwfull !! Vectorize this please !
        # if jnp.all(x==node_j):      ## Non-differentiable case
        #     continue
        # rbf_grad = nodal_gradient(x, node=node_j, rbf=rbf)
        rbf_grad = jnp.where(jnp.all(x==centers[j]), 0., nodal_gradient(x, center=centers[j], rbf=rbf))       ## TODO To account for non-differentiable case
        final_grad = final_grad.at[:].add(lambdas[j] * rbf_grad)

    for j in range(gammas.shape[0]):                                                   ### TODO: Use VMAP to vectorise this too !!
        monomial = Partial(make_monomial, id=j)
        polynomial_grad = nodal_gradient(x, monomial=monomial)
        final_grad = final_grad.at[:].add(gammas[j] * polynomial_grad)

    return final_grad





def divergence(x, field, centers, rbf=None):
    """ Computes the divergence of vector quantity s at position x """
    dfieldx_dx = gradient(x, field[...,0], centers, rbf)[0]
    dfieldy_dy = gradient(x, field[...,1], centers, rbf)[1]

    return dfieldx_dx + dfieldy_dy


def laplacian(x, field, centers, rbf=None):
    """ Computes the laplacian of quantity s at position x """
    ## Find coefficients for s
    # lambdas, gammas = compute_coefficients(field, cloud, rbf, max_degree)

    ## Now, compute the laplacian of the field

    N = centers.shape[0]
    lambdas, gammas = field[:N], field[N:]

    final_lap = jnp.array([0.])
    for j in range(lambdas.shape[0]):
        # rbf_lap = nodal_laplacian(x, node=node_j, rbf=rbf)
        rbf_lap = jnp.where(jnp.all(x==centers[j]), 0., nodal_laplacian(x, center=centers[j], rbf=rbf))
        final_lap = final_lap.at[:].add(lambdas[j] * rbf_lap)

    for j in range(gammas.shape[0]):
        monomial = Partial(make_monomial, id=j)
        poly_lap = nodal_laplacian(x, monomial=monomial)
        final_lap = final_lap.at[:].add(gammas[j] * poly_lap)

    return final_lap[0]


## Devise different LU, LDL decomposition strategies make functions here
def pde_solver( diff_operator:callable,
                rhs_operator:callable,
                cloud:Cloud, 
                boundary_conditions:dict, 
                rbf:callable,
                max_degree:int,
                diff_args = None,
                rhs_args = None):
    """ Solve a PDE """

    # nodal_operator = jax.jit(nodal_operator, static_argnums=2)
    # nodal_operator = jax.jit(nodal_operator)
    # global_operator = jax.jit(global_operator
    # 

    ### For rememmering purposes
    UPDEC.BRF = rbf
    UPDEC.MAX_DEGREE = max_degree
    UPDEC.DIM = cloud.dim


    # 
    # TODO Here
    nb_monomials = compute_nb_monomials(max_degree, cloud.dim)

    B1 = assemble_B(diff_operator, cloud, rbf, nb_monomials, diff_args)
    rhs = assemble_q(rhs_operator, boundary_conditions, cloud, rbf, nb_monomials, rhs_args)

    sol_vals = jnp.linalg.solve(B1, rhs)
    # sol_coeffs = compute_coefficients(sol_vals, cloud, rbf, max_degree)
    sol_coeffs = new_compute_coefficients(sol_vals, cloud, rbf, nb_monomials)

    # return sol_vals, jnp.concatenate(sol_coeffs)         ## TODO: return an object like solve_ivp
    return SteadySol(sol_vals, sol_coeffs)
