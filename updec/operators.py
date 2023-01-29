import jax
import jax.numpy as jnp

from functools import partial

from updec.utils import compute_nb_monomials
from updec.cloud import Cloud
from updec.utils import make_nodal_rbf, make_monomial
from updec.assembly import assemble_A, assemble_B, assemble_q


def nodal_value(x, node=None, monomial=None, rbf=None):
    """ Computes the rbf and polynomial functions """
    """ x: gradient at position x 
        node: the node defining the rbf function (if for a monomial)
        rbf: rbf function to use
        monomial: the id of the monomial (if for a monomial)
    """
    ## Only one of node_j or monomial_j can be given
    if node != None:
        nodal_rbf = partial(make_nodal_rbf, rbf=rbf)
        return nodal_rbf(x, node)
    elif monomial != None:
        monomial_func = partial(make_monomial, id=monomial)
        return monomial_func(x)


## Does calling this all the time cause problems ?
def nodal_gradient(x, node=None, monomial=None, rbf=None):
    """ Computes the gradients of the RBF and polynomial functions """
    """ x: gradient at position x 
        node: the node defining the rbf function (if for a monomial)
        rbf: rbf function to use
        monomial: the id of the monomial (if for a monomial)
    """
    ## Only one of node_j or monomial_j can be given
    if node != None:
        nodal_rbf = partial(make_nodal_rbf, rbf=rbf)
        # return jax.grad(nodal_rbf)(x, node)
        return jnp.where(jnp.all(x==node), jax.grad(nodal_rbf)(x, node), jnp.array([0., 0.]))
    elif monomial != None:
        monomial_func = partial(make_monomial, id=monomial)
        return jax.grad(monomial_func)(x)

### N.B: """ No divergence for RBF and Polynomial functions, because they are scalars """

def nodal_laplacian(x, node=None, monomial=None, rbf=None):
    """ Computes the lapalcian of the RBF and polynomial functions """
    if node != None:
        nodal_rbf = partial(make_nodal_rbf, rbf=rbf)
        return jnp.trace(jax.jacfwd(jax.grad(nodal_rbf))(x, node))      ## TODO: try reverse mode
    elif monomial != None:
        monomial_func = partial(make_monomial, id=monomial)
        return jnp.trace(jax.jacfwd(jax.grad(monomial_func))(x))



def compute_coefficients(field:jnp.DeviceArray, cloud:Cloud, rbf:callable, max_degree:int):
    """ Find nodal and polynomial coefficients for scaar field s """
    N = cloud.N
    M = compute_nb_monomials(max_degree, 2)     ## Carefull with the problem dimension: 2

    ##TODO solve the linear system quicker (store and cache LU decomp) 
    A = assemble_A(cloud, rbf, M)
    rhs = jnp.concatenate((field, jnp.zeros((M))))
    coefficients = jnp.linalg.solve(A, rhs)

    lambdas = coefficients[:N]
    gammas = coefficients[N:]

    return lambdas, gammas


def gradient(x, field, cloud:Cloud, rbf=None, max_degree=2):
    """ Computes the gradient of field quantity s at position x """
    ## Find coefficients for s on the cloud
    lambdas, gammas = compute_coefficients(field, cloud, rbf, max_degree)
    sorted_nodes = cloud.sort_jnp_nodes
    monomial_ids = jnp.arange(0, gammas.shape[0])

    ## Now, compute the gradient of the field
    # final_grad = jnp.array([0.,0.])

    # for j in range(lambdas.shape[0]):               ### TODO: Use VMAP here and SUM afterwards
    #     node_j = cloud.nodes[j]
    #     # if jnp.all(x==node_j):      ## Non-differentiable case
    #     #     continue
    #     rbf_grad = nodal_gradient(x, node=node_j, rbf=rbf)
    #     final_grad = final_grad.at[:].add(lambdas[j] * rbf_grad)

    # for j in range(gammas.shape[0]):
    #     polynomial_grad = nodal_gradient(x, monomial=j)
    #     final_grad = final_grad.at[:].add(gammas[j] * polynomial_grad)

    nodal_grad_vec_rbf = jax.vmap(nodal_gradient, in_axes=(None, 0, None, None), out_axes=0)
    nodal_grad_vec_mon = jax.vmap(nodal_gradient, in_axes=(None, None, 0, None), out_axes=0)

    final_grad = final_grad.at[:].add(jnp.sum(lambdas*nodal_grad_vec_rbf(x, sorted_nodes), axis=0))
    final_grad = final_grad.at[:].add(jnp.sum(gammas*nodal_grad_vec_mon(x, monomial_ids), axis=0))

    return final_grad


def divergence(x, field, cloud, rbf=None, max_degree=2):
    """ Computes the divergence of vector quantity s at position x """
    dfieldx_dx = gradient(x, field[...,0], cloud, rbf, max_degree)[0]
    dfieldy_dy = gradient(x, field[...,1], cloud, rbf, max_degree)[1]

    return dfieldx_dx + dfieldy_dy


def laplacian(x, field, cloud, rbf=None, max_degree=2):
    """ Computes the laplacian of quantity s at position x """
    ## Find coefficients for s
    lambdas, gammas = compute_coefficients(field, cloud, rbf, max_degree)

    ## Now, compute the laplacian of the field
    final_lap = jnp.array([0.])
    for j in range(lambdas.shape[0]):
        node_j = cloud.nodes[j]
        if jnp.all(x==node_j):      ## Non-differentiable case !! TODO !! 
            continue
        rbf_lap = nodal_laplacian(x, node=node_j, rbf=rbf)
        final_lap = final_lap.at[:].add(lambdas[j] * rbf_lap)

    for j in range(gammas.shape[0]):
        poly_lap = nodal_laplacian(x, monomial=j)
        final_lap = final_lap.at[:].add(gammas[j] * poly_lap)

    return final_lap[0]


## Devise different LU LDL decomposition strategies make functions here
def pde_solver(nodal_operator:callable, 
                global_operator:callable, 
                cloud:Cloud, 
                boundary_conditions:dict, 
                rbf:callable, 
                max_degree:int,
                *args):
    """ Solve a PDE """

    nodal_operator = jax.jit(nodal_operator, static_argnums=2)
    global_operator = jax.jit(global_operator)

    B1 = assemble_B(nodal_operator, cloud, rbf, max_degree, *args)
    rhs = assemble_q(global_operator, cloud, boundary_conditions)
    return jnp.linalg.solve(B1, rhs)
