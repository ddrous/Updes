import jax
import jax.numpy as jnp
from jax.tree_util import Partial, tree_map

from functools import cache, lru_cache, partial

# from updec.config import RBF, MAX_DEGREE, DIM
import updec.config as UPDEC
from updec.utils import make_nodal_rbf, make_monomial, compute_nb_monomials, SteadySol, polyharmonic, gaussian, make_all_monomials
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
    if center != None:          ## TODO if else can be handled with grace in Jax
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
    if center != None:  ## TODO if alse can be handled with grace in Jax
        # nodal_rbf = Partial(make_nodal_rbf, rbf=rbf)
        # return jax.grad(nodal_rbf)(x, node)
        # return jnp.where(jnp.all(x==node), jax.grad(nodal_rbf)(x, node), jnp.array([0., 0.]))
        return core_gradient_rbf(rbf)(x, center)                    ## TODO: benchmark agains line above to see the cost of avoiding NaNs
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
        # return jnp.trace(core_laplacian_rbf(rbf)(x, center))      ## TODO: try reverse mode
        return jnp.nan_to_num(jnp.trace(core_laplacian_rbf(rbf)(x, center)), neginf=0., posinf=0.)      ## TODO: try reverse mode
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

_nodal_value_rbf_vec = jax.vmap(nodal_value, in_axes=(None, 0, None, None), out_axes=0)

def value(x, field, centers, rbf=None):
    """ Computes the gradient of field quantity s at position x """
    ## Find coefficients for s on the cloud

    ## Now, compute the value of the field
    N = centers.shape[0]
    lambdas, gammas = field[:N], field[N:]

    val_rbf = _nodal_value_rbf_vec(x, centers, rbf, None)
    final_val = jnp.sum(jnp.nan_to_num(lambdas*val_rbf, posinf=0., neginf=0.), axis=0)

    all_monomials = make_all_monomials(gammas.shape[0])
    for j in range(gammas.shape[0]):    ### TODO: Use VMAP to vectorise this too !!
        polynomial_val = nodal_value(x, monomial=all_monomials[j])
        final_val = final_val + jnp.nan_to_num(gammas[j] * polynomial_val, posinf=0., neginf=0.)

    return final_val




_nodal_gradient_rbf_vec = jax.vmap(nodal_gradient, in_axes=(None, 0, None, None), out_axes=0)

### ATEMPT TO VECTORIZE MONOMIALS THTOUGH TREE_MAP ###
# @partial(jax.vmap, in_axes=(None, 0), out_axes=0)
# def _nodal_gradient_mon(monomial, x):
#     return nodal_gradient(x, None, None, monomial)
# # _nodal_gradient_mon_vec = jax.vmap(nodal_gradient, in_axes=(None, None, None, 0), out_axes=0)
# @cache
# def _nodal_gradient_mon_vec(monomials):                     ## Vectorized using pytrees
#     return jnp.stack(tree_map(_nodal_gradient_mon, monomials))


# @Partial(jax.jit, static_argnums=3)
def gradient(x, field, centers, rbf=None):
    """ Computes the gradient of field quantity s at position x """

    N = centers.shape[0]
    lambdas, gammas = field[:N], field[N:]

    grads_rbf = _nodal_gradient_rbf_vec(x, centers, rbf, None)              ## TODO remove all NaNs
    # print(grads_rbf)
    lambdas = jnp.stack([lambdas, lambdas], axis=-1)                        ## TODO Why is Jax unable to broadcast below ?
    final_grad = jnp.sum(jnp.nan_to_num(lambdas*grads_rbf, posinf=0., neginf=0.), axis=0)

    all_monomials = make_all_monomials(gammas.shape[0])
    for j in range(gammas.shape[0]):                                                   ### TODO: Use VMAP to vectorise this too !!
        polynomial_grad = nodal_gradient(x, monomial=all_monomials[j])
        # print(polynomial_grad)
        final_grad = final_grad.at[:].add(jnp.nan_to_num(gammas[j] * polynomial_grad, posinf=0., neginf=0.))

    return final_grad


gradient_vec = jax.vmap(gradient, in_axes=(0, None, None, None), out_axes=0)


def divergence(x, field, centers, rbf=None):
    """ Computes the divergence of vector quantity s at position x """
    dfieldx_dx = gradient(x, field[...,0], centers, rbf)[0]
    dfieldy_dy = gradient(x, field[...,1], centers, rbf)[1]

    return dfieldx_dx + dfieldy_dy

divergence_vec = jax.vmap(divergence, in_axes=(0, None, None, None), out_axes=0)


_nodal_laplacian_rbf_vec = jax.vmap(nodal_laplacian, in_axes=(None, 0, None, None), out_axes=0)

def laplacian(x, field, centers, rbf=None):
    """ Computes the laplacian of quantity field at position x """
    ## Find coefficients for s
    # lambdas, gammas = compute_coefficients(field, cloud, rbf, max_degree)

    ## Now, compute the laplacian of the field

    N = centers.shape[0]
    lambdas, gammas = field[:N], field[N:]

    laps_rbf = _nodal_laplacian_rbf_vec(x, centers, rbf, None)              ## TODO remove all NaNs
    rbf_lap = jnp.sum(jnp.nan_to_num(lambdas*laps_rbf, posinf=0., neginf=0.), axis=0)


    all_monomials = make_all_monomials(gammas.shape[0])
    mon_lap = jnp.array([0.])
    for j in range(gammas.shape[0]):
        # monomial = Partial(make_monomial, id=j)
        poly_lap = nodal_laplacian(x, monomial=all_monomials[j])
        mon_lap = mon_lap.at[:].add(gammas[j] * poly_lap)

    return rbf_lap + mon_lap[0]

laplacian_vec = jax.vmap(laplacian, in_axes=(0, None, None, None), out_axes=0)


def interpolate_field(field, cloud1, cloud2):
    """ Interpolates field from cloud1 to cloud2 """

    assert cloud1.N == cloud2.N, "the two clouds do not contain the same number of nodes"   ## TODO: Make sure only the renumbering differs

    sorted_map1 = sorted(cloud1.renumbering_map.items(), key=lambda x:x[0])
    indexer1 = jnp.array(list(dict(sorted_map1).values()))
    field_orig = field[indexer1]

    indexer2 = jnp.array(list(cloud2.renumbering_map.keys()))

    return field_orig[indexer2]            ## TODO Think of a way to do this


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
    diff_operator = jax.jit(diff_operator, static_argnums=[2,3])
    rhs_operator = jax.jit(rhs_operator, static_argnums=2)

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
