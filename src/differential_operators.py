import jax
import jax.numpy as jnp
from rbf_functions import *
from rbf_matrices import *
from utils import distance
from cloud import Cloud

### These quantities should be defined in main
rbf = "polyharmonic"
max_degree = 2
# cloud = Cloud(Nx=7, Ny=5)
# invA = jnp.linalg.inv(assemble_A(cloud, rbf))



## Does calling this all the time cause problems ?
def nodal_gradient(x, node_j=None, monomial_j=None):
    """ Computes the gradients of the RBF and polynomial functions """
    ## Only one of node_j or monomial_j can be given
    if node_j:
        return jax.grad(nodal_rbf)(x, node_j)
    elif monomial_j:
        return jax.grad(polynomial(monomial_j))(x)

### N.B: """ No divergence for RBF and Polynomial functions, because they are scalars """

def nodal_laplacian(x, node_j=None, monomial_j=None):
    """ Computes the lapalcian of the RBF and polynomial functions """
    if node_j:
        gradxx = jax.grad(jax.grad(nodal_rbf)(x, node_j)[0])[0]
        gradyy = jax.grad(jax.grad(nodal_rbf)(x, node_j)[1])[1]
    elif monomial_j:
        gradxx = jax.grad(jax.grad(polynomial(monomial_j))(x)[0])[0]
        gradyy = jax.grad(jax.grad(polynomial(monomial_j))(x)[1])[1]

    return gradxx + gradyy


def gradient(x, s, cloud):
    """ Computes the gradient of quantity s at position x """
    ## Find coefficients for s on the cloud
    lambdas, gammas = compute_coefficients(s, cloud, rbf, max_degree)

    ## Now, compute the gradient of the field
    final_grad = jnp.array([0.,0.])
    for j in range(lambdas.shape[0]):
        rbf_grad = nodal_gradient(x, node_j=cloud.nodes[j], monomial_j=None)
        final_grad = final_grad.at[:].add(lambdas[j] * rbf_grad)

    for j in range(gammas.shape[0]):
        polynomial_grad = nodal_gradient(x, node_j=None, monomial_j=j)
        final_grad = final_grad.at[:].add(gammas[j] * polynomial_grad)

    return final_grad


def divergence(x, s, cloud):
    """ Computes the divergence of vector quantity s at position x """
    ds1dx1 = gradient(x, s[...,0], cloud)[0]
    ds2dx2 = gradient(x, s[...,1], cloud)[1]

    return ds1dx1 + ds2dx2


def laplacian(x, s, cloud):
    """ Computes the laplacian of quantity s at position x """
    ## Find coefficients for s
    lambdas, gammas = compute_coefficients(s, cloud, rbf, max_degree)

    ## Now, compute the laplacian of the field
    final_lap = jnp.array([0.])
    for j in range(lambdas.shape[0]):
        rbf_lap = nodal_gradient(x, node_j=cloud.nodes[j], monomial_j=None)
        final_lap = final_lap.at[:].add(lambdas[j] * rbf_lap)

    for j in range(gammas.shape[0]):
        poly_lap = nodal_gradient(x, node_j=None, monomial_j=j)
        final_lap = final_lap.at[:].add(gammas[j] * poly_lap)

    return final_lap
