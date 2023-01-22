import jax
import jax.numpy as jnp
from rbf_functions import *
from utils import distance

rbf = polyharmonic

def nodal_rbf(x, node_id):
    return rbf(distance(x, node_id))

def gradient(x, node_id=None, monomial_id=None):
    if node_id:
        return jax.grad(rbf)(x, node_id)

    elif monomial_id:
        return jax.grad(polynomial(monomial_id))(x)

def divergence(x, node_id=None, monomial_id=None):
    if node_id:
        pass


def laplacian(x, node_id=None, monomial_id=None):
    pass