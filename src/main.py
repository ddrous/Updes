import jax
import jax.numpy as jnp
from differential_operators import *
from cloud import *

cloud = Cloud()
field = jnp.ones((cloud.N, 2))

## Operates on radial basis functions and polynomials at position x
def my_diff_operator(x, node_j, monomial_j, *args):
    return  args[0] * laplacian(x, node_j, monomial_j)

## Operates on entire fields at position x
def my_rhs(x):
    return divergence(x, field, cloud)
