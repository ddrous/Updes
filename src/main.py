import jax
import jax.numpy as jnp
from differential_operators import *

def my_operator(x, node_id=None, monomial_id=None, *args):
    return  args[0] * laplacian(x, node_id, monomial_id)
