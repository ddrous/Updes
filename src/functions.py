import jax
import jax.numpy as jnp
from utils import distance


def multiquadric(r):
    eps = 1
    return jnp.sqrt(1 + (eps*r)**2)

def polyharmonic(r):
    a = 1
    return r**(2*a+1)


