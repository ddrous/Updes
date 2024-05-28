import os
import jax
import jax.numpy as jnp

RBF = None          ## Default radial basis function
MAX_DEGREE = None              ## Max degree for monomials
DIM = 2                     ## Default problem dimension
__version__ = "1.0.2"       ## Package version  ## TODO check if okay to do this here

PREALLOCATE = False
if not PREALLOCATE:
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = "false"


FLOAT64 = True
jax.config.update("jax_enable_x64", FLOAT64)   ## Use double precision by default

jnp.set_printoptions(linewidth=jnp.inf)         ## Print arrays on the same line
