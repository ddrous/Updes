import jax
import jax.numpy as jnp

## Euclidian distance
def distance(node1, node2):
    return jnp.linalg.norm(node1 - node2)**2
