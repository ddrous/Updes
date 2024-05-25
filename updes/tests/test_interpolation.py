#%%

import jax
import jax.numpy as jnp

from updes import *

# from torch.utils.tensorboard import SummaryWriter
# import datetime


RBF = polyharmonic
MAX_DEGREE = 4
Nx = 8
Ny = Nx
SUPPORT_SIZE = Nx*Ny-1

# print(run_name)

facet_types1={"South":"n", "West":"d", "North":"n", "East":"d"}
cloud1 = SquareCloud(Nx=Nx, Ny=Ny, facet_types=facet_types1, support_size=SUPPORT_SIZE)
print(cloud1.global_indices)

facet_types2={"South":"d", "West":"n", "North":"d", "East":"n"}
cloud2 = SquareCloud(Nx=Nx, Ny=Ny, facet_types=facet_types2, support_size=SUPPORT_SIZE)
print(cloud2.global_indices)


# field = jnp.arange(0, cloud1.N)
# cloud1.visualize_field(field)


# #%%
# new_field = interpolate_field(field, cloud, cloud)
# cloud.visualize_field(new_field)

#%%
# sorted_map1 = sorted(cloud1.renumbering_map.items(), key=lambda x:x[0])
# indexer1 = jnp.array(list(dict(sorted_map1).values()))
# field_orig = field[indexer1]
# print(field_orig)
# print()
# indexer2 = jnp.array(list(cloud2.renumbering_map.keys()))
# new_field = field_orig[indexer2]

# new_field = interpolate_field(field, cloud1, cloud2)

# cloud2.visualize_field(new_field)
# print(new_field)



# %%

## Run the test. Should be equal on the internal nodes at the very least
# assert jnp.allclose(field[:cloud1.Ni], new_field[:cloud2.Ni], atol=1e-12)

# %%

def test_interpolation():
    field = jnp.arange(0, cloud1.N)
    new_field = interpolate_field(field, cloud1, cloud2)
    assert jnp.allclose(field[:cloud1.Ni], new_field[:cloud2.Ni], atol=1e-12)

# test_interpolation(field, cloud1, cloud2)