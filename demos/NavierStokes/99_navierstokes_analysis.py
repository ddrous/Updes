# %%

"""
Comparison of NavierStokes control methods
"""

import jax.numpy as jnp

import matplotlib.pyplot as plt

from updec import *


#%%

DATAFOLDER = "./data/Comparison/"


facet_types_vel = {"Wall":"d", "Inflow":"d", "Outflow":"n", "Blowing":"d", "Suction":"d"}
facet_types_phi = {"Wall":"n", "Inflow":"n", "Outflow":"d", "Blowing":"n", "Suction":"n"}

cloud_vel = GmshCloud(filename="./meshes/channel_blowing_suction.py", facet_types=facet_types_vel, mesh_save_location=DATAFOLDER)
cloud_phi = GmshCloud(filename=DATAFOLDER+"mesh.msh", facet_types=facet_types_phi)

#%%

print("Loading arrays for DAL")

dal_arrays = jnp.load(DATAFOLDER+"dal.npz")
arraynames = dal_arrays.files
print(arraynames)





#%%

print("Loading arrays for DP")

dp_arrays = jnp.load(DATAFOLDER+"dp.npz")
arraynames = dp_arrays.files
print(arraynames)




#%%

print("Loading arrays for PINN forward")

pinn_fw_arrays = jnp.load(DATAFOLDER+"pinn_forward.npz")
arraynames = pinn_fw_arrays.files
print(arraynames)



#%%

print("Loading arrays for PINN inverse step 1")

Wj_id = 0
pinn_inv_1_arrays = jnp.load(DATAFOLDER+"pinn_inv_1_"+str(Wj_id)+".npz")
arraynames = pinn_inv_1_arrays.files
print(arraynames)





#%%

print("Loading arrays for PINN inverse step 2")

Wj_id = 0
pinn_inv_2_arrays = jnp.load(DATAFOLDER+"pinn_inv_2_"+str(Wj_id)+".npz")
arraynames = pinn_inv_2_arrays.files
print(arraynames)

pinn_inv_2_arrays['mem_time_cum']


#%%

print("Loading arrays for PINN inverse: weight vs cost")

pinn_inv_final_arrays = jnp.load(DATAFOLDER+"pinn_inv_2_final.npz")
arraynames = pinn_inv_final_arrays.files
print(arraynames)

#%%

## Time vs Max Memory vs Best Accuracy for all three methods
## These arrays will be filled by hand



# %%

## Now begins the comparison
