# %%

"""
Control of Laplace equation with Direct Adjoint Looping (DAL)
"""

import jax
import jax.numpy as jnp
import optax

import matplotlib.pyplot as plt
from tqdm import tqdm

from updec import *

#%%

RBF = polyharmonic
MAX_DEGREE = 1


DATAFOLDER = "./data/Comparison/"
Nx = 15
Ny = Nx
LR = 1e-2
EPOCHS = 5


facet_types={"North":"d", "South":"d", "West":"d", "East":"d"}
train_cloud = SquareCloud(Nx=Nx, Ny=Ny, facet_types=facet_types, noise_key=None, support_size="max")

train_cloud.visualize_cloud(s=0.1, title="Training cloud", figsize=(5,4));

#%%

print("Loading arrays for DAL")

dal_arrays = jnp.load(DATAFOLDER+"dal.npz")
arraynames = dal_arrays.files
print(arraynames)


objective_cost_dal = dal_arrays["objective_cost"]
north_mse_dal = dal_arrays["north_mse"]
optimal_bcn_dal = dal_arrays["optimal_bcn"]
exact_solution_dal = dal_arrays["exact_solution"]
optimal_solution_dal = dal_arrays["optimal_solution"]



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
