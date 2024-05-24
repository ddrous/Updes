# %%

"""
Super-Scaled Updes on the Laplace equation with RBFs
"""

# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = "false"
import time

import jax
import jax.numpy as jnp

# jax.config.update('jax_platform_name', 'cpu')
# jax.config.update("jax_enable_x64", True)

import matplotlib.pyplot as plt
import seaborn as sns

from updes import *

DATAFOLDER = "./data/TempFolder/"

RBF = partial(polyharmonic, a=1)
MAX_DEGREE = 0

Nx = Ny = 10
SUPPORT_SIZE = "max"
# SUPPORT_SIZE = 9*1
facet_types={"South":"n", "West":"d", "North":"d", "East":"d"}

start = time.time()
cloud = SquareCloud(Nx=Nx, Ny=Ny, facet_types=facet_types, support_size=SUPPORT_SIZE)
walltime = time.time() - start

print(f"Cloud generation walltime: {walltime:.2f} seconds")

# cloud.visualize_cloud(s=0.5, figsize=(7,6));

## %%

def my_diff_operator(x, center=None, rbf=None, monomial=None, fields=None):
    return nodal_laplacian(x, center, rbf, monomial)

def my_rhs_operator(x, centers=None, rbf=None, fields=None):
    return -0.0

d_north = lambda node: jnp.sin(jnp.pi * node[0])
d_zero = lambda node: 0.0
boundary_conditions = {"South":d_zero, "West":d_zero, "North":d_north, "East":d_zero}

start = time.time()
sol = pde_solver_jit(diff_operator=my_diff_operator, 
                rhs_operator = my_rhs_operator, 
                cloud = cloud, 
                boundary_conditions = boundary_conditions, 
                rbf=RBF,
                max_degree=MAX_DEGREE)
walltime = time.time() - start

minutes = walltime // 60 % 60
seconds = walltime % 60
print(f"Walltime: {minutes} minutes {seconds:.2f} seconds")

## RBF solution
rbf_sol = sol.vals


fig = plt.figure(figsize=(6*1,5))
ax= fig.add_subplot(1, 1, 1, projection='3d')
cloud.visualize_field(rbf_sol, cmap="jet", projection="3d", title="Laplace with RBFs", ax=ax);
plt.show()
## Savefig
fig.savefig(DATAFOLDER+"super_scaled.png", dpi=300)
