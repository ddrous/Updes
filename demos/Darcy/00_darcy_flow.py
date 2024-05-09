# %%
# %load_ext autoreload
# %autoreload 2

# %%

"""
Updecs on the Darcy-Flow equation with RBFs
"""

import time

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from updes import *
key = None

DATAFOLDER = "./data/"
make_dir(DATAFOLDER)

RBF = partial(polyharmonic, a=1)
MAX_DEGREE = 1

Nx = 30
Ny = 30
SUPPORT_SIZE = "max"

# %%

## Permeability field

def perm_diff_operator(x, center=None, rbf=None, monomial=None, fields=None):
    val = nodal_value(x, center, rbf, monomial)
    return val

def perm_rhs_operator(x, centers=None, rbf=None, fields=None):
    return value(x, fields[:,0], centers, rbf)

bc_perm = lambda x: 0.
bc_perm = {"South":bc_perm, "West":bc_perm, "North":bc_perm, "East":bc_perm}

dtype = "d"
facet_types_perm = {"South":dtype, "North":dtype, "West":dtype, "East":dtype}
cloud_perm = SquareCloud(Nx=Nx, Ny=Ny, facet_types=facet_types_perm, noise_key=key, support_size=SUPPORT_SIZE)

def perm_func(x, y):
    tmp = jnp.where(x>=y, 0., 0.90)
    tmp = jnp.where(x<=0.5, tmp, 0)
    return jnp.where(y**2>=0.1, tmp, 0.90)
xy = cloud_perm.sorted_nodes
permeability = jax.vmap(perm_func)(xy[:,0], xy[:,1])

perm_field = pde_solver_jit(diff_operator=perm_diff_operator,
                    rhs_operator=perm_rhs_operator,
                    rhs_args=[permeability],
                    cloud=cloud_perm,
                    boundary_conditions=bc_perm, 
                    rbf=RBF,
                    max_degree=MAX_DEGREE,)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.75*2,3))
cloud_perm.visualize_field(field=perm_field.vals, title="Permeability field", ax=ax1);


# %%

## Darcy flow solve equation solve

facet_types={"South":"d", "North":"d", "West":"d", "East":"d"}
cloud = SquareCloud(Nx=Nx, Ny=Ny, facet_types=facet_types, noise_key=key, support_size=SUPPORT_SIZE)

# cloud.visualize_cloud(s=0.1, figsize=(4,3));

def my_diff_operator(x, center=None, rbf=None, monomial=None, fields=None):
    lap = nodal_laplacian(x, center, rbf, monomial)
    return lap

def my_rhs_operator(x, centers=None, rbf=None, fields=None):
    return value(x, fields[:,0], centers, rbf)

d_zero = lambda x: 0.
boundary_conditions = {"South":d_zero, "West":d_zero, "North":d_zero, "East":d_zero}

start = time.time()

ufield = pde_solver_jit(diff_operator=my_diff_operator,
                    rhs_operator = my_rhs_operator,
                    rhs_args=[perm_field.vals],
                    cloud = cloud,
                    boundary_conditions = boundary_conditions, 
                    rbf=RBF,
                    max_degree=MAX_DEGREE,)

walltime = time.time() - start

minutes = walltime // 60 % 60
seconds = walltime % 60
print(f"Walltime: {minutes} minutes {seconds:.2f} seconds")

cloud.visualize_field(ufield.vals, cmap="jet", title=f"Solution field", ax=ax2, levels=200)
plt.show()

# %%

# plt.draw()
plt.savefig(DATAFOLDER+"darcy_flow.png", dpi=300, bbox_inches="tight");
