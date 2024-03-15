# %%
%load_ext autoreload
%autoreload 2

# %%

"""
Test of the Updec package on the Wave equation with RBFs: 
PDE here: https://en.wikipedia.org/wiki/Convection%E2%80%93diffusion_equation
"""

import time

import jax
import jax.numpy as jnp

# jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)

from updec import *
# key = jax.random.PRNGKey(13)
key = None

# from torch.utils.tensorboard import SummaryWriter
jax.numpy.set_printoptions(precision=2)

RUN_NAME = "TempFolder"
DATAFOLDER = "./data/" + RUN_NAME +"/"
# DATAFOLDER = "demos/Advection/data/"+RUN_NAME+"/"
make_dir(DATAFOLDER)

RBF = partial(polyharmonic, a=3)
# RBF = gaussian
MAX_DEGREE = 2

DT = 5e-4
NB_TIMESTEPS = 500
PLOT_EVERY = 5

## Diffusive constant
C = 1.

Nx = 25
Ny = 25
SUPPORT_SIZE = "max"

# facet_types={"South":"d", "North":"d", "West":"d", "East":"d"}
facet_types={"South":"n", "North":"n", "West":"n", "East":"n"}
cloud = SquareCloud(Nx=Nx, Ny=Ny, facet_types=facet_types, noise_key=key, support_size=SUPPORT_SIZE)

# cloud.visualize_cloud(s=0.1, figsize=(7,6));

# cloud.facet_types
# cloud.facet_nodes
# print("Local supports:", cloud.local_supports[0])
print(cloud.Np)
# print(jnp.flip(cloud.global_indices.T, axis=0))
# cloud.print_global_indices()
# print(cloud.sorted_nodes)
# cloud.sorted_outward_normals
# cloud.outward_normals

# %%

def my_diff_operator(x, center=None, rbf=None, monomial=None, fields=None):
    val = nodal_value(x, center, rbf, monomial)
    lap = nodal_laplacian(x, center, rbf, monomial)
    return (val/(DT**2)) + C*lap

def my_rhs_operator(x, centers=None, rbf=None, fields=None):
    u_prev = value(x, fields[:,0], centers, rbf)
    u_prev_prev = value(x, fields[:,1], centers, rbf)
    return (2*u_prev - u_prev_prev)/(DT**2)

d_zero = lambda x: 0.
boundary_conditions = {"South":d_zero, "West":d_zero, "North":d_zero, "East":d_zero}

## Uo is a 2D gaussian centered at the middle of the domain
def gaussian(x, y, x0, y0, sigma):
    return jnp.exp(-((x-x0)**2 + (y-y0)**2) / (2*sigma**2))
xy = cloud.sorted_nodes
u0 = gaussian(xy[:,0], xy[:,1], 0.85, 0.85, 1/10)

## Begin timestepping for 100 steps
cloud.visualize_field(u0, cmap="coolwarm", title=f"Step {0}", vmin=0, vmax=1, figsize=(6,6),colorbar=False);


# %%
ulist = [u0, 1*DT + u0]

start = time.time()

for i in range(1, NB_TIMESTEPS+1):
    uprev = ulist[-1]
    uprevprev = ulist[-2]

    ufield = pde_solver_jit(diff_operator=my_diff_operator,
                        rhs_operator = my_rhs_operator,
                        rhs_args=[uprev, uprevprev],
                        cloud = cloud,
                        boundary_conditions = boundary_conditions, 
                        rbf=RBF,
                        max_degree=MAX_DEGREE,)
    ulist.append(ufield.vals)

    # if i<=3 or i%PLOT_EVERY==0:
    #     print(f"Step {i}")
    #     # plt.cla()
    #     # cloud.visualize_field(ulist[-1], cmap="coolwarm", projection="3d", title=f"Step {i}")
    #     ax, _ = cloud.visualize_field(ulist[-1], cmap="coolwarm", title=f"Step {i}", vmin=None, vmax=None, figsize=(6,6),colorbar=False)
    #     # plt.draw()
    #     plt.show()


walltime = time.time() - start

minutes = walltime // 60 % 60
seconds = walltime % 60
print(f"Walltime: {minutes} minutes {seconds:.2f} seconds")



# %%

## Clip ulist arrays between -1 and 1
ulist = [jnp.clip(u, -1, 1) for u in ulist]

filename = DATAFOLDER + "wave.gif"
cloud.animate_fields([ulist], cmaps="coolwarm", filename=filename, figsize=(7.5,6), titles=["Wave with RBFs"]);



# %%

# ulist
