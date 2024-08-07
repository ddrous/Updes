# %%
# %load_ext autoreload
# %autoreload 2

# %%

"""
Test of the Updec package on the Advection-Diffusion equation with RBFs: 
PDE here: https://en.wikipedia.org/wiki/Convection%E2%80%93diffusion_equation
"""

import time

import jax
import jax.numpy as jnp

# jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)

from updes import *
# key = jax.random.PRNGKey(13)
key = None

# from torch.utils.tensorboard import SummaryWriter


RUN_NAME = "TempFolder"
DATAFOLDER = "./data/" + RUN_NAME +"/"
# DATAFOLDER = "demos/Advection/data/"+RUN_NAME+"/"
make_dir(DATAFOLDER)

RBF = partial(polyharmonic, a=1)
# RBF = gaussian
MAX_DEGREE = 0

DT = 1e-4
NB_TIMESTEPS = 30
PLOT_EVERY = 10

## Diffusive constant
K = 0.0
VEL = jnp.array([500.0, 0.0])

Nx = 35
Ny = 35
SUPPORT_SIZE = "max"

# facet_types={"South":"p1", "North":"p1", "West":"p2", "East":"p2"}
facet_types={"South":"p1", "North":"p1", "West":"p2", "East":"p2"}
cloud = SquareCloud(Nx=Nx, Ny=Ny, facet_types=facet_types, noise_key=key, support_size=SUPPORT_SIZE)

cloud.visualize_cloud(s=0.1, figsize=(7,3));


# %%

def my_diff_operator(x, center=None, rbf=None, monomial=None, fields=None):
    val = nodal_value(x, center, rbf, monomial)
    grad = nodal_gradient(x, center, rbf, monomial)
    lap = nodal_laplacian(x, center, rbf, monomial)
    return (val/DT) + jnp.dot(VEL, grad) - K*lap + fields[0]*val

def my_rhs_operator(x, centers=None, rbf=None, fields=None):
    val = value(x, fields[:,0], centers, rbf)
    return (val/DT)

d_zero = lambda x: 0.
boundary_conditions = {"South":d_zero, "West":d_zero, "North":d_zero, "East":d_zero}

## u0 is zero everywhere except at a point in the middle
# u0 = jnp.zeros(cloud.N)
# source_id = int(cloud.N*0.71)
# source_neighbors = jnp.array(cloud.local_supports[source_id][:cloud.N//40])
# # source_id = 0
# # source_neighbors = jnp.array(cloud.local_supports[source_id][:1])
# u0 = u0.at[source_neighbors].set(0.95)

def gaussian(x, y, x0, y0, sigma):
    return jnp.exp(-((x-x0)**2 + (y-y0)**2) / (2*sigma**2))
xy = cloud.sorted_nodes
u0 = gaussian(xy[:,0], xy[:,1], 0.35, 0.5, 1/20)

u_sink = 0e-5*gaussian(xy[:,0], xy[:,1], 0.85, 0.5, 1/100)


## Begin timestepping for 100 steps

# fig = plt.figure(figsize=(6,3))
# ax1= fig.add_subplot(1, 1, 1, projection='3d')
# ax = fig.add_subplot(1, 1, 1)


u = u0.copy()
ulist = [u]

start = time.time()

for i in range(1, NB_TIMESTEPS+1):
    ufield = pde_solver_jit(diff_operator=my_diff_operator,
                        rhs_operator = my_rhs_operator,
                        diff_args=[u_sink],
                        rhs_args=[u],
                        cloud = cloud,
                        boundary_conditions = boundary_conditions, 
                        rbf=RBF,
                        max_degree=MAX_DEGREE,)

    u = ufield.vals
    ulist.append(u)

    # if i<=3 or i%PLOT_EVERY==0:
    #     print(f"Step {i}")
    #     # plt.cla()
    #     # cloud.visualize_field(u, cmap="jet", projection="3d", title=f"Step {i}")
    #     ax, _ = cloud.visualize_field(u, cmap="jet", title=f"Step {i}", vmin=0, vmax=1, figsize=(6*10,3*10), colorbar=False, levels=200)
    #     plt.show()


walltime = time.time() - start

minutes = walltime // 60 % 60
seconds = walltime % 60
print(f"Walltime: {minutes} minutes {seconds:.2f} seconds")



# %%

filename = DATAFOLDER + "adv_diff_sink.gif"
cloud.animate_fields([ulist], cmaps="jet", filename=filename, figsize=(7,3), titles=["Advection-Diffusion with Sink"]);



# %%

