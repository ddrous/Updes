# %%
%load_ext autoreload
%autoreload 2

# %%

"""
Test of the Updec package on the 2D Burgers equation with RBFs: 
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
jax.numpy.set_printoptions(precision=2)

RUN_NAME = "TempFolder"
DATAFOLDER = "./data/" + RUN_NAME +"/"
# DATAFOLDER = "demos/Advection/data/"+RUN_NAME+"/"
make_dir(DATAFOLDER)

RBF = partial(polyharmonic, a=3)
# RBF = gaussian
MAX_DEGREE = 2

DT = 1e-3
NB_TIMESTEPS = 100
PLOT_EVERY = 20

## Diffusive constant

Nx = 25
Ny = 25
SUPPORT_SIZE = "max"

# facet_types={"South":"d", "North":"d", "West":"d", "East":"d"}
facet_types={"South":"n", "North":"n", "West":"n", "East":"n"}
cloud = SquareCloud(Nx=Nx, Ny=Ny, facet_types=facet_types, noise_key=key, support_size=SUPPORT_SIZE)

# %% [markdown]
# ## Define the PDE
# $$
# \frac{\partial u}{\partial t} + u\frac{\partial u}{\partial x} + v \frac{\partial u}{\partial y} = 0
# $$
# $$
# \frac{\partial v}{\partial t} + u\frac{\partial v}{\partial x} + v \frac{\partial v}{\partial y} = 0
# $$

# when discretised in time (implicit scheme), we get
# $$
# \frac{u}{\Delta t} + u\frac{\partial u}{\partial x} + \tilde v \frac{\partial u}{\partial y} =  \frac{u^{\text{old}}}{\Delta t}
# $$
# $$
# \frac{v}{\Delta t} + \tilde u \frac{\partial v}{\partial x} + v \frac{\partial v}{\partial y} =  \frac{v^{\text{old}}}{\Delta t}
# $$

# - $\tilde u$ indicates u at the current time step, which we are improving to a fixed point. Similarly for $\tilde v$.
# - $u^{\text{old}}$ and $v^{\text{old}}$ are the values of u and v at the previous time step.

# %%

def my_diff_operator_u(x, center=None, rbf=None, monomial=None, fields=None):
    _, v_tilde = fields[0], fields[1]
    u_val = nodal_value(x, center, rbf, monomial)
    u_grad = nodal_gradient(x, center, rbf, monomial)
    return (u_val/(DT)) + u_val*u_grad[0] + v_tilde*u_grad[1]

def my_rhs_operator_u(x, centers=None, rbf=None, fields=None):
    u_old, _ = fields[:, 0], fields[:, 1]
    return value(x, u_old, centers, rbf)/DT

def my_diff_operator_v(x, center=None, rbf=None, monomial=None, fields=None):
    u_tilde, _ = fields[0], fields[1]
    v_val = nodal_value(x, center, rbf, monomial)
    v_grad = nodal_gradient(x, center, rbf, monomial)
    return (v_val/(DT)) + u_tilde*v_grad[0] + v_val*v_grad[1]

def my_rhs_operator_v(x, centers=None, rbf=None, fields=None):
    _, v_old = fields[:, 0], fields[:, 1]
    return value(x, v_old, centers, rbf)/DT


d_zero = lambda x: 0.
boundary_conditions_u = {"South":d_zero, "West":d_zero, "North":d_zero, "East":d_zero}
boundary_conditions_v = {"South":d_zero, "West":d_zero, "North":d_zero, "East":d_zero}



# %% [markdown]
# ## Initial conditions
# $$ u(x, y, 0) = \sin(\pi x) \cdot \sin(\pi y) $$
# $$ v(x, y, 0) = \cos(\pi x) \cdot \cos(\pi y) $$


# %%
## Uo is a 2D gaussian centered at the middle of the domain
xy = cloud.sorted_nodes
u0 = jnp.sin(1*jnp.pi*xy[:,0]) * jnp.sin(1*jnp.pi*xy[:,1])
v0 = jnp.cos(1*jnp.pi*xy[:,0]) * jnp.cos(1*jnp.pi*xy[:,1])

## Begin timestepping for 100 steps
cloud.visualize_field(u0, cmap="coolwarm", title=f"Step {0}", vmin=0, vmax=1, figsize=(6,6),colorbar=False);


# %%
sol_list = [[u0, v0]]

start = time.time()

for i in range(1, NB_TIMESTEPS+1):
    sol_prev = sol_list[-1]

    sol_now = pde_multi_solver(
        diff_operators=[my_diff_operator_u, my_diff_operator_v],
        rhs_operators=[my_rhs_operator_u, my_rhs_operator_v],
        diff_args=[sol_prev, sol_prev],
        rhs_args=[sol_prev, sol_prev],
        cloud=cloud,
        boundary_conditions=[boundary_conditions_u, boundary_conditions_v],
        nb_iters=5,
        tol=1e-10,
        rbf=RBF,
        max_degree=MAX_DEGREE,
    )

    sol_list.append([s.vals for s in sol_now])

    if i<=3 or i%PLOT_EVERY==0:
        print(f"Step {i}")
        # plt.cla()
        # cloud.visualize_field(ulist[-1], cmap="coolwarm", projection="3d", title=f"Step {i}")
        ax, _ = cloud.visualize_field(sol_list[-1][0], cmap="coolwarm", title=f"u utep {i}", vmin=None, vmax=None, figsize=(6,6), colorbar=False)
        # plt.draw()
        plt.show()


walltime = time.time() - start

minutes = walltime // 60 % 60
seconds = walltime % 60
print(f"Walltime: {minutes} minutes {seconds:.2f} seconds")



# %%

ulist = [sol[0] for sol in sol_list]
filename = DATAFOLDER + "burgers_u.gif"
cloud.animate_fields([ulist[:65:2]], cmaps="coolwarm", filename=filename, levels=200, duration=5, figsize=(7.5,6), titles=["Burgers with RBFs"]);

# normlist = [jnp.sqrt(sol[0]**2 + sol[1]**2) for sol in sol_list]
# filename = DATAFOLDER + "burgers_norm.mp4"
# cloud.animate_fields([normlist], cmaps="coolwarm", filename=filename, levels=200, duration=10, figsize=(7.5,6), titles=["Burgers with RBFs - norm"]);


# %%


