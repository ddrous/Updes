#%%
%load_ext autoreload
%autoreload 2

# %%

"""
Test of the Updec package on the 2D Gray-Scott PDE for chemical reaction
"""


# %% [markdown]
# ## Define the PDE
# $$
# \frac{\partial u}{\partial t} = D_u \nabla^2 u - uv^2 + F(1 - u)
# $$
# $$
# \frac{\partial v}{\partial t} = D_v \nabla^2 v + uv^2 - (F + k)v
# $$

# $ D_u = 0.16, D_v=0.08, F=0.035, k=0.065 $

# This can go in the rhs of the PDE
# u and v are concatenated in a single vector U = [u, v]
# The tilde is solved in an iterative manner


# %% 
import time

import jax
import jax.numpy as jnp
import diffrax

# jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)

from updec import *

SEED = 2026
# key = jax.random.PRNGKey(SEED)
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

NB_TIMESTEPS = 100
PLOT_EVERY = 2

Du = 0.2097
Dv = 0.105
F = 0.03
k = 0.062

Nx = 32
Ny = 32
SUPPORT_SIZE = "max"

facet_types={"South":"p1", "North":"p1", "West":"p2", "East":"p2"}
# facet_types={"South":"d", "North":"d", "West":"d", "East":"d"}
cloud = SquareCloud(Nx=Nx, Ny=Ny, facet_types=facet_types, noise_key=key, support_size=SUPPORT_SIZE)


# %%

def my_diff_operator_u(x, center=None, rbf=None, monomial=None, fields=None):
    u_val = nodal_value(x, center, rbf, monomial)
    return u_val

def my_rhs_operator_u(x, centers=None, rbf=None, fields=None):
    u_old, v_tilde = fields[:, 0], fields[:, 1]
    u_val = value(x, u_old, centers, rbf)
    v_val = value(x, v_tilde, centers, rbf)
    u_lap = laplacian(x, u_old, centers, rbf)
    return Du*u_lap - u_val*v_val**2 + F*(1 - u_val)

def my_diff_operator_v(x, center=None, rbf=None, monomial=None, fields=None):
    v_val = nodal_value(x, center, rbf, monomial)
    return v_val

def my_rhs_operator_v(x, centers=None, rbf=None, fields=None):
    u_tilde, v_old = fields[:, 0], fields[:, 1]
    u_val = value(x, u_tilde, centers, rbf)
    v_val = value(x, v_old, centers, rbf)
    v_lap = laplacian(x, v_old, centers, rbf)
    return Dv*v_lap + u_val*v_val**2 - (F + k)*v_val


d_zero = lambda x: 0.
boundary_conditions_u = {"South":d_zero, "West":d_zero, "North":d_zero, "East":d_zero}
boundary_conditions_v = {"South":d_zero, "West":d_zero, "North":d_zero, "East":d_zero}



# %% [markdown]
# ## Initial conditions
# $$ u(x, y, t=0) = 1 - \epsilon_1(x, y) $$
# $$ v(x, y, t=0) = \epsilon_2(x, y) $$
# the epsilon are different for each blob

# %%

u0 = 0.95 * jnp.ones(Nx*Ny)
v0 = 0.05 * jnp.ones(Nx*Ny)
n_block = 3
for k in range(n_block):
    ## Pick a central point coordinate
    x_min, y_min = jax.random.uniform(jax.random.PRNGKey(time.time_ns()), (2,), minval=0., maxval=0.8)
    x_max, y_max = x_min + 0.1, y_min + 0.1
    coords = cloud.sorted_nodes

    u0 = jnp.where((coords[:, 0] > x_min) & (coords[:, 0] < x_max) & (coords[:, 1] > y_min) & (coords[:, 1] < y_max), 0., u0)
    v0 = jnp.where((coords[:, 0] > x_min) & (coords[:, 0] < x_max) & (coords[:, 1] > y_min) & (coords[:, 1] < y_max), 1., v0)

## Begin timestepping for 100 steps
fig, ax = plt.subplots(1, 2, figsize=(6*2, 6))

cloud.visualize_field(u0, cmap="gist_ncar", title=f"u Step {0}", vmin=0, vmax=1, colorbar=False, ax=ax[0]);
cloud.visualize_field(u0, cmap="gist_ncar", title=f"v Step {0}", vmin=0, vmax=1,colorbar=False, ax=ax[1]);

plt.draw()


# %%


def gray_scott_vector_field(t, U, *args):
    u, v = jnp.split(U, 2)
    sol_prev = [u, v]
    sol_now = pde_multi_solver_unbounded(
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
    u_t, v_t = [s.vals for s in sol_now]
    return jnp.concatenate([u_t, v_t])



start = time.time()

U0 = jnp.concatenate([u0, v0])

t_span = [0, 400]
t_eval = jnp.linspace(t_span[0], t_span[1], NB_TIMESTEPS+1)

# subdivisions = 5
# Us = RK4(gray_scott_vector_field, t_span=t_span, y0=U0, t_eval=t_eval, subdivisions=subdivisions)



# use diffrax instead, with the DoPri5 integrator
solution = diffrax.diffeqsolve(diffrax.ODETerm(gray_scott_vector_field),
                               diffrax.Tsit5(),
                            #    args=(selected_params),
                               t0=t_span[0],
                               t1=t_span[1],
                               dt0=1e-1,
                               y0=U0,
                               stepsize_controller=diffrax.PIDController(rtol=1e-1, atol=1e-2),
                               saveat=diffrax.SaveAt(ts=t_eval),
                               max_steps=4096*1)
Us = solution.ys



walltime = time.time() - start

print(Us)

minutes = walltime // 60 % 60
seconds = walltime % 60
print(f"Walltime: {minutes} minutes {seconds:.2f} seconds")


# %%
ulist = Us[::NB_TIMESTEPS//100, :Nx*Ny]
filename = DATAFOLDER + "gray_scott_rk4_u.mp4"
cloud.animate_fields([ulist], cmaps="gist_ncar", filename=filename, levels=100, duration=10, figsize=(7.5,6), titles=["Gray-Scott with RBFs - u"]);

# U_norm = Us[::NB_TIMESTEPS*subdivisions//100].reshape(-1, 2, Nx*Ny)
# normlist = jnp.linalg.norm(U_norm, axis=1)
# filename = DATAFOLDER + "gray_scott_rk4_norm.mp4"
# cloud.animate_fields([normlist], cmaps="gist_ncar", filename=filename, levels=100, duration=10, figsize=(7.5,6), titles=["Gray-Scott with RBFs - norm"]);

