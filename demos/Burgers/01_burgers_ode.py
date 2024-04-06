#%%
%load_ext autoreload
%autoreload 2

# %%

"""
Test of the Updec package on the 2D Burgers equation with RBFs: 
PDE here: https://en.wikipedia.org/wiki/Convection%E2%80%93diffusion_equation
"""


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
# \frac{\partial u}{\partial t} = - u\frac{\partial u}{\partial x} - \tilde v \frac{\partial u}{\partial y}
# $$
# $$
# \frac{\partial v}{\partial t} = - \tilde u \frac{\partial v}{\partial x} - v \frac{\partial v}{\partial y}
# $$

# This can go in the rhs of the PDE
# u and v are concatenated in a single vector U = [u, v]
# The tilde is solved in an iterative manner


# %% 
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

# DT = 1e-3
NB_TIMESTEPS = 200
PLOT_EVERY = 2

Nx = 10
Ny = 10
SUPPORT_SIZE = "max"

# facet_types={"South":"d", "North":"d", "West":"d", "East":"d"}
facet_types={"South":"n", "North":"n", "West":"n", "East":"n"}
cloud = SquareCloud(Nx=Nx, Ny=Ny, facet_types=facet_types, noise_key=key, support_size=SUPPORT_SIZE)


# %%

def my_diff_operator_u(x, center=None, rbf=None, monomial=None, fields=None):
    u_val = nodal_value(x, center, rbf, monomial)
    return u_val

def my_rhs_operator_u(x, centers=None, rbf=None, fields=None):
    u_old, v_tilde = fields[:, 0], fields[:, 1]
    u_val = value(x, u_old, centers, rbf)
    v_val = value(x, v_tilde, centers, rbf)
    u_grad = gradient(x, u_old, centers, rbf)
    return -u_val*u_grad[0] - v_val*u_grad[1]

def my_diff_operator_v(x, center=None, rbf=None, monomial=None, fields=None):
    v_val = nodal_value(x, center, rbf, monomial)
    return v_val

def my_rhs_operator_v(x, centers=None, rbf=None, fields=None):
    u_tilde, v_old = fields[:, 0], fields[:, 1]
    u_val = value(x, u_tilde, centers, rbf)
    v_val = value(x, v_old, centers, rbf)
    v_grad = gradient(x, v_old, centers, rbf)
    return -u_val*v_grad[0] - v_val*v_grad[1]


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


def burgers_vector_field(t, U, *args):
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





def RK4(fun, t_span, y0, *args, t_eval=None, subdivisions=1, **kwargs):
    """ Perform numerical integration with a time step divided by the evaluation subdivision factor (Not necessarily equally spaced). If we get NaNs, we can try to increasing the subdivision factor for finer time steps."""
    if t_eval is None:
        if t_span[0] is None:
            t_eval = jnp.array([t_span[1]])
            raise Warning("t_span[0] is None. Setting t_span[0] to 0.")
        elif t_span[1] is None:
            raise ValueError("t_span[1] must be provided if t_eval is not.")
        else:
            t_eval = jnp.array(t_span)

    hs = t_eval[1:] - t_eval[:-1]
    t_ = t_eval[:-1, None] + jnp.arange(subdivisions)[None, :]*hs[:, None]/subdivisions
    t_solve = jnp.concatenate([t_.flatten(), t_eval[-1:]])
    eval_indices = jnp.arange(0, t_solve.size, subdivisions)

    def step(state, t):
        t_prev, y_prev = state
        h = t - t_prev
        k1 = h * fun(t_prev, y_prev, *args)
        k2 = h * fun(t_prev + h/2., y_prev + k1/2., *args)
        k3 = h * fun(t_prev + h/2., y_prev + k2/2., *args)
        k4 = h * fun(t + h, y_prev + k3, *args)
        y = y_prev + (k1 + 2*k2 + 2*k3 + k4) / 6.
        return (t, y), y

    _, ys = jax.lax.scan(step, (t_solve[0], y0), t_solve[:])
    return ys[eval_indices, :]




start = time.time()

U0 = jnp.concatenate([u0, v0])

t_span = [0, 1]
t_eval = jnp.linspace(0, *t_span, NB_TIMESTEPS+1)
Us = RK4(burgers_vector_field, t_span=t_span, y0=U0, t_eval=t_eval, subdivisions=1)

walltime = time.time() - start

print(Us)

minutes = walltime // 60 % 60
seconds = walltime % 60
print(f"Walltime: {minutes} minutes {seconds:.2f} seconds")






# %%
ulist = Us[::NB_TIMESTEPS//10, :Nx*Ny]
filename = DATAFOLDER + "burgers_rk4_u.mp4"
cloud.animate_fields([ulist], cmaps="coolwarm", filename=filename, levels=200, duration=10, figsize=(7.5,6), titles=["Burgers with RBFs - u"]);

U_norm = Us[::NB_TIMESTEPS//10].reshape(-1, 2, Nx*Ny)
normlist = jnp.linalg.norm(U_norm, axis=1)
filename = DATAFOLDER + "burgers_rk4_norm.mp4"
cloud.animate_fields([normlist], cmaps="coolwarm", filename=filename, levels=200, duration=10, figsize=(7.5,6), titles=["Burgers with RBFs - norm"]);
