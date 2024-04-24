# %%

"""
Case from: Perspectives in Flow Control and Optimization (Gunzburger 2003).pdf [Page 77]
Numerical scheme from: Numerical methods for the Navier Stokes equations (Hans Petter Langtangen, 2012), [Page 5]]
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
MAX_DEGREE = 1

DT = 1e-3
NB_TIMESTEPS = 200
PLOT_EVERY = 20

BETA = 0.
NU = 0.01

## Diffusive constant

Nx = 20
Ny = 20
SUPPORT_SIZE = "max"

facet_types_vel={"South":"d", "North":"d", "West":"d", "East":"d"}
facet_types_pre={"South":"n", "North":"n", "West":"n", "East":"n"}
cloud_vel = SquareCloud(Nx=Nx, Ny=Ny, facet_types=facet_types_vel, noise_key=key, support_size=SUPPORT_SIZE)
cloud_pre = SquareCloud(Nx=Nx, Ny=Ny, facet_types=facet_types_pre, noise_key=key, support_size=SUPPORT_SIZE)


# %%

def my_diff_operator_u(x, center=None, rbf=None, monomial=None, fields=None):
    u_val = nodal_value(x, center, rbf, monomial)
    return u_val

def my_rhs_operator_u(x, centers=None, rbf=None, fields=None):
    u_prev, v_prev, p_rev, f_u_prev = fields[:, 0], fields[:, 1], fields[:, 2], fields[:, 3]
    u_val = value(x, u_prev, centers, rbf)
    v_val = value(x, v_prev, centers, rbf)
    u_grad = gradient(x, u_prev, centers, rbf)
    p_grad = gradient(x, p_rev, centers, rbf)
    u_lap = laplacian(x, u_prev, centers, rbf)
    f_u_val = value(x, f_u_prev, centers, rbf)
    return u_val - DT*(u_val*u_grad[0] + v_val*u_grad[1] + BETA*p_grad[0] -NU*u_lap + f_u_val)

def my_diff_operator_v(x, center=None, rbf=None, monomial=None, fields=None):
    v_val = nodal_value(x, center, rbf, monomial)
    return v_val

def my_rhs_operator_v(x, centers=None, rbf=None, fields=None):
    u_prev, v_prev, p_rev, f_v_prev = fields[:, 0], fields[:, 1], fields[:, 2], fields[:, 3]
    u_val = value(x, u_prev, centers, rbf)
    v_val = value(x, v_prev, centers, rbf)
    v_grad = gradient(x, v_prev, centers, rbf)
    p_grad = gradient(x, p_rev, centers, rbf)
    v_lap = laplacian(x, v_prev, centers, rbf)
    f_v_val = value(x, f_v_prev, centers, rbf)
    return v_val - DT*(u_val*v_grad[0] + v_val*v_grad[1] + BETA*p_grad[1] -NU*v_lap + f_v_val)

def my_diff_operator_p(x, center=None, rbf=None, monomial=None, fields=None):
    p_lap = nodal_laplacian(x, center, rbf, monomial)
    return p_lap

def my_rhs_operator_p(x, centers=None, rbf=None, fields=None):
    U_prev = fields[:, 0:2]
    # U_div = divergence(x, U_prev, centers, rbf)
    U_div = jnp.clip(divergence(x, U_prev, centers, rbf), -1e-2, 1e-2)
    return U_div / DT

d_zero = lambda x: 0.
boundary_conditions = {"South":d_zero, "West":d_zero, "North":d_zero, "East":d_zero}



# %%
## Uo is a 2D gaussian centered at the middle of the domain
xy = cloud_vel.sorted_nodes
def Phi_z(z):
    return (1.-jnp.cos(0.8*jnp.pi*z))*(1.-z)**2

def Phi_x_Phi_y(x, y):
    return Phi_z(x)*Phi_z(y)

u0 = 10 * jax.vmap(jax.grad(Phi_x_Phi_y, 1))(xy[:,0], xy[:,1])
v0 = -10 * jax.vmap(jax.grad(Phi_x_Phi_y, 0))(xy[:,0], xy[:,1])
p0 = jnp.zeros_like(u0)


## Make two plots horizontal for uo and vo
fig, ax = plt.subplots(1, 2, figsize=(12,6))

## Begin timestepping for 100 steps
cloud_vel.visualize_field(u0, cmap="coolwarm", title=f"u{0}", vmin=0, vmax=1, figsize=(6,6),colorbar=False, ax=ax[0]);
cloud_vel.visualize_field(v0, cmap="coolwarm", title=f"v{0}", vmin=0, vmax=1, figsize=(6,6),colorbar=False, ax=ax[1]);


# %%
sol_list = [[u0, v0, p0]]

start = time.time()

for i in range(1, NB_TIMESTEPS+1):
    u_prev, v_prev, p_prev = sol_list[-1]
    f_u = jnp.zeros_like(u_prev)
    f_v = jnp.zeros_like(v_prev)

    p_prev_ = interpolate_field(p_prev, cloud_pre, cloud_vel)

    ## Compute u* with the PDE solver
    ufield = pde_solver_jit(diff_operator=my_diff_operator_u,
                    rhs_operator = my_rhs_operator_u,
                    rhs_args=[u_prev, v_prev, p_prev_, f_u],
                    cloud = cloud_vel,
                    boundary_conditions = boundary_conditions, 
                    rbf=RBF,
                    max_degree=MAX_DEGREE,)
    ustar = ufield.vals

    ## Compute v* with the PDE solver
    vfield = pde_solver_jit(diff_operator=my_diff_operator_v,
                    rhs_operator = my_rhs_operator_v,
                    rhs_args=[u_prev, v_prev, p_prev_, f_v],
                    cloud = cloud_vel,
                    boundary_conditions = boundary_conditions, 
                    rbf=RBF,
                    max_degree=MAX_DEGREE,)
    vstar = vfield.vals

    ## Compute p* with the PDE solver
    u_star_ = interpolate_field(ustar, cloud_vel, cloud_pre)
    v_star_ = interpolate_field(vstar, cloud_vel, cloud_pre)

    phisol_ = pde_solver_jit(diff_operator=my_diff_operator_p,
                    rhs_operator = my_rhs_operator_p,
                    rhs_args=[u_star_, v_star_],
                    cloud = cloud_pre,
                    boundary_conditions = boundary_conditions, 
                    rbf=RBF,
                    max_degree=MAX_DEGREE,)

    p_next = BETA*p_prev + phisol_.vals

    ## Todo make p_next has zero integral
    # p_next = p_next - integrate_field(p_next, cloud_pre, RBF, MAX_DEGREE)
    p_next = p_next - jnp.mean(p_next)


    gradphi_ = gradient_vec(cloud_pre.sorted_nodes, phisol_.coeffs, cloud_pre.sorted_nodes, RBF)
    ## Limit the gradient of phi
    gradphi_ = jnp.clip(gradphi_, -1e-3, 1e-3)

    ## Interpolate p and gradphi onto cloud_vel
    gradphi = interpolate_field(gradphi_, cloud_pre, cloud_vel)

    u_next = ustar - (gradphi[:,0] *DT)
    v_next = vstar - (gradphi[:,1] *DT)

    ## Pint the max of all quantities
    # print(f"maximum values Step {i}:")
    # print(f"u: {jnp.max(jnp.abs(u_next))}")
    # print(f"v: {jnp.max(jnp.abs(v_next))}")
    # print(f"p: {jnp.max(jnp.abs(p_next))}")
    # print(f"integral p: {integrate_field(p_next, cloud_pre, RBF, MAX_DEGREE)}")

    u_prev, v_prev, p_prev = u_next, v_next, p_next
    sol_now = [u_next, v_next, p_next]
    sol_list.append(sol_now)

    if i<=3 or i%PLOT_EVERY==0:
        print(f"Step {i}")
        fig, ax = plt.subplots(1, 2, figsize=(11,5))
        # plt.cla()
        # cloud.visualize_field(ulist[-1], cmap="coolwarm", projection="3d", title=f"Step {i}")
        _, _ = cloud_vel.visualize_field(sol_list[-1][0], cmap="coolwarm", title=f"u utep {i}", vmin=None, vmax=None, figsize=(6,6), colorbar=True, ax=ax[0])
        _, _ = cloud_vel.visualize_field(sol_list[-1][1], cmap="coolwarm", title=f"v utep {i}", vmin=None, vmax=None, figsize=(6,6), colorbar=True, ax=ax[1])
        # plt.draw()
        plt.show()


walltime = time.time() - start

minutes = walltime // 60 % 60
seconds = walltime % 60
print(f"Walltime: {minutes} minutes {seconds:.2f} seconds")



# %%

# ulist = [sol[0] for sol in sol_list]
# filename = DATAFOLDER + "burgers_u.mp4"
# cloud_vel.animate_fields([ulist], cmaps="coolwarm", filename=filename, levels=200, duration=10, figsize=(7.5,6), titles=["Burgers with RBFs - u"]);

normlist = [jnp.sqrt(sol[0]**2 + sol[1]**2) for sol in sol_list]
filename = DATAFOLDER + "ns_norm.gif"
cloud_vel.animate_fields([normlist[::10]], cmaps="coolwarm", filename=filename, levels=200, duration=10, figsize=(7.5,6), titles=["Navier-Stokes with RBFs"]);


# %%


