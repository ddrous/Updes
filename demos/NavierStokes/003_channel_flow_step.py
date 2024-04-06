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

DT = 5e-2
NB_TIMESTEPS = 500
PLOT_EVERY =100

BETA = 0.
NU = 0.01


facet_types_vel = {"Wall1":"d", "Wall2":"d", "Wall3":"d", "Wall4":"d", "Inflow":"d", "Outflow":"n"}
facet_types_phi = {"Wall1":"n", "Wall2":"n", "Wall3":"n", "Wall4":"n", "Inflow":"n", "Outflow":"d"}

cloud_vel = GmshCloud(filename="./meshes/channel_step.py", facet_types=facet_types_vel, support_size="max")
cloud_pre = GmshCloud(filename="./meshes/mesh.msh", facet_types=facet_types_phi, support_size="max")



fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.5,2.0*2), sharex=True)
cloud_vel.visualize_cloud(ax=ax1, s=1, title="Cloud for velocity", xlabel=False);
cloud_pre.visualize_cloud(ax=ax2, s=1, title=r"Cloud for $\phi$");

print("Total number of nodes:", cloud_vel.N)

parabolic = jax.jit(lambda x: 1.5 - 6*(x[1]**2))
zero = jax.jit(lambda x: 0.0)
one = jax.jit(lambda x: -0.25)

bc_u = {"Wall1":zero, "Wall2":zero, "Wall3":zero, "Wall4":zero, "Inflow":parabolic, "Outflow":zero}
bc_v = {"Wall1":zero, "Wall2":zero, "Wall3":zero, "Wall4":zero, "Inflow":zero, "Outflow":zero}
bc_phi = {"Wall1":zero, "Wall2":zero, "Wall3":zero, "Wall4":zero, "Inflow":zero, "Outflow":zero}


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
    ret_val = u_val - DT*(u_val*u_grad[0] + v_val*u_grad[1] + BETA*p_grad[0] -NU*u_lap + f_u_val)
    return jnp.clip(ret_val, -1e-2, 1e-2)

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
    ret_val = v_val - DT*(u_val*v_grad[0] + v_val*v_grad[1] + BETA*p_grad[1] -NU*v_lap + f_v_val)
    return jnp.clip(ret_val, -1e-2, 1e-2)

def my_diff_operator_p(x, center=None, rbf=None, monomial=None, fields=None):
    p_lap = nodal_laplacian(x, center, rbf, monomial)
    return p_lap

def my_rhs_operator_p(x, centers=None, rbf=None, fields=None):
    U_prev = fields[:, 0:2]
    # U_div = divergence(x, U_prev, centers, rbf)
    U_div = jnp.clip(divergence(x, U_prev, centers, rbf), -1e-2, 1e-2)
    return U_div / DT




# %%
## Uo is a 2D gaussian centered at the middle of the domain
u0 = jnp.zeros((cloud_vel.N,))
v0 = jnp.zeros((cloud_vel.N,))
p0 = jnp.zeros((cloud_pre.N,))

## Make two plots horizontal for uo and vo
# fig, ax = plt.subplots(1, 2, figsize=(12,6))

## Begin timestepping for 100 steps
# cloud_vel.visualize_field(u0, cmap="coolwarm", title=f"u{0}", vmin=0, vmax=1, figsize=(6,6),colorbar=False, ax=ax[0]);
# cloud_vel.visualize_field(v0, cmap="coolwarm", title=f"v{0}", vmin=0, vmax=1, figsize=(6,6),colorbar=False, ax=ax[1]);


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
                    boundary_conditions = bc_u, 
                    rbf=RBF,
                    max_degree=MAX_DEGREE,)
    ustar = ufield.vals

    ## Compute v* with the PDE solver
    vfield = pde_solver_jit(diff_operator=my_diff_operator_v,
                    rhs_operator = my_rhs_operator_v,
                    rhs_args=[u_prev, v_prev, p_prev_, f_v],
                    cloud = cloud_vel,
                    boundary_conditions = bc_v, 
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
                    boundary_conditions = bc_phi, 
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
        fig, ax = plt.subplots(2, 1, figsize=(9.5,2.0*2))
        # plt.cla()
        # cloud.visualize_field(ulist[-1], cmap="coolwarm", projection="3d", title=f"Step {i}")
        _, _ = cloud_vel.visualize_field(sol_list[-1][0], cmap="coolwarm", title=f"u utep {i}", vmin=None, vmax=None, colorbar=True, ax=ax[0])
        _, _ = cloud_vel.visualize_field(sol_list[-1][1], cmap="coolwarm", title=f"v utep {i}", vmin=None, vmax=None, colorbar=True, ax=ax[1])
        # plt.draw()
        plt.show()


walltime = time.time() - start

minutes = walltime // 60 % 60
seconds = walltime % 60
print(f"Walltime: {minutes} minutes {seconds:.2f} seconds")



# %%

normlist = [jnp.nan_to_num(jnp.sqrt(sol[0]**2 + sol[1]**2)) for sol in sol_list[::20]]
# normlist = [jnp.sqrt(sol[0]**2 + sol[1]**2) for sol in sol_list[::20]]
filename = DATAFOLDER + "ns_norm_channel.gif"
cloud_vel.animate_fields([normlist], cmaps="coolwarm", filename=filename, levels=50, duration=10, figsize=(9.5,2.0*1), titles=["Navier Stokes with RBFs - norm"]);


# %%


