# %%
"""
Control of Navier-Stokes equation with differentiable physics
"""

import jax
import jax.numpy as jnp
from jax.tree_util import Partial
import optax
from functools import partial
from tqdm import tqdm
jax.config.update('jax_platform_name', 'cpu')           ## TODO Slow on GPU on Daffy Duck !``

from updec import *

# %%


# EXPERIMENET_ID = random_name()
EXPERIMENET_ID = "ChannelDiffPhys"
DATAFOLDER = "../data/" + EXPERIMENET_ID +"/"
# make_dir(DATAFOLDER)


# %%
### Constants for the problem

RBF = polyharmonic      ## Can define which rbf to use
MAX_DEGREE = 1

Re = 100
Pa = 0.

NB_ITER = 3


# %%

facet_types_vel = {"Wall":"d", "Inflow":"d", "Outflow":"n", "Blowing":"d", "Suction":"d"}
facet_types_phi = {"Wall":"n", "Inflow":"n", "Outflow":"d", "Blowing":"n", "Suction":"n"}

cloud_vel = GmshCloud(filename="../meshes/channel_blowing_suction.py", facet_types=facet_types_vel, mesh_save_location=DATAFOLDER)
cloud_phi = GmshCloud(filename=DATAFOLDER+"mesh.msh", facet_types=facet_types_phi)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6,3*2), sharex=True)
cloud_vel.visualize_cloud(ax=ax1, s=1, title="Cloud for velocity", xlabel=False);
cloud_phi.visualize_cloud(ax=ax2, s=1, title=r"Cloud for $\phi$");

# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.5*2,5))
# cloud_vel.visualize_normals(ax=ax1, title="Normals for velocity")
# cloud_phi.visualize_normals(ax=ax2,title="Normals for phi", zoom_region=(0.25,1.25,-0.1,1.1));


# %%


# @Partial(jax.jit, static_argnums=[2,3])
def diff_operator_u(x, center=None, rbf=None, monomial=None, fields=None):
    U_prev = jnp.array([fields[0], fields[1]])
    u_grad = nodal_gradient(x, center, rbf, monomial)
    u_lap = nodal_laplacian(x, center, rbf, monomial)
    return jnp.dot(U_prev, u_grad) - u_lap/Re

# @Partial(jax.jit, static_argnums=[2])
def rhs_operator_u(x, centers=None, rbf=None, fields=None):
    grad_px = gradient(x, fields[:, 0], centers, rbf)[0]
    return -grad_px



# @Partial(jax.jit, static_argnums=[2,3])
def diff_operator_v(x, center=None, rbf=None, monomial=None, fields=None):
    U_prev = jnp.array([fields[0], fields[1]])
    v_grad = nodal_gradient(x, center, rbf, monomial)
    v_lap = nodal_laplacian(x, center, rbf, monomial)
    return jnp.dot(U_prev, v_grad) - v_lap/Re

# @Partial(jax.jit, static_argnums=[2])
def rhs_operator_v(x, centers=None, rbf=None, fields=None):
    grad_py = gradient(x, fields[:, 0], centers, rbf)[1]
    return  -grad_py



# @Partial(jax.jit, static_argnums=[2,3])
def diff_operator_phi(x, center=None, rbf=None, monomial=None, fields=None):
    return nodal_laplacian(x, center, rbf, monomial)

# @Partial(jax.jit, static_argnums=[2])
def rhs_operator_phi(x, centers=None, rbf=None, fields=None):
    return divergence(x, fields[:, :2], centers, rbf)




# @Partial(jax.jit, static_argnums=[2])     ## TODO FIX THIS
def simulate_forward_navier_stokes(cloud_vel, 
                                    cloud_phi, 
                                    inflow_control=None, 
                                    # Re=Re, Pa=Pa, 
                                    NB_ITER=NB_ITER, 
                                    RBF=RBF, MAX_DEGREE=MAX_DEGREE): 
    """ Simuulates a forward Navier Stokes problem using an iterative approach """




    ## Initial states, all defined on cloud_vel
    u = jnp.zeros((cloud_vel.N,))
    v = jnp.zeros((cloud_vel.N,))

    p_ = jnp.zeros((cloud_phi.N,))       ## on cloud_phi        ##TODO set this to p_a on Outlet
    out_nodes_p = jnp.array(cloud_phi.facet_nodes["Outflow"])
    p_ = p_.at[out_nodes_p].set(Pa)


    parabolic = jax.jit(lambda x: 4*x[1]*(1.-x[1]))
    blowing = jax.jit(lambda x: 0.3)
    suction = jax.jit(lambda x: 0.3)
    zero = jax.jit(lambda x: 0.)

    in_nodes_vel = jnp.array(cloud_vel.facet_nodes["Inflow"])
    if inflow_control != None:
        u_inflow = inflow_control
    else:
        print("WARNING: Input velocity not provided, using parabolic profile")
        u_inflow = jax.vmap(parabolic)(cloud_vel.sorted_nodes[in_nodes_vel])


    bc_u = {"Wall":zero, "Inflow":u_inflow, "Outflow":zero, "Blowing":zero, "Suction":zero}
    bc_v = {"Wall":zero, "Inflow":zero, "Outflow":zero, "Blowing":blowing, "Suction":suction}
    bc_phi = {"Wall":zero, "Inflow":zero, "Outflow":zero, "Blowing":zero, "Suction":zero}


    u_list = [u]
    v_list = [v]
    vel_list = [jnp.linalg.norm(jnp.stack([u,v], axis=-1), axis=-1)]
    p_list = [p_]

    for i in tqdm(range(NB_ITER)):

        p = interpolate_field(p_, cloud_phi, cloud_vel)

        usol = pde_solver_jit(diff_operator=diff_operator_u, 
                        diff_args=[u, v],
                        rhs_operator = rhs_operator_u, 
                        rhs_args=[p], 
                        cloud = cloud_vel, 
                        boundary_conditions = bc_u,
                        rbf=RBF,
                        max_degree=MAX_DEGREE)

        vsol = pde_solver_jit(diff_operator=diff_operator_v,
                        diff_args=[u, v],
                        rhs_operator = rhs_operator_v,
                        rhs_args=[p], 
                        cloud = cloud_vel, 
                        boundary_conditions = bc_v,
                        rbf=RBF,
                        max_degree=MAX_DEGREE)

        ustar , vstar = usol.vals, vsol.vals     ## Star
        Ustar = jnp.stack([ustar,vstar], axis=-1)

        u_ = interpolate_field(ustar, cloud_vel, cloud_phi)
        v_ = interpolate_field(vstar, cloud_vel, cloud_phi)

        phisol_ = pde_solver_jit(diff_operator=diff_operator_phi,
                        rhs_operator = rhs_operator_phi,
                        rhs_args=[u_,v_], 
                        cloud = cloud_phi, 
                        boundary_conditions = bc_phi,
                        rbf=RBF,
                        max_degree=MAX_DEGREE)

        p_ = p_ + phisol_.vals
        gradphi_ = gradient_vec(cloud_phi.sorted_nodes, phisol_.coeffs, cloud_phi.sorted_nodes, RBF)    ## TODO don't use this
        # gradphi_ = cartesian_gradient_vec(range(cloud_phi.N), phisol_.vals, cloud_phi)    

        gradphi = interpolate_field(gradphi_, cloud_phi, cloud_vel)

        U = Ustar - gradphi
        u, v = U[:,0], U[:,1]
        vel = jnp.linalg.norm(U, axis=-1)

        # print("Maximums of u and v:", jnp.max(u), jnp.max(v))

        u_list.append(u)
        v_list.append(v)
        vel_list.append(vel)
        p_list.append(p_)

    return u_list, v_list, vel_list, p_list


# %%

# test_u_inflow = jnp.array([6.48059152e-05, -1.91390260e-05, -4.44326238e-05, -7.00814192e-03, -1.10402573e-01,  4.75611668e-05,  6.73281570e-05])

# test_u_inflow = jnp.zeros(len(cloud_vel.facet_nodes["Inflow"]))


# print(f"\nStarting RBF simulation with {cloud_vel.N} nodes\n")
# u_list, v_list, vel_list, p_list = simulate_forward_navier_stokes(cloud_vel, cloud_phi, inflow_control=test_u_inflow)

# print("\nSimulation complete. Saving all files to %s" % DATAFOLDER)

# renum_map_vel = jnp.array(list(cloud_vel.renumbering_map.keys()))
# renum_map_p = jnp.array(list(cloud_phi.renumbering_map.keys()))

# jnp.savez(DATAFOLDER+'u.npz', renum_map_vel, jnp.stack(u_list, axis=0))
# jnp.savez(DATAFOLDER+'v.npz', renum_map_vel, jnp.stack(v_list, axis=0))
# jnp.savez(DATAFOLDER+'vel.npz', renum_map_vel, jnp.stack(vel_list, axis=0))
# jnp.savez(DATAFOLDER+'p.npz', renum_map_p, jnp.stack(p_list, axis=0))


# %%


# print("\nSaving complete. Now running visualisation ...")

# pyvista_animation(DATAFOLDER, "vel", duration=5, vmin=0.0, vmax=1.08)
# pyvista_animation(DATAFOLDER, "p", duration=5, vmin=-0.15, vmax=0.45)

# # pyvista_animation(DATAFOLDER, "vel", duration=5)
# # pyvista_animation(DATAFOLDER, "p", duration=5)

# vel_arr_prt = jnp.stack(vel_list, axis=0)
# print("max and min vels:", jnp.max(vel_arr_prt), jnp.min(vel_arr_prt))

# %%


















# %%

### Direct adjoitn Looping 


## Constants
LR = 1e-1
GAMMA = 0.995
EPOCHS = 200      ## More than enough for 50 iter and 360 nodes


out_nodes_vel = jnp.array(cloud_vel.facet_nodes["Outflow"])
in_nodes_p = jnp.array(cloud_phi.facet_nodes["Inflow"])

y_in = cloud_phi.sorted_nodes[in_nodes_p, 1]
y_out = cloud_vel.sorted_nodes[out_nodes_vel, 1]

zero = jax.jit(lambda x: 0.)
parabolic = jax.jit(lambda x: 4*x[1]*(1.-x[1]))
u_parab = jax.vmap(parabolic)(cloud_vel.sorted_nodes[out_nodes_vel])
u_zero = jax.vmap(zero)(cloud_vel.sorted_nodes[out_nodes_vel])

def cost_val_fn(u, v):
    u_out = u[out_nodes_vel]
    v_out = v[out_nodes_vel]

    integrand = (u_out-u_parab)**2 + v_out**2
    return 0.5 * jnp.trapz(integrand, x=y_out)

forward_sim_args = {"cloud_vel":cloud_vel,
                    "cloud_phi": cloud_phi,
                    "inflow_control":None,
                    "NB_ITER":NB_ITER,
                    "RBF":RBF,
                    "MAX_DEGREE":MAX_DEGREE    
                    }

@jax.jit
def loss_fn(u_inflow):
    forward_sim_args["inflow_control"] = u_inflow
    u_list, v_list, _, _ = simulate_forward_navier_stokes(**forward_sim_args)

    # print("I GOT CALLED WITH u_inflow:\n", u_inflow, "\n and u_final is:\n", u_list[-1])

    return cost_val_fn(u_list[-1], v_list[-1])

grad_loss_fn = jax.value_and_grad(loss_fn)





los_fn_vec = jax.vmap(loss_fn, in_axes=0, out_axes=0)

@jax.jit
def new_grad_loss_fn(u_inflow):
    t = 1e-6
    # n = u_inflow.shape[0]
    h = jnp.diag(jnp.ones_like(u_inflow))

    # grad = jnp.zeros_like(u_inflow)
    # for i in range(u_inflow.shape[0]):
    #     part[i] = loss_fn(u_inflow + t*h[i]) - loss_fn(u_inflow - t*h[i])

    val = loss_fn(u_inflow)

    # grad = (los_fn_vec(u_inflow + t*h) - los_fn_vec(u_inflow - t*h)) / (2*t)

    grad = (los_fn_vec(u_inflow + t*h) - val*jnp.ones_like(u_inflow)) / (t)

    # print("I GOT CALLED WITH\n", u_inflow, "\nand val is\n", val, "\nand grad is\n", grad)

    return val, grad



optimal_u_inflow = jnp.zeros(in_nodes_p.shape)       ## Optimised quantity
scheduler = optax.piecewise_constant_schedule(init_value=LR,
                                            boundaries_and_scales={int(EPOCHS*0.4):0.1, int(EPOCHS*0.8):0.1})
optimiser = optax.adam(learning_rate=scheduler)
opt_state = optimiser.init(optimal_u_inflow)

history_cost = []

for step in range(1, EPOCHS+1):

    loss, grad = grad_loss_fn(optimal_u_inflow)
    # loss, grad = new_grad_loss_fn(optimal_u_inflow)

    # learning_rate = LR * (GAMMA**step)
    # optimal_u_inflow = optimal_u_inflow - grad * learning_rate

    updates, opt_state = optimiser.update(grad, opt_state, optimal_u_inflow)
    optimal_u_inflow = optax.apply_updates(optimal_u_inflow, updates)

    history_cost.append(loss)

    if step<=3 or step%10==0:
        print("\nEpoch: %-5d  InitLR: %.4f    Loss: %.10f    GradNorm: %.4f" % (step, LR, loss, jnp.linalg.norm(grad)))
        # print("\nEpoch: %-5d  LR: %.4f    Loss: %.10f    GradNorm: %.4f" % (step, learning_rate, loss, jnp.linalg.norm(grad)))

        print("Optimized inflow vel:", optimal_u_inflow)
        plot(optimal_u_inflow, y_in, "--", label="Optimised DP", xlim=None, title=f"Inflow velocity");      ## TODO put a xlim to this !

        plt.show()


#%%
plot(history_cost, label='Cost objective', x_label='epochs', title="Loss", y_scale="log");


#%%

print("\nOptimisation complete. Now running final simulation for visualisation")

forward_sim_args["inflow_control"] = optimal_u_inflow
u_list, v_list, vel_list, p_list = simulate_forward_navier_stokes(**forward_sim_args)
parab_error = jnp.mean((u_list[-1][out_nodes_vel] - u_parab)**2)


fig, ax1 = plt.subplots(1,1, figsize=(6*1,5))

plot(u_parab, y_out, "-", label=r"$u$ target", y_label=r"$y$", xlim=(-0.1, 1.1), figsize=(5,3), ax=ax1)
plot(u_list[-1][out_nodes_vel], y_out, "--", label=r"$u$ DP", ax=ax1, title=f"Outlet velocity / MSE = {parab_error:.4f}");
plot(u_zero, y_out, "-", label=r"$v$ target", y_label=r"$y$", ax=ax1)
plot(v_list[-1][out_nodes_vel], y_out, "--", label=r"$v$ DP", ax=ax1);


#%%

print("\nFinal simulation complete. Now running visualisation ...")


renum_map_vel = jnp.array(list(cloud_vel.renumbering_map.keys()))
renum_map_p = jnp.array(list(cloud_phi.renumbering_map.keys()))

jnp.savez(DATAFOLDER+'u.npz', renum_map_vel, jnp.stack(u_list, axis=0))
jnp.savez(DATAFOLDER+'v.npz', renum_map_vel, jnp.stack(v_list, axis=0))
jnp.savez(DATAFOLDER+'vel.npz', renum_map_vel, jnp.stack(vel_list, axis=0))
jnp.savez(DATAFOLDER+'p.npz', renum_map_p, jnp.stack(p_list, axis=0))

pyvista_animation(DATAFOLDER, "u", duration=5, vmin=jnp.min(u_list[-1]), vmax=jnp.max(u_list[-1]))
pyvista_animation(DATAFOLDER, "v", duration=5, vmin=jnp.min(v_list[-1]), vmax=jnp.max(v_list[-1]))
pyvista_animation(DATAFOLDER, "vel", duration=5, vmin=jnp.min(vel_list[-1]), vmax=jnp.max(vel_list[-1]))
pyvista_animation(DATAFOLDER, "p", duration=5, vmin=jnp.min(p_list[-1]), vmax=jnp.max(p_list[-1]))

# %%
