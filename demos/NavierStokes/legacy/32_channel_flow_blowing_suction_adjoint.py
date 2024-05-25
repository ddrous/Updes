
# %%
import jax
import jax.numpy as jnp
from jax.tree_util import Partial
from functools import partial
import optax
from tqdm import tqdm
jax.config.update('jax_platform_name', 'cpu')           ## TODO Slow on GPU on Daffy Duck !

from updes import *
import sys

# EXPERIMENET_ID = random_name()
EXPERIMENET_ID = "ChannelAdjoint2"
DATAFOLDER = "../data/" + EXPERIMENET_ID +"/"
# make_dir(DATAFOLDER)



### Constants for the problem
# EPS = 10.0
# RBF = partial(gaussian, eps=EPS)
RBF = polyharmonic
MAX_DEGREE = 1

Re = 100        ## Make sure the same constants are used for the forward problem
Pa = 0.

NB_ITER = 10    ## 50 works for 360 nodes (lc=0.2, ref_io=2, ref_bs=5)


# %%


facet_types_lamb = {"Wall":"d", "Inflow":"d", "Outflow":"r", "Blowing":"d", "Suction":"d"}
facet_types_mu = {"Wall":"n", "Inflow":"n", "Outflow":"d", "Blowing":"n", "Suction":"n"}

cloud_lamb = GmshCloud(filename="../meshes/channel_blowing_suction.py", facet_types=facet_types_lamb, mesh_save_location=DATAFOLDER)
cloud_mu = GmshCloud(filename=DATAFOLDER+"mesh.msh", facet_types=facet_types_mu)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6,3*2), sharex=True)
cloud_lamb.visualize_cloud(ax=ax1, s=6, title=r"Cloud for $\lambda$", xlabel=False);
cloud_mu.visualize_cloud(ax=ax2, s=6, title=r"Cloud for $\mu$");

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.5*2,5))
cloud_lamb.visualize_normals(ax=ax1, title="Normals for lambda", zoom_region=(1.4,1.6,-0.1,1.1));
cloud_mu.visualize_normals(ax=ax2,title="Normals for mu", zoom_region=(0.4,0.6,-0.1,0.1));



# %%

def diff_operator_l1(x, center=None, rbf=None, monomial=None, fields=None):
    div_U = fields[2]
    l1_val = nodal_value(x, center, rbf, monomial)
    U_val = jnp.array([fields[0], fields[1]])
    l1_grad = nodal_gradient(x, center, rbf, monomial)
    l1_lap = nodal_laplacian(x, center, rbf, monomial)
    return div_U*l1_val + jnp.dot(U_val, l1_grad) + l1_lap/Re

def rhs_operator_l1(x, centers=None, rbf=None, fields=None):
    grad_pi_x = gradient(x, fields[:, 0], centers, rbf)[0]
    return -grad_pi_x



def diff_operator_l2(x, center=None, rbf=None, monomial=None, fields=None):
    div_U = fields[2]
    l2_val = nodal_value(x, center, rbf, monomial)
    U_val = jnp.array([fields[0], fields[1]])
    l2_grad = nodal_gradient(x, center, rbf, monomial)
    l2_lap = nodal_laplacian(x, center, rbf, monomial)
    return div_U*l2_val + jnp.dot(U_val, l2_grad) + l2_lap/Re

def rhs_operator_l2(x, centers=None, rbf=None, fields=None):
    grad_pi_y = gradient(x, fields[:, 0], centers, rbf)[1]
    return -grad_pi_y



def diff_operator_mu(x, center=None, rbf=None, monomial=None, fields=None):
    return nodal_laplacian(x, center, rbf, monomial)

def rhs_operator_mu(x, centers=None, rbf=None, fields=None):
    return divergence(x, fields[:, :2], centers, rbf)      ## TODO check this minus



# @Partial(jax.jit, static_argnums=[2,3])       ## TODO FIX THIS
def simulate_adjoint_navier_stokes(cloud_lamb, 
                                    cloud_mu, 
                                    cloud_vel=None,
                                    u=None, v=None,
                                    NB_ITER=NB_ITER,
                                    RBF=RBF, MAX_DEGREE=MAX_DEGREE):

    """ Simulates a adjoint Navier Stokes problem using an iterative approach """


    lambda1 = jnp.zeros((cloud_lamb.N,))
    lambda2 = jnp.zeros((cloud_lamb.N,))
    # out_nodes_lamb = jnp.array(cloud_lamb.facet_nodes["Outflow"])

    pi_ = jnp.zeros((cloud_mu.N,))       ## on cloud_mu        ##TODO set this to p_a on Outlet
    out_nodes_pi = jnp.array(cloud_mu.facet_nodes["Outflow"])
    pi_ = pi_.at[out_nodes_pi].set(Pa)


    parabolic = jax.jit(lambda x: 4*x[1]*(1.-x[1]))
    zero = jax.jit(lambda x: 0.)


    ## Quantities for Robin boundary condition
    if cloud_vel == None:
        print("WARNING: no cloud for forward grad computation. Using cloud for lambda ")
        cloud_vel = cloud_lamb
        if u != None or v != None:
            print("ERROR: You've provided a cloud for velocity without an actual velocity !")
            sys.exit(1)
        if u == None:
            print("WARNING: u velocity not provided, using ones")
            u = jnp.ones((cloud_vel.N))
        if v == None:
            print("WARNING: v velocity not provided, using ones")
            v = jnp.ones((cloud_vel.N))


    out_nodes_vel = jnp.array(cloud_vel.facet_nodes["Outflow"])
    u1 = u[out_nodes_vel]      ## For robin BCs
    u2 = v[out_nodes_vel]
    u_parab = jax.vmap(parabolic)(cloud_vel.sorted_nodes[out_nodes_vel])
    pi_out = pi_[out_nodes_pi]

    ### grad_u = cartesian_gradient_vec(range(cloud_vel.N), u, cloud_vel)
    ### grad_v = cartesian_gradient_vec(range(cloud_vel.N), v, cloud_vel)

    # grad_u = gradient_vals_vec(cloud_vel.sorted_nodes, u, cloud_vel, RBF, MAX_DEGREE)
    # grad_v = gradient_vals_vec(cloud_vel.sorted_nodes, v, cloud_vel, RBF, MAX_DEGREE)

    # grad_u = interpolate_field(grad_u, cloud_vel, cloud_lamb)   ## TODO Check this !
    # grad_v = interpolate_field(grad_v, cloud_vel, cloud_lamb)

    bc_l1 = {"Wall":zero, "Inflow":zero, "Outflow":((u1-u_parab+pi_out)*Re, u1*Re), "Blowing":zero, "Suction":zero}
    bc_l2 = {"Wall":zero, "Inflow":zero, "Outflow":(u2*Re, u1*Re), "Blowing":zero, "Suction":zero}
    bc_mu = {"Wall":zero, "Inflow":zero, "Outflow":zero, "Blowing":zero, "Suction":zero}


    l1_list = [lambda1]
    l2_list = [lambda2]
    lnorm_list = []
    pi_list = [pi_]

    for i in tqdm(range(NB_ITER)):

        pi = interpolate_field(pi_, cloud_mu, cloud_lamb)
        div_U = divergence_vec(cloud_mu.sorted_nodes, jnp.stack([u, v], axis=-1), cloud_mu.sorted_nodes, RBF)
        # print("min max of div_U: ", jnp.min(div_U), jnp.max(div_U))


        l1sol = pde_solver_jit(diff_operator=diff_operator_l1, 
                        diff_args=[u, v, div_U],
                        rhs_operator = rhs_operator_l1, 
                        rhs_args=[pi],
                        cloud = cloud_lamb,
                        boundary_conditions = bc_l1,
                        rbf=RBF,
                        max_degree=MAX_DEGREE)

        l2sol = pde_solver_jit(diff_operator=diff_operator_l2,
                        diff_args=[u, v, div_U],
                        rhs_operator = rhs_operator_l2,
                        rhs_args=[pi], 
                        cloud = cloud_lamb, 
                        boundary_conditions = bc_l2,
                        rbf=RBF,
                        max_degree=MAX_DEGREE)

        l1star, l2star = l1sol.vals, l2sol.vals     ## Star
        Lstar = jnp.stack([l1star,l2star], axis=-1)

        l1_ = interpolate_field(l1star, cloud_lamb, cloud_mu)
        l2_ = interpolate_field(l2star, cloud_lamb, cloud_mu)

        musol_ = pde_solver_jit(diff_operator=diff_operator_mu,
                        rhs_operator = rhs_operator_mu,
                        rhs_args=[l1_, l2_], 
                        cloud = cloud_mu, 
                        boundary_conditions = bc_mu,
                        rbf=RBF,
                        max_degree=MAX_DEGREE)

        pi_ = pi_ + musol_.vals

        gradmu_ = gradient_vec(cloud_mu.sorted_nodes, musol_.coeffs, cloud_mu.sorted_nodes, RBF)
        # gradmu_ = cartesian_gradient_vec(range(cloud_mu.N), musol_.vals, cloud_mu)

        gradmu = interpolate_field(gradmu_, cloud_mu, cloud_lamb)

        L = Lstar - gradmu      ## TODO Minus sign !
        l1, l2 = L[:,0], L[:,1]
        lnorm = jnp.linalg.norm(L, axis=-1)

        # print("Maximums of lambda 1 and 2:", jnp.max(l1), jnp.max(l2))

        l1_list.append(l1)
        l2_list.append(l2)
        lnorm_list.append(lnorm)
        pi_list.append(pi_)

    return l1_list, l2_list, lnorm_list, pi_list


# %%



# print(f"\nStarting RBF simulation with {cloud_lamb.N} nodes\n")
# l1_list, l2_list, lnorm_list, pi_list = simulate_adjoint_navier_stokes(cloud_lamb, cloud_mu)


# print("\nSimulation complete. Saving all files to %s" % DATAFOLDER)

# renum_map_vel = jnp.array(list(cloud_lamb.renumbering_map.keys()))
# renum_map_p = jnp.array(list(cloud_mu.renumbering_map.keys()))

# jnp.savez(DATAFOLDER+'l1.npz', renum_map_vel, jnp.stack(l1_list, axis=0))
# jnp.savez(DATAFOLDER+'l2.npz', renum_map_vel, jnp.stack(l2_list, axis=0))
# jnp.savez(DATAFOLDER+'lnorm.npz', renum_map_vel, jnp.stack(lnorm_list, axis=0))
# jnp.savez(DATAFOLDER+'pi.npz', renum_map_p, jnp.stack(pi_list, axis=0))


# # %%

# print("\nSaving complete. Now running visualisation ...")

# pyvista_animation(DATAFOLDER, "lnorm", duration=5, vmin=None, vmax=None)
# pyvista_animation(DATAFOLDER, "pi", duration=5, vmin=None, vmax=None)














# %%

### Direct adjoitn Looping 


## Constants
LR = 1e-1
GAMMA = 0.995
EPOCHS = 30     ## 3 More than enough for 50 iter and 360 nodes, but at least 4 needed for 314 nodes


## Bluid new clouds for forward problem (different boundary conditions)
facet_types_vel = {"Wall":"d", "Inflow":"d", "Outflow":"n", "Blowing":"d", "Suction":"d"}
facet_types_phi = {"Wall":"n", "Inflow":"n", "Outflow":"d", "Blowing":"n", "Suction":"n"}
cloud_vel = GmshCloud(filename=DATAFOLDER+"mesh.msh", facet_types=facet_types_vel)
cloud_phi = GmshCloud(filename=DATAFOLDER+"mesh.msh", facet_types=facet_types_phi)


## Import forward solver
simulate_forward_navier_stokes = __import__('30_channel_flow_blowing_suction').simulate_forward_navier_stokes

in_nodes_vel = jnp.array(cloud_vel.facet_nodes["Inflow"])
out_nodes_vel = jnp.array(cloud_vel.facet_nodes["Outflow"])
in_nodes_lamb = jnp.array(cloud_lamb.facet_nodes["Inflow"])
in_nodes_pi = jnp.array(cloud_mu.facet_nodes["Inflow"])

y_in = cloud_lamb.sorted_nodes[in_nodes_lamb, 1]
y_out = cloud_vel.sorted_nodes[out_nodes_vel, 1]

zero = jax.jit(lambda x: 0.)
parabolic = jax.jit(lambda x: 4*x[1]*(1.-x[1]))
u_parab = jax.vmap(parabolic)(cloud_vel.sorted_nodes[out_nodes_vel])
u_zero = jax.vmap(zero)(cloud_vel.sorted_nodes[out_nodes_vel])

@jax.jit
def cost_val_fn(u, v, u_parab):
    u_out = u[out_nodes_vel]
    v_out = v[out_nodes_vel]

    integrand = (u_out-u_parab)**2 + v_out**2
    return 0.5 * jnp.trapz(integrand, x=y_out)

@jax.jit
def cost_grad_fn(l1, pi_):
    grad_l1 = gradient_vals_vec(cloud_lamb.sorted_nodes[in_nodes_lamb], l1, cloud_lamb, RBF, MAX_DEGREE)

    # grad_l1 = cartesian_gradient_vec(range(cloud_lamb.N), l1, cloud_lamb)       ## TODO: use the one above
    # grad_l1 = grad_l1[in_nodes_lamb, :]

    return pi_[in_nodes_pi] + grad_l1[:, 0]/Re


forward_sim_args = {"cloud_vel":cloud_vel,
                    "cloud_phi": cloud_phi,
                    "inflow_control":None,
                    # "Re":Re,
                    # "Pa":Pa,
                    "NB_ITER":NB_ITER,
                    "RBF":RBF,
                    "MAX_DEGREE":MAX_DEGREE
                    }
adjoint_sim_args = {"cloud_lamb":cloud_lamb,
                    "cloud_mu": cloud_mu,
                    "u":None,"v":None,
                    "cloud_vel":cloud_vel,
                    "NB_ITER":NB_ITER,
                    "RBF":RBF,
                    "MAX_DEGREE":MAX_DEGREE
                    }


# optimal_u_inflow = jnp.zeros(in_nodes_lamb.shape)       ## Optimised quantity
optimal_u_inflow = jax.vmap(parabolic)(cloud_vel.sorted_nodes[in_nodes_vel])
scheduler = optax.piecewise_constant_schedule(init_value=LR,
                                            boundaries_and_scales={int(EPOCHS*0.4):0.1, int(EPOCHS*0.8):0.1})
optimiser = optax.adam(learning_rate=scheduler)
opt_state = optimiser.init(optimal_u_inflow)

history_cost = []
parab_out_mse = []




for step in range(1, EPOCHS+1):


    forward_sim_args["inflow_control"] = optimal_u_inflow
    u_list, v_list, vel_list, p_list = simulate_forward_navier_stokes(**forward_sim_args)


    # grad_u = cartesian_gradient_vec(range(cloud_vel.N), u, cloud_vel)     ## TODO USE catesian, and interpolate please !
    # grad_v = cartesian_gradient_vec(range(cloud_vel.N), v, cloud_vel)

    # grad_u = gradient_vals_vec(cloud_vel.sorted_nodes, u_list[-1], cloud_vel, RBF, MAX_DEGREE)
    # grad_v = gradient_vals_vec(cloud_vel.sorted_nodes, v_list[-1], cloud_vel, RBF, MAX_DEGREE)


    adjoint_sim_args["u"] = u_list[-1]
    adjoint_sim_args["v"] = v_list[-1]
    l1_list, l2_list, lnorm_list, pi_list = simulate_adjoint_navier_stokes(**adjoint_sim_args)

    ### Optimsation start ###
    loss = cost_val_fn(u_list[-1], v_list[-1], u_parab)
    grad = cost_grad_fn(l1_list[-1], pi_list[-1])

    # learning_rate = LR * (GAMMA**step)
    # optimal_u_inflow = optimal_u_inflow - grad * learning_rate          ## Gradient descent !

    updates, opt_state = optimiser.update(grad, opt_state, optimal_u_inflow)
    optimal_u_inflow = optax.apply_updates(optimal_u_inflow, updates)

    parab_error = jnp.mean((u_list[-1][out_nodes_vel] - u_parab)**2)
    history_cost.append(loss)
    parab_out_mse.append(parab_error)

    if step<=3 or step%5==0:
        print("\nEpoch: %-5d  InitLR: %.4f    Loss: %.10f    GradNorm: %.4f  TestMSE: %.6f" % (step, LR, loss, jnp.linalg.norm(grad), parab_error))


        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(6*2,5))

        plot(u_parab, y_out, "-", label=r"$u$ target", y_label=r"$y$", xlim=(-0.1, 1.1), figsize=(5,3), ax=ax1)
        plot(u_list[-1][out_nodes_vel], y_out, "--", label=r"$u$ DAL", ax=ax1, title=f"Outlet velocity / MSE = {parab_error:.4f}");
        plot(u_zero, y_out, "-", label=r"$v$ target", y_label=r"$y$", ax=ax1)
        plot(v_list[-1][out_nodes_vel], y_out, "--", label=r"$v$ DAL", ax=ax1);

        print("Optimized inflow vel:", optimal_u_inflow)
        plot(optimal_u_inflow, y_in, "--", label="Optimised DAL", ax=ax2, xlim=None, title=f"Inflow velocity");

        plt.show()


#%%
ax = plot(history_cost[:], label='Cost objective', x_label='epochs', title="Loss", y_scale="log");
plot(parab_out_mse[:], label='Test MSE at outlet', x_label='epochs', title="Loss", y_scale="log", ax=ax);


#%%

print("\nOptimisation complete complete. Now running visualisation ...")

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
