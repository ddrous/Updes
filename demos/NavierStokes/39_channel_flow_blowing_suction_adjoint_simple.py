
# %%

"""
Control of Navier Stokes equation with DAL (Adjoint fomulation)
"""

import jax
import jax.numpy as jnp
import optax
from tqdm import tqdm
jax.config.update('jax_platform_name', 'cpu')

import tracemalloc, time

from updes import *
import sys

EXPERIMENET_ID = "ChannelAdjoint2_Simple"
DATAFOLDER = "./data/" + EXPERIMENET_ID +"/"
make_dir(DATAFOLDER)

## Save data for comparison
COMPFOLDER = "./data/" + "Comparison" +"/"
make_dir(COMPFOLDER)

### Constants for the problem
RBF = polyharmonic
MAX_DEGREE = 1

Re = 100.
Pa = 0.
ALPHA = 0.02

NB_ITER = 500


## Constants for gradient descent
LR = 1e-2
# GAMMA = 0.995
EPOCHS = 30






# %%


facet_types_vel = {"Wall":"d", "Inflow":"d", "Outflow":"n", "Blowing":"d", "Suction":"d"}
facet_types_p = {"Wall":"n", "Inflow":"n", "Outflow":"d", "Blowing":"n", "Suction":"n"}

cloud_vel = GmshCloud(filename="./meshes/channel_blowing_suction.py", facet_types=facet_types_vel, mesh_save_location=DATAFOLDER)
cloud_p = GmshCloud(filename=DATAFOLDER+"mesh.msh", facet_types=facet_types_p)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6,3*2), sharex=True)
cloud_vel.visualize_cloud(ax=ax1, s=2, title=r"Cloud for $\mathbf{u}$", xlabel=False);
cloud_p.visualize_cloud(ax=ax2, s=2, title=r"Cloud for $p$");


# %%

start = time.process_time()
tracemalloc.start()




def diff_operator_u(x, center=None, rbf=None, monomial=None, fields=None):
    U_prev = jnp.array([fields[0], fields[1]])
    u_grad = nodal_gradient(x, center, rbf, monomial)
    u_lap = nodal_laplacian(x, center, rbf, monomial)
    return jnp.dot(U_prev, u_grad) - u_lap/Re

def rhs_operator_u(x, centers=None, rbf=None, fields=None):
    grad_px = gradient(x, fields[:, 0], centers, rbf)[0]
    return -grad_px


def diff_operator_v(x, center=None, rbf=None, monomial=None, fields=None):
    U_prev = jnp.array([fields[0], fields[1]])
    v_grad = nodal_gradient(x, center, rbf, monomial)
    v_lap = nodal_laplacian(x, center, rbf, monomial)
    return jnp.dot(U_prev, v_grad) - v_lap/Re

def rhs_operator_v(x, centers=None, rbf=None, fields=None):
    grad_py = gradient(x, fields[:, 0], centers, rbf)[1]
    return -grad_py


def diff_operator_p(x, center=None, rbf=None, monomial=None, fields=None):
    # invA = jnp.array([fields[0], fields[1]])
    invA = fields[:2]
    return nodal_div_grad(x, center, rbf, monomial, invA)
    # return nodal_laplacian(x, center, rbf, monomial)

def rhs_operator_p(x, centers=None, rbf=None, fields=None):
    return divergence(x, fields[:, :2], centers, rbf)




def simulate_navier_stokes(cloud_vel, 
                            cloud_p, 
                            NB_ITER=NB_ITER, 
                            RBF=RBF,
                            MAX_DEGREE=MAX_DEGREE):

    """ Simulates a adjoint Navier Stokes problem using an iterative approach """

    parabolic = jax.jit(lambda x: 4*x[1]*(1.-x[1]))
    atmospheric = jax.jit(lambda x: Pa)
    blowing = jax.jit(lambda x: 0.3)
    suction = jax.jit(lambda x: 0.3)
    zero = jax.jit(lambda x: 0.)

    u = jnp.ones((cloud_vel.N,))
    v = jnp.ones((cloud_vel.N,))*0.3
    p_ = jnp.zeros((cloud_p.N,))


    bc_u = {"Wall":zero, "Inflow":parabolic, "Outflow":zero, "Blowing":zero, "Suction":zero}
    bc_u = boundary_conditions_func_to_arr(bc_u, cloud_vel)
    bc_v = {"Wall":zero, "Inflow":zero, "Outflow":zero, "Blowing":blowing, "Suction":suction}
    bc_v = boundary_conditions_func_to_arr(bc_v, cloud_vel)
    bc_p = {"Wall":zero, "Inflow":zero, "Outflow":atmospheric, "Blowing":zero, "Suction":zero}
    bc_p = boundary_conditions_func_to_arr(bc_p, cloud_p)


    u_list = [u]
    v_list = [v]
    vel_list = [jnp.sqrt(u**2 + v**2)]
    p_list = [p_]

    for i in tqdm(range(NB_ITER)):

        p = interpolate_field(p_, cloud_p, cloud_vel)

        usol = pde_solver_jit_with_bc(diff_operator=diff_operator_u,
                        diff_args=[u, v],
                        rhs_operator = rhs_operator_u,
                        rhs_args=[p],
                        cloud = cloud_vel,
                        boundary_conditions = bc_u,
                        rbf=RBF,
                        max_degree=MAX_DEGREE)

        vsol = pde_solver_jit_with_bc(diff_operator=diff_operator_v,
                        diff_args=[u, v],
                        rhs_operator = rhs_operator_v,
                        rhs_args=[p],
                        cloud = cloud_vel,
                        boundary_conditions = bc_v,
                        rbf=RBF,
                        max_degree=MAX_DEGREE)

        ustar, vstar = usol.vals, vsol.vals
        M1, M2 = usol.mat, vsol.mat

        diagInvA1 = 1./M1.diagonal()
        diagInvA2 = 1./M2.diagonal()

        if jnp.any(jnp.isnan(diagInvA1)) or jnp.any(jnp.isnan(diagInvA2)):
            print("WARNING: NaNs in diagonal of A. Setting to 1.")
            diagInvA1 = jnp.where(jnp.isnan(diagInvA1), 1., diagInvA1)
            diagInvA2 = jnp.where(jnp.isnan(diagInvA2), 1., diagInvA2)

        diagInvA1_ = interpolate_field(diagInvA1, cloud_vel, cloud_p)
        diagInvA2_ = interpolate_field(diagInvA2, cloud_vel, cloud_p)

        ustar_ = interpolate_field(ustar, cloud_vel, cloud_p)
        vstar_ = interpolate_field(vstar, cloud_vel, cloud_p)

        pprimesol_ = pde_solver_jit_with_bc(diff_operator=diff_operator_p,
                        diff_args=[diagInvA1_, diagInvA2_],
                        rhs_operator = rhs_operator_p,
                        rhs_args=[ustar_, vstar_],
                        cloud = cloud_p,
                        boundary_conditions = bc_p,
                        rbf=RBF,
                        max_degree=MAX_DEGREE)

        p_ = p_list[-1] + pprimesol_.vals

        gradpprime_ = gradient_vec(cloud_p.sorted_nodes, pprimesol_.coeffs, cloud_p.sorted_nodes, RBF)
        gradpprime = interpolate_field(gradpprime_, cloud_p, cloud_vel)

        u = ustar - jnp.diag(diagInvA1)@gradpprime[:,0]
        v = vstar - jnp.diag(diagInvA2)@gradpprime[:,1]

        p_ = ALPHA * p_ + (1-ALPHA) * p_list[-1]

        u_list.append(u)
        v_list.append(v)
        vel_list.append(jnp.sqrt(u**2 + v**2))
        p_list.append(p_)

    return u_list, v_list, vel_list, p_list


# %%

# def simulate_navier_stokes(cloud_vel, 
#                             cloud_p, 
#                             NB_ITER=NB_ITER, 
#                             RBF=RBF,
#                             MAX_DEGREE=MAX_DEGREE):

#     """ Simulates a adjoint Navier Stokes problem using an iterative approach """

#     parabolic = jax.jit(lambda x: 4*x[1]*(1.-x[1]))
#     atmospheric = jax.jit(lambda x: Pa)
#     blowing = jax.jit(lambda x: 0.3)
#     suction = jax.jit(lambda x: 0.3)
#     zero = jax.jit(lambda x: 0.)

#     u = jnp.ones((cloud_vel.N,))
#     v = jnp.ones((cloud_vel.N,)) * 0.3
#     p_ = jnp.zeros((cloud_p.N,))


#     bc_u = {"Wall":zero, "Inflow":parabolic, "Outflow":zero, "Blowing":zero, "Suction":zero}
#     bc_u = boundary_conditions_func_to_arr(bc_u, cloud_vel)
#     bc_v = {"Wall":zero, "Inflow":zero, "Outflow":zero, "Blowing":blowing, "Suction":suction}
#     bc_v = boundary_conditions_func_to_arr(bc_v, cloud_vel)
#     bc_p = {"Wall":zero, "Inflow":zero, "Outflow":atmospheric, "Blowing":zero, "Suction":zero}
#     bc_p = boundary_conditions_func_to_arr(bc_p, cloud_p)


#     u_list = [u]
#     v_list = [v]
#     vel_list = [jnp.sqrt(u**2 + v**2)]
#     p_list = [p_]

#     for i in tqdm(range(NB_ITER)):

#         p = interpolate_field(p_, cloud_p, cloud_vel)

#         usol = pde_solver_jit_with_bc(diff_operator=diff_operator_u,
#                         diff_args=[u, v],
#                         rhs_operator = rhs_operator_u,
#                         rhs_args=[p],
#                         cloud = cloud_vel,
#                         boundary_conditions = bc_u,
#                         rbf=RBF,
#                         max_degree=MAX_DEGREE)

#         vsol = pde_solver_jit_with_bc(diff_operator=diff_operator_v,
#                         diff_args=[u, v],
#                         rhs_operator = rhs_operator_v,
#                         rhs_args=[p],
#                         cloud = cloud_vel,
#                         boundary_conditions = bc_v,
#                         rbf=RBF,
#                         max_degree=MAX_DEGREE)

#         u, v = usol.vals, vsol.vals
#         M1, M2 = usol.mat, vsol.mat
#         ALPHA = 0.01

#         # A1 = jnp.diag(M1.diagonal())
#         # A2 = jnp.diag(M2.diagonal())
#         # gradp_ = gradient_vals_vec(cloud_p.sorted_nodes, p_, cloud_p, RBF, MAX_DEGREE)
#         # gradp = interpolate_field(gradp_, cloud_p, cloud_vel)
#         # relax = (1-ALPHA)/ALPHA
#         # u = jnp.linalg.solve(M1+relax*A1, -gradp[:,0] + (relax*A1)@u_list[-1])
#         # v = jnp.linalg.solve(M2+relax*A2, -gradp[:,1] + (relax*A2)@v_list[-1])


#         diagInvA1 = 1./M1.diagonal()
#         diagInvA2 = 1./M2.diagonal()
    
#         ## Pint diags
#         # print("M1: ", M1)
#         # print("Diag A1: ", diagInvA1)

#         if jnp.any(jnp.isnan(diagInvA1)) or jnp.any(jnp.isnan(diagInvA2)):
#             print("WARNING: NaNs in diagonal of A. Setting to 1.")
#             diagInvA1 = jnp.where(jnp.isnan(diagInvA1), 1., diagInvA1)
#             diagInvA2 = jnp.where(jnp.isnan(diagInvA2), 1., diagInvA2)

#         diagInvA1_ = interpolate_field(diagInvA1, cloud_vel, cloud_p)
#         diagInvA2_ = interpolate_field(diagInvA2, cloud_vel, cloud_p)

#         # diagInvA1_ = jnp.ones(cloud_p.N)
#         # diagInvA2_ = jnp.ones(cloud_p.N)

#         # diagInvA1_ = jnp.arange(cloud_p.N) / cloud_p.N
#         # diagInvA2_ = jnp.arange(cloud_p.N) / cloud_p.N

#         H1 = jnp.diag(M1.diagonal())@u - M1@u
#         H2 = jnp.diag(M2.diagonal())@v - M2@v

#         invA1 = jnp.diag(diagInvA1)
#         invA2 = jnp.diag(diagInvA2)

#         invA1H1_ = interpolate_field(invA1 @ H1, cloud_vel, cloud_p)
#         invA2H2_ = interpolate_field(invA2 @ H2, cloud_vel, cloud_p)

#         psol_ = pde_solver_jit_with_bc(diff_operator=diff_operator_p,
#                         diff_args=[diagInvA1_, diagInvA2_],
#                         rhs_operator = rhs_operator_p,
#                         rhs_args=[invA1H1_, invA2H2_],
#                         cloud = cloud_p,
#                         boundary_conditions = bc_p,
#                         rbf=RBF,
#                         max_degree=MAX_DEGREE)

#         # p_ = psol_.vals
#         p_ = ALPHA * psol_.vals + (1-ALPHA) * p_list[-1]

#         # gradp_ = gradient_vec(cloud_p.sorted_nodes, psol_.coeffs, cloud_p.sorted_nodes, RBF)
#         gradp_ = gradient_vals_vec(cloud_p.sorted_nodes, p_, cloud_p, RBF, MAX_DEGREE)
#         gradp = interpolate_field(gradp_, cloud_p, cloud_vel)

#         # u = invA1@H1 - invA1@gradp[:,0]
#         # v = invA2@H2 - invA2@gradp[:,1]

#         u_list.append(u)
#         v_list.append(v)
#         vel_list.append(jnp.sqrt(u**2 + v**2))
#         p_list.append(p_)

#     return u_list, v_list, vel_list, p_list





#%%



print(f"\nStarting RBF simulation with {cloud_vel.N} nodes\n")

u_list, v_list, vel_list, p_list = simulate_navier_stokes(cloud_vel, cloud_p)


print("\nSimulation complete. Saving all files to %s" % DATAFOLDER)

renum_map_vel = jnp.array(list(cloud_vel.renumbering_map.keys()))
renum_map_p = jnp.array(list(cloud_p.renumbering_map.keys()))

jnp.savez(DATAFOLDER+'u.npz', renum_map_vel, jnp.stack(u_list, axis=0))
jnp.savez(DATAFOLDER+'v.npz', renum_map_vel, jnp.stack(v_list, axis=0))
jnp.savez(DATAFOLDER+'vel.npz', renum_map_vel, jnp.stack(vel_list, axis=0))
jnp.savez(DATAFOLDER+'p.npz', renum_map_p, jnp.stack(p_list, axis=0))


#%%

print("\nSaving complete. Now running visualisation ...")

pyvista_animation(DATAFOLDER, "u", duration=15, vmin=jnp.min(u_list[-1]), vmax=jnp.max(u_list[-1]))
pyvista_animation(DATAFOLDER, "v", duration=15, vmin=jnp.min(v_list[-1]), vmax=jnp.max(v_list[-1]))
pyvista_animation(DATAFOLDER, "vel", duration=15, vmin=jnp.min(vel_list[-1]), vmax=jnp.max(vel_list[-1]))
pyvista_animation(DATAFOLDER, "p", duration=15, vmin=jnp.min(p_list[-1]), vmax=jnp.max(p_list[-1]))






































# %%


facet_types_lamb = {"Wall":"d", "Inflow":"d", "Outflow":"r", "Blowing":"d", "Suction":"d"}
facet_types_mu = {"Wall":"n", "Inflow":"n", "Outflow":"d", "Blowing":"n", "Suction":"n"}

cloud_lamb = GmshCloud(filename="./meshes/channel_blowing_suction.py", facet_types=facet_types_lamb, mesh_save_location=DATAFOLDER)
cloud_mu = GmshCloud(filename=DATAFOLDER+"mesh.msh", facet_types=facet_types_mu)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6,3*2), sharex=True)
cloud_lamb.visualize_cloud(ax=ax1, s=2, title=r"Cloud for $\lambda$", xlabel=False);
cloud_mu.visualize_cloud(ax=ax2, s=2, title=r"Cloud for $\Pi$");

# %%

start = time.process_time()
tracemalloc.start()


def diff_operator_l1(x, center=None, rbf=None, monomial=None, fields=None):
    # lambda_val = jnp.array([nodal_value(x, center, rbf, monomial), fields[0]])
    lambda_val = jnp.array([fields[0], fields[1]])
    U_val = jnp.array([fields[2], fields[3]])
    U_grad_T = jnp.array([fields[4], fields[5]])
    lambda_grad = nodal_gradient(x, center, rbf, monomial)
    lambda_lap = nodal_laplacian(x, center, rbf, monomial)
    return jnp.dot(lambda_val, U_grad_T) - jnp.dot(U_val, lambda_grad) - lambda_lap/Re

def rhs_operator_l1(x, centers=None, rbf=None, fields=None):
    grad_pi_x = gradient(x, fields[:, 0], centers, rbf)[0]
    return -grad_pi_x


def diff_operator_l2(x, center=None, rbf=None, monomial=None, fields=None):
    # lambda_val = jnp.array([fields[0], nodal_value(x, center, rbf, monomial)])
    lambda_val = jnp.array([fields[0], fields[1]])
    U_val = jnp.array([fields[2], fields[3]])
    U_grad_T = jnp.array([fields[4], fields[5]])
    lambda_grad = nodal_gradient(x, center, rbf, monomial)
    lambda_lap = nodal_laplacian(x, center, rbf, monomial)
    return jnp.dot(lambda_val, U_grad_T) - jnp.dot(U_val, lambda_grad) - lambda_lap/Re

def rhs_operator_l2(x, centers=None, rbf=None, fields=None):
    grad_pi_y = gradient(x, fields[:, 0], centers, rbf)[1]
    return -grad_pi_y


def diff_operator_pi(x, center=None, rbf=None, monomial=None, fields=None):
    invA = fields[:2]
    return nodal_div_grad(x, center, rbf, monomial, invA)

def rhs_operator_pi(x, centers=None, rbf=None, fields=None):
    return divergence(x, fields[:, :2], centers, rbf)


def simulate_adjoint_navier_stokes(cloud_lamb, 
                                    cloud_mu, 
                                    cloud_vel=None,
                                    u=None, v=None, 
                                    ALPHA=0.01,
                                    NB_ITER=NB_ITER, 
                                    RBF=RBF, MAX_DEGREE=MAX_DEGREE):

    """ Simulates a adjoint Navier Stokes problem using an iterative approach """
    l1 = jnp.ones((cloud_lamb.N,))
    l2 = jnp.ones((cloud_lamb.N,)) * 0.3
    pi_ = jnp.zeros((cloud_mu.N,))

    out_nodes_lamb = jnp.array(cloud_lamb.facet_nodes["Outflow"])
    out_nodes_pi = jnp.array(cloud_mu.facet_nodes["Outflow"])

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
            print("WARNING: u velocity not provided, using parabolic")
            u = jax.vmap(parabolic)(cloud_lamb.sorted_nodes)
        if v == None:
            print("WARNING: v velocity not provided, using almost zeros")
            # v = jax.vmap(parabolic)(cloud_lamb.sorted_nodes) / 1000
            v = jnp.ones_like(u) / 1000



    out_nodes_vel = jnp.array(cloud_vel.facet_nodes["Outflow"])
    u_parab = jax.vmap(parabolic)(cloud_vel.sorted_nodes[out_nodes_vel])

    u1 = u[out_nodes_vel]
    u2 = v[out_nodes_vel]

    grad_u = gradient_vals_vec(cloud_vel.sorted_nodes, u, cloud_vel, RBF, MAX_DEGREE)
    grad_v = gradient_vals_vec(cloud_vel.sorted_nodes, v, cloud_vel, RBF, MAX_DEGREE)
    grad_u = interpolate_field(grad_u, cloud_vel, cloud_lamb)
    grad_v = interpolate_field(grad_v, cloud_vel, cloud_lamb)

    l1_list = [l1]
    l2_list = [l2]
    lnorm_list = [jnp.sqrt(l1**2 + l2**2)]
    pi_list = [pi_]

    for i in tqdm(range(NB_ITER)):

        pi = interpolate_field(pi_, cloud_mu, cloud_lamb)

        bc_l1 = {"Wall":zero, "Inflow":zero, "Outflow":((u1-u_parab+pi[out_nodes_lamb])*Re, u1*Re), "Blowing":zero, "Suction":zero}
        # bc_l1 = {"Wall":zero, "Inflow":zero, "Outflow":((u1-u_parab)*Re, u1*Re), "Blowing":zero, "Suction":zero}
        # bc_l1 = {"Wall":zero, "Inflow":zero, "Outflow":((u1-u_parab+pi[out_nodes_lamb])*Re, u1*Re), "Blowing":zero, "Suction":zero}
        bc_l1 = boundary_conditions_func_to_arr(bc_l1, cloud_lamb)

        # bc_l1 = ((u1-u_parab-pi_[out_nodes_pi])*Re, u1*Re)

        l1sol = pde_solver_jit_with_bc(diff_operator=diff_operator_l1,
                        diff_args=[l1, l2, u, v, grad_u[:,0], grad_v[:,0]],
                        # diff_args=[l1, l2, 0*u, 0*v, 0*grad_u[:,0], 0*grad_v[:,0]],
                        rhs_operator = rhs_operator_l1,
                        rhs_args=[pi],
                        cloud = cloud_lamb,
                        boundary_conditions = bc_l1,
                        rbf=RBF,
                        max_degree=MAX_DEGREE)

        bc_l2 = {"Wall":zero, "Inflow":zero, "Outflow":(u2*Re, u1*Re), "Blowing":zero, "Suction":zero}
        # bc_l2 = {"Wall":zero, "Inflow":zero, "Outflow":(u2*Re*0, u1*Re), "Blowing":zero, "Suction":zero}
        bc_l2 = boundary_conditions_func_to_arr(bc_l2, cloud_lamb)

        l2sol = pde_solver_jit_with_bc(diff_operator=diff_operator_l2,
                        diff_args=[l1, l2, u, v, grad_u[:,1], grad_v[:,1]],
                        # diff_args=[l1, l2, 0*u, 0*v, 0*grad_u[:,1], 0*grad_v[:,1]],
                        rhs_operator = rhs_operator_l2,
                        rhs_args=[pi],
                        cloud = cloud_lamb,
                        boundary_conditions = bc_l2,
                        rbf=RBF,
                        max_degree=MAX_DEGREE)


        l1star, l2star = l1sol.vals, l2sol.vals
        M1, M2 = l1sol.mat, l2sol.mat

        diagInvA1 = 1./M1.diagonal()
        diagInvA2 = 1./M2.diagonal()

        if jnp.any(jnp.isnan(diagInvA1)) or jnp.any(jnp.isnan(diagInvA2)):
            print("WARNING: NaNs in diagonal of A. Setting to 1.")
            diagInvA1 = jnp.where(jnp.isnan(diagInvA1), 1., diagInvA1)
            diagInvA2 = jnp.where(jnp.isnan(diagInvA2), 1., diagInvA2)

        diagInvA1_ = interpolate_field(diagInvA1, cloud_lamb, cloud_mu)
        diagInvA2_ = interpolate_field(diagInvA2, cloud_lamb, cloud_mu)

        l1star_ = interpolate_field(l1star, cloud_lamb, cloud_mu)
        l2star_ = interpolate_field(l2star, cloud_lamb, cloud_mu)

        grad_l1 = gradient_vals_vec(cloud_lamb.sorted_nodes[out_nodes_lamb], l1, cloud_lamb, RBF, MAX_DEGREE)
        new_pi_out = u1*l1[out_nodes_lamb] + u_parab - u1 + grad_l1[...,0]/Re
        bc_pi = {"Wall":zero, "Inflow":zero, "Outflow":new_pi_out, "Blowing":zero, "Suction":zero}
        # bc_pi = {"Wall":zero, "Inflow":zero, "Outflow":new_pi_out*0, "Blowing":zero, "Suction":zero}
        bc_pi = boundary_conditions_func_to_arr(bc_pi, cloud_mu)


        piprimesol_ = pde_solver_jit_with_bc(diff_operator=diff_operator_pi,
                        diff_args=[diagInvA1_, diagInvA2_],
                        rhs_operator = rhs_operator_pi,
                        rhs_args=[l1star_, l2star_],
                        cloud = cloud_mu,
                        boundary_conditions = bc_pi,
                        rbf=RBF,
                        max_degree=MAX_DEGREE)

        pi_ = pi_list[-1] + piprimesol_.vals

        gradpiprime_ = gradient_vec(cloud_mu.sorted_nodes, piprimesol_.coeffs, cloud_mu.sorted_nodes, RBF)
        gradpiprime = interpolate_field(gradpiprime_, cloud_mu, cloud_lamb)

        # l1 = l1star
        # l2 = l2star

        l1 = l1star - jnp.diag(diagInvA1)@gradpiprime[:,0]
        l2 = l2star - jnp.diag(diagInvA2)@gradpiprime[:,1]

        # l1 = ALPHA * l1 + (1-ALPHA) * l1_list[-1]
        # l2 = ALPHA * l2 + (1-ALPHA) * l2_list[-1]

        pi_ = ALPHA * pi_ + (1-ALPHA) * pi_list[-1]


        l1_list.append(l1)
        l2_list.append(l2)
        lnorm_list.append(jnp.sqrt(l1**2 + l2**2))
        pi_list.append(pi_)

    return l1_list, l2_list, lnorm_list, pi_list





#%%



print(f"\nStarting RBF simulation with {cloud_lamb.N} nodes\n")


## Load latest u and v
u = jnp.load(DATAFOLDER+"u.npz")["arr_1"][-1]
v = jnp.load(DATAFOLDER+"v.npz")["arr_1"][-1]

l1_list, l2_list, lnorm_list, pi_list = simulate_adjoint_navier_stokes(cloud_lamb, cloud_mu, u=u, v=v, cloud_vel=cloud_vel, NB_ITER=100, ALPHA=0.005)

print("\nSimulation complete. Saving all files to %s" % DATAFOLDER)

renum_map_vel = jnp.array(list(cloud_lamb.renumbering_map.keys()))
renum_map_p = jnp.array(list(cloud_mu.renumbering_map.keys()))

jnp.savez(DATAFOLDER+'l1.npz', renum_map_vel, jnp.stack(l1_list, axis=0))
jnp.savez(DATAFOLDER+'l2.npz', renum_map_vel, jnp.stack(l2_list, axis=0))
jnp.savez(DATAFOLDER+'lnorm.npz', renum_map_vel, jnp.stack(lnorm_list, axis=0))
jnp.savez(DATAFOLDER+'pi.npz', renum_map_p, jnp.stack(pi_list, axis=0))


#%%

print("\nSaving complete. Now running visualisation ...")

pyvista_animation(DATAFOLDER, "l1", duration=15, vmin=jnp.min(l1_list[-1]), vmax=jnp.max(l1_list[-1]))
pyvista_animation(DATAFOLDER, "l2", duration=15, vmin=jnp.min(l2_list[-1]), vmax=jnp.max(l2_list[-1]))
pyvista_animation(DATAFOLDER, "lnorm", duration=15, vmin=jnp.min(lnorm_list[-1]), vmax=jnp.max(lnorm_list[-1]))
pyvista_animation(DATAFOLDER, "pi", duration=15, vmin=jnp.min(pi_list[-1]), vmax=jnp.max(pi_list[-1]))






































# %%

### Direct adjoitn Looping

## Bluid new clouds for forward problem (different boundary conditions)
facet_types_vel = {"Wall":"d", "Inflow":"d", "Outflow":"n", "Blowing":"d", "Suction":"d"}
facet_types_phi = {"Wall":"n", "Inflow":"n", "Outflow":"d", "Blowing":"n", "Suction":"n"}
cloud_vel = GmshCloud(filename=DATAFOLDER+"mesh.msh", facet_types=facet_types_vel)
cloud_phi = GmshCloud(filename=DATAFOLDER+"mesh.msh", facet_types=facet_types_phi)


## Import forward solver
simulate_forward_navier_stokes = __import__('30_channel_flow_blowing_suction').simulate_forward_navier_stokes

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
    # grad_l1 = cartesian_gradient_vec(range(cloud_lamb.N), l1, cloud_lamb)
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

in_nodes_vel = jnp.array(cloud_vel.facet_nodes["Inflow"])
optimal_u_inflow = jax.vmap(parabolic)(cloud_vel.sorted_nodes[in_nodes_vel])

scheduler = optax.piecewise_constant_schedule(init_value=LR,
                                            boundaries_and_scales={int(EPOCHS*0.5):0.1, int(EPOCHS*0.75):0.1})
optimiser = optax.adam(learning_rate=scheduler)
opt_state = optimiser.init(optimal_u_inflow)

history_cost = []
parab_out_mse = []




for step in range(1, EPOCHS+1):


    forward_sim_args["inflow_control"] = optimal_u_inflow
    u_list, v_list, vel_list, p_list = simulate_forward_navier_stokes(**forward_sim_args)

    # grad_u = gradient_vals_vec(cloud_vel.sorted_nodes, u_list[-1], cloud_vel, RBF, MAX_DEGREE)
    # grad_v = gradient_vals_vec(cloud_vel.sorted_nodes, v_list[-1], cloud_vel, RBF, MAX_DEGREE)

    adjoint_sim_args["u"] = u_list[-1]
    # adjoint_sim_args["u"] = u_list[-1].at[:].set(0.)
    # adjoint_sim_args["u"] = jax.vmap(parabolic)(cloud_vel.sorted_nodes)
    adjoint_sim_args["v"] = v_list[-1]
    # adjoint_sim_args["v"] = v_list[-1].at[:].set(0.)
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

        # print("Optimized inflow vel:", optimal_u_inflow)
        plot(optimal_u_inflow, y_in, "--", label="Optimised DAL", ax=ax2, xlim=None, title=f"Inflow velocity");

        plt.show()


mem_usage = tracemalloc.get_traced_memory()[1]
exec_time = time.process_time() - start

print("A few performance details:")
print(" Peak memory usage: ", mem_usage, 'bytes')
print(' CPU execution time:', exec_time, 'seconds')

tracemalloc.stop()



#%%
ax = plot(history_cost[:], label='Cost objective', x_label='epochs', title="Loss", y_scale="log");
# plot(parab_out_mse[:], label='Test MSE at outlet', x_label='epochs', title="Loss", y_scale="log", ax=ax);


#%%


print("\nOptimisation complete complete. Now running visualisation of direct state")

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

print("\nNow running visualisation of adjoint simulation...")

renum_map_vel = jnp.array(list(cloud_lamb.renumbering_map.keys()))
renum_map_p = jnp.array(list(cloud_mu.renumbering_map.keys()))

jnp.savez(DATAFOLDER+'l1.npz', renum_map_vel, jnp.stack(l1_list, axis=0))
jnp.savez(DATAFOLDER+'l2.npz', renum_map_vel, jnp.stack(l2_list, axis=0))
jnp.savez(DATAFOLDER+'lnorm.npz', renum_map_vel, jnp.stack(lnorm_list, axis=0))
jnp.savez(DATAFOLDER+'pi.npz', renum_map_p, jnp.stack(pi_list, axis=0))

# pyvista_animation(DATAFOLDER, "l1", duration=15, vmin=jnp.min(l1_list[-1]), vmax=jnp.max(l1_list[-1]))
# pyvista_animation(DATAFOLDER, "l2", duration=15, vmin=jnp.min(l2_list[-1]), vmax=jnp.max(l2_list[-1]))
# pyvista_animation(DATAFOLDER, "lnorm", duration=15, vmin=jnp.min(lnorm_list[-1]), vmax=jnp.max(lnorm_list[-1]))
# pyvista_animation(DATAFOLDER, "pi", duration=15, vmin=jnp.min(pi_list[-1]), vmax=jnp.max(pi_list[-1]))

pyvista_animation(DATAFOLDER, "l1", duration=5)
pyvista_animation(DATAFOLDER, "l2", duration=5)
pyvista_animation(DATAFOLDER, "lnorm", duration=5)
pyvista_animation(DATAFOLDER, "pi", duration=5)



# %%

jnp.savez(COMPFOLDER+"dal", objective_cost=history_cost, outflow_mse=parab_out_mse, optimal_control=optimal_u_inflow, mem_time=jnp.array([mem_usage, exec_time]))
