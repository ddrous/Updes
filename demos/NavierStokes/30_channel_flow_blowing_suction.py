
# %%

"""
Control of Navier Stokes equation with DAL (Direct fomulation)
"""

import jax
import jax.numpy as jnp
from jax.tree_util import Partial
from functools import partial
from tqdm import tqdm
jax.config.update('jax_platform_name', 'cpu')           ## TODO Slow on GPU on Daffy Duck !

from updec import *

# %%


# EXPERIMENET_ID = random_name()
EXPERIMENET_ID = "ChannelAdjoint1"
DATAFOLDER = "./data/" + EXPERIMENET_ID +"/"
make_dir(DATAFOLDER)


# %%
### Constants for the problem

RBF = polyharmonic      ## Can define which rbf to use
MAX_DEGREE = 1

Re = 10
Pa = 0.

NB_ITER = 100


# %%

facet_types_vel = {"Wall":"d", "Inflow":"d", "Outflow":"n", "Blowing":"d", "Suction":"d"}
facet_types_phi = {"Wall":"n", "Inflow":"n", "Outflow":"d", "Blowing":"n", "Suction":"n"}

cloud_vel = GmshCloud(filename="./meshes/channel_blowing_suction.py", facet_types=facet_types_vel, mesh_save_location=DATAFOLDER)
cloud_phi = GmshCloud(filename=DATAFOLDER+"mesh.msh", facet_types=facet_types_phi)

if __name__ == "__main__":
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6,3*2), sharex=True)
    cloud_vel.visualize_cloud(ax=ax1, s=6, title="Cloud for velocity", xlabel=False);
    cloud_phi.visualize_cloud(ax=ax2, s=6, title=r"Cloud for $\phi$");

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.5*2,5))
    cloud_vel.visualize_normals(ax=ax1, title="Normals for velocity")
    cloud_phi.visualize_normals(ax=ax2,title="Normals for phi", zoom_region=(0.25,1.25,-0.1,1.1));



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
    vel_list = []
    p_list = [p_]


    # compiled_pde_solver = jax.jit(pde_solver, static_argnums=[0,2,4,5,6,7])     ## BCs are static here

    # @jax.jit
    # def compiled_pde_solver_with_bc(u_inflow, diff_args, rhs_args):
    #     bc_u = {"Wall":zero, "Inflow":u_inflow, "Outflow":zero, "Cylinder":zero, "Blowing":zero, "Suction":zero}
    #     return pde_solver(diff_operator=diff_operator_u, 
    #                     diff_args=diff_args,
    #                     rhs_operator = rhs_operator_u, 
    #                     rhs_args=rhs_args, 
    #                     cloud = cloud_vel, 
    #                     boundary_conditions = bc_u,
    #                     rbf=RBF,
    #                     max_degree=MAX_DEGREE)


    # for i in tqdm(range(NB_ITER), disable=True):
    for i in range(NB_ITER):
        # print("Starting iteration %d" % i)

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


    grad_u = gradient_vals_vec(cloud_vel.sorted_nodes, u, cloud_vel, RBF, MAX_DEGREE)
    grad_v = gradient_vals_vec(cloud_vel.sorted_nodes, v, cloud_vel, RBF, MAX_DEGREE)

    # print("OLD MAX abs grad u: ", jnp.max(jnp.abs(grad_u)))
    # print("OLD MAX abs grad v: ", jnp.max(jnp.abs(grad_v)))


    usol = pde_solver_jit(diff_operator=diff_operator_id, 
                    diff_args=None,
                    rhs_operator = rhs_operator_id, 
                    rhs_args=[u],
                    cloud = cloud_vel,
                    boundary_conditions = bc_u,
                    rbf=RBF,
                    max_degree=MAX_DEGREE)
    vsol = pde_solver_jit(diff_operator=diff_operator_id, 
                    diff_args=None,
                    rhs_operator = rhs_operator_id, 
                    rhs_args=[v],
                    cloud = cloud_vel,
                    boundary_conditions = bc_v,
                    rbf=RBF,
                    max_degree=MAX_DEGREE)

    grad_u = gradient_vec(cloud_vel.sorted_nodes, usol.coeffs, cloud_vel.sorted_nodes, RBF)
    grad_v = gradient_vec(cloud_vel.sorted_nodes, vsol.coeffs, cloud_vel.sorted_nodes, RBF)

    # print("MAX abs grad u: ", jnp.max(jnp.abs(grad_u)))
    # print("MAX abs grad v: ", jnp.max(jnp.abs(grad_v)))

    # print("MAX abs u: ", jnp.max(jnp.abs(u)))
    # print("MAX abs v: ", jnp.max(jnp.abs(v)))


    return u_list, v_list, vel_list, p_list


def diff_operator_id(x, center=None, rbf=None, monomial=None, fields=None):
    return nodal_value(x, center, rbf, monomial)

def rhs_operator_id(x, centers=None, rbf=None, fields=None):
    return value(x, fields[:, 0], centers, rbf)



# %%

if __name__ == "__main__":

    print(f"\nStarting RBF simulation with {cloud_vel.N} nodes\n")
    u_list, v_list, vel_list, p_list = simulate_forward_navier_stokes(cloud_vel, cloud_phi)


    print("\nSimulation complete. Saving all files to %s" % DATAFOLDER)

    renum_map_vel = jnp.array(list(cloud_vel.renumbering_map.keys()))
    renum_map_p = jnp.array(list(cloud_phi.renumbering_map.keys()))

    jnp.savez(DATAFOLDER+'u.npz', renum_map_vel, jnp.stack(u_list, axis=0))
    jnp.savez(DATAFOLDER+'v.npz', renum_map_vel, jnp.stack(v_list, axis=0))
    jnp.savez(DATAFOLDER+'vel.npz', renum_map_vel, jnp.stack(vel_list, axis=0))
    jnp.savez(DATAFOLDER+'p.npz', renum_map_p, jnp.stack(p_list, axis=0))


# %%


if __name__ == "__main__":

    print("\nSaving complete. Now running visualisation ...")

    pyvista_animation(DATAFOLDER, "vel", duration=5, vmin=0.0, vmax=1.1)
    pyvista_animation(DATAFOLDER, "p", duration=5, vmin=-0.25, vmax=0.45)


# %%
