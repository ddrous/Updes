
# # %%
# import jax
# import jax.numpy as jnp
# from jax.tree_util import Partial
# from tqdm import tqdm

# from updec import *


# ### Constants for the problem
# RBF = polyharmonic
# MAX_DEGREE = 1

# Re = 10
# Du = Re
# DT = 1e-6

# Pa = 0.

# NB_ITER = 1
# NB_REFINEMENTS = 20

# EXPERIMENET_ID = "StepFlow"
# DATAFOLDER = "./data/" + EXPERIMENET_ID +"/"
# make_dir(DATAFOLDER)

# # %%


# facet_types_vel = {"Wall1":"d", "Wall2":"d","Wall3":"d","Wall4":"d", "Inflow":"d", "Outflow":"n"}
# facet_types_phi = {"Wall1":"n", "Wall2":"n","Wall3":"n","Wall4":"n", "Inflow":"n", "Outflow":"d"}

# cloud_vel = GmshCloud(filename="./meshes/channel_step.py", facet_types=facet_types_vel, mesh_save_location=DATAFOLDER)
# cloud_phi = GmshCloud(filename=DATAFOLDER+"mesh.msh", facet_types=facet_types_phi)

# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5,1.4*2), sharex=True)
# cloud_vel.visualize_cloud(ax=ax1, s=0.5, title="Cloud for velocity", xlabel=False);
# cloud_phi.visualize_cloud(ax=ax2, s=0.5, title=r"Cloud for $\phi$");



# # %%
# print(f"\nStarting RBF simulations with {cloud_vel.N} nodes\n")


# def diff_operator_u(x, center=None, rbf=None, monomial=None, fields=None):
#     U = jnp.array([fields[0], fields[1]])
#     u_val = nodal_value(x, center, rbf, monomial)
#     u_grad = nodal_gradient(x, center, rbf, monomial)
#     u_lap = nodal_laplacian(x, center, rbf, monomial)
#     return 3*u_val/(2*DT) + jnp.dot(U, u_grad) - u_lap/Du

# def rhs_operator_u(x, centers=None, rbf=None, fields=None):
#     u_rhs = value(x, fields[:, 0], centers, rbf)
#     grad_px = gradient(x, fields[:, 1], centers, rbf)[0]
#     return u_rhs/(2*DT) - grad_px



# def diff_operator_v(x, center=None, rbf=None, monomial=None, fields=None):
#     U = jnp.array([fields[0], fields[1]])
#     v_val = nodal_value(x, center, rbf, monomial)
#     v_grad = nodal_gradient(x, center, rbf, monomial)
#     v_lap = nodal_laplacian(x, center, rbf, monomial)
#     return 3*v_val/(2*DT) + jnp.dot(U, v_grad) - v_lap/Du

# def rhs_operator_v(x, centers=None, rbf=None, fields=None):
#     v_rhs = value(x, fields[:, 0], centers, rbf)
#     grad_py = gradient(x, fields[:, 1], centers, rbf)[1]
#     return v_rhs/(2*DT) - grad_py



# def diff_operator_phi(x, center=None, rbf=None, monomial=None, fields=None):
#     return nodal_laplacian(x, center, rbf, monomial)

# def rhs_operator_phi(x, centers=None, rbf=None, fields=None):
#     return divergence(x, fields[:, :2], centers, rbf)


# ## Initial states, all defined on cloud_vel
# u_now = u_prev = jnp.zeros((cloud_vel.N,))
# v_now = v_prev = jnp.zeros((cloud_vel.N,))
# in_nodes = jnp.array(cloud_vel.facet_nodes["Inflow"])
# u_prev = u_prev.at[in_nodes].set(1.)
# u_now = u_now.at[in_nodes].set(1.)

# p_now_ = p_prev_ = jnp.zeros((cloud_phi.N,))
# out_nodes = jnp.array(cloud_phi.facet_nodes["Outflow"])
# p_prev_ = p_prev_.at[out_nodes].set(Pa)
# p_now_ = p_now_.at[out_nodes].set(Pa)


# ones = jax.jit(lambda x: 1.)
# zero = jax.jit(lambda x: 0.)
# atmospheric = jax.jit(lambda x: Pa)

# bc_u = {"Wall1":zero, "Wall2":zero, "Wall3":zero, "Wall4":zero, "Inflow":ones, "Outflow":zero}
# bc_v = {"Wall1":zero, "Wall2":zero, "Wall3":zero, "Wall4":zero, "Inflow":zero, "Outflow":zero}
# bc_phi = {"Wall1":zero, "Wall2":zero, "Wall3":zero, "Wall4":zero, "Inflow":zero, "Outflow":atmospheric}


# u_list = []
# v_list = []
# vel_list = []
# p_list = []

# for i in tqdm(range(NB_ITER)):
#     # print("Starting iteration %d" % i)

#     ## For all innner loop
#     u_rhs = 4*u_now - u_prev
#     v_rhs = 4*v_now - v_prev

#     ## Only for k=0
#     u_next_prev = 2*u_now - u_prev
#     v_next_prev = 2*v_now - v_prev
#     p_next_prev_ = 3*p_now_/2 - p_prev_/2

#     for k in range(NB_REFINEMENTS):
#         p_next_prev = interpolate_field(p_next_prev_, cloud_phi, cloud_vel)

#         usol = pde_solver_jit(diff_operator=diff_operator_u,
#                         diff_args=[u_next_prev, v_next_prev],
#                         rhs_operator = rhs_operator_u,
#                         rhs_args=[u_rhs, p_next_prev],
#                         cloud = cloud_vel,
#                         boundary_conditions = bc_u,
#                         rbf=RBF,
#                         max_degree=MAX_DEGREE)
#         u_star_now = usol.vals

#         vsol = pde_solver_jit(diff_operator=diff_operator_v,
#                         diff_args=[u_next_prev, v_next_prev],
#                         rhs_operator = rhs_operator_v,
#                         rhs_args=[v_rhs, p_next_prev], 
#                         cloud = cloud_vel,
#                         boundary_conditions = bc_v,
#                         rbf=RBF,
#                         max_degree=MAX_DEGREE)
#         v_star_now = vsol.vals

#         U_star_now = jnp.stack([u_star_now, v_star_now], axis=-1)
#         u_star_now_ = interpolate_field(u_star_now, cloud_vel, cloud_phi)
#         v_star_now_ = interpolate_field(v_star_now, cloud_vel, cloud_phi)

#         phisol_ = pde_solver_jit(diff_operator=diff_operator_phi,
#                         rhs_operator = rhs_operator_phi,
#                         rhs_args=[u_star_now_, v_star_now_], 
#                         cloud = cloud_phi, 
#                         boundary_conditions = bc_phi,
#                         rbf=RBF,
#                         max_degree=MAX_DEGREE)

#         p_next_now_ = p_next_prev_ + 3*phisol_.vals/(2*DT)

#         # p_next_now_ = p_next_now_.at[out_nodes].set(Pa)

#         gradphi_ = gradient_vec(cloud_phi.sorted_nodes, phisol_.coeffs, cloud_phi.sorted_nodes, RBF)

#         gradphi = interpolate_field(gradphi_, cloud_phi, cloud_vel)
#         U_next_now = U_star_now - gradphi
#         u_next_now, v_next_now = U_next_now[:,0], U_next_now[:,1]

#         vel_next_now = jnp.linalg.norm(U_next_now, axis=-1)

#         u_next_prev = u_next_now 
#         v_next_prev = v_next_now 
#         p_next_prev_ = p_next_now_

#     u_prev = u_now.copy()
#     v_pref = v_now.copy()
#     p_prev_ = p_now_.copy()

#     u_now = u_next_now.copy()
#     v_now = v_next_now.copy()
#     p_now_ = p_next_now_.copy()

#     u_list.append(u_next_now)
#     v_list.append(v_next_now)
#     vel_list.append(vel_next_now)
#     p_list.append(p_next_now_)



# # %%

# print("\nSimulation complete. Saving all files to %s" % DATAFOLDER)


# renum_map_vel = jnp.array(list(cloud_vel.renumbering_map.keys()))
# renum_map_p = jnp.array(list(cloud_phi.renumbering_map.keys()))

# jnp.savez(DATAFOLDER+'u.npz', renum_map_vel, jnp.stack(u_list, axis=0))
# jnp.savez(DATAFOLDER+'v.npz', renum_map_vel, jnp.stack(v_list, axis=0))
# jnp.savez(DATAFOLDER+'vel.npz', renum_map_vel, jnp.stack(vel_list, axis=0))
# jnp.savez(DATAFOLDER+'p.npz', renum_map_p, jnp.stack(p_list, axis=0))

# # plt.show()



# # %%

# print("\nSaving complete. Now running visualisation ...")

# # pyvista_animation(DATAFOLDER, "u", duration=5)
# # pyvista_animation(DATAFOLDER, "v", duration=5)
# pyvista_animation(DATAFOLDER, "vel", duration=5)
# # pyvista_animation(DATAFOLDER, "p", duration=5)
























# # %%

# from tqdm import tqdm

# import jax
# import jax.numpy as jnp

# from updec import *

# EXPERIMENET_ID = "StepFlow"
# DATAFOLDER = "./data/" + EXPERIMENET_ID +"/"
# make_dir(DATAFOLDER)


# facet_types_vel = {"Wall1":"d", "Wall2":"d","Wall3":"d","Wall4":"d", "Inflow":"d", "Outflow":"n"}
# facet_types_phi = {"Wall1":"n", "Wall2":"n","Wall3":"n","Wall4":"n", "Inflow":"n", "Outflow":"d"}

# cloud_vel = GmshCloud(filename="./meshes/channel_step.py", facet_types=facet_types_vel, mesh_save_location=DATAFOLDER)
# cloud_phi = GmshCloud(filename=DATAFOLDER+"mesh.msh", facet_types=facet_types_phi, mesh_save_location=DATAFOLDER)

# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5,1.4*2), sharex=True)
# cloud_vel.visualize_cloud(ax=ax1, s=0.5, title="Cloud for velocity", xlabel=False);
# cloud_phi.visualize_cloud(ax=ax2, s=0.5, title=r"Cloud for $\phi$");

# print("Total number of nodes:", cloud_vel.N)


# RBF = polyharmonic      ## Can define which rbf to use
# MAX_DEGREE = 1

# NB_ITER = 10

# Re = 20
# RHO = 1
# NU = 1
# DT = 1e-6


# # Pa = 101325.0
# Pa = 0.0
# beta = 0.

# def diff_operator_u(x, center=None, rbf=None, monomial=None, fields=None):
#     return nodal_value(x, center, rbf, monomial)

# def rhs_operator_u(x, centers=None, rbf=None, fields=None):
#     u, v = fields[:, 0], fields[:, 1]
#     grad_u = fields[:, 2:4]
#     lapu = fields[:, 4]
#     return  u + DT*(-jnp.dot(fields[:, 2], grad_u) + NU*lapu)


# def diff_operator_v(x, center=None, rbf=None, monomial=None, fields=None):
#     return nodal_value(x, center, rbf, monomial)


# def rhs_operator_v(x, centers=None, rbf=None, fields=None):
#     u, v = fields[:, 0], fields[:, 1]
#     grad_v = fields[:, 2:4]
#     lapv = fields[:, 4]
#     return  v + DT*(-dot_vec(fields[:, 2], grad_v) + NU*lapv)


# def diff_operator_phi(x, center=None, rbf=None, monomial=None, fields=None):
#     return nodal_laplacian(x, center, rbf, monomial)

# def rhs_operator_phi(x, centers=None, rbf=None, fields=None):
#     return RHO * divergence(x, fields[:, :2], centers, rbf) / DT


# ## Initial states, all defined on cloud_vel
# u = jnp.zeros((cloud_vel.N,))
# v = jnp.zeros((cloud_vel.N,))

# p_ = jnp.zeros((cloud_phi.N,))
# out_nodes = jnp.array(cloud_phi.facet_nodes["Outflow"])
# p_ = p_.at[out_nodes].set(Pa)


# ones = jax.jit(lambda x: 1.)
# zero = jax.jit(lambda x: 0.)
# atmospheric = jax.jit(lambda x: Pa)

# bc_u = {"Wall1":zero, "Wall2":zero, "Wall3":zero, "Wall4":zero, "Inflow":ones, "Outflow":zero}
# bc_v = {"Wall1":zero, "Wall2":zero, "Wall3":zero, "Wall4":zero, "Inflow":zero, "Outflow":zero}
# bc_phi = {"Wall1":zero, "Wall2":zero, "Wall3":zero, "Wall4":zero, "Inflow":zero, "Outflow":atmospheric}


# u_list = []
# v_list = []
# vel_list = []
# p_list = []

# for i in tqdm(range(NB_ITER)):
#     # print("Starting iteration %d" % i)

#     p = interpolate_field(p_, cloud_phi, cloud_vel)

#     gradu = gradient_vec(cloud_vel.sorted_nodes, u, cloud_vel.sorted_nodes, RBF)
#     gradv = gradient_vec(cloud_vel.sorted_nodes, v, cloud_vel.sorted_nodes, RBF)

#     lapu = laplacian_vec(cloud_vel.sorted_nodes, u, cloud_vel.sorted_nodes, RBF)
#     lapv = laplacian_vec(cloud_vel.sorted_nodes, v, cloud_vel.sorted_nodes, RBF)

#     usol = pde_solver_jit(diff_operator=diff_operator_u, 
#                     # diff_args=[u, v],
#                     rhs_operator = rhs_operator_u, 
#                     rhs_args=[u, v, gradu[:,0], gradu[:,1], lapu], 
#                     cloud = cloud_vel, 
#                     boundary_conditions = bc_u,
#                     rbf=RBF,
#                     max_degree=MAX_DEGREE)

#     vsol = pde_solver_jit(diff_operator=diff_operator_v,
#                     # diff_args=[u, v],
#                     rhs_operator = rhs_operator_v,
#                     rhs_args=[v, v, gradv[:,0], gradv[:,1], lapv], 
#                     cloud = cloud_vel, 
#                     boundary_conditions = bc_v,
#                     rbf=RBF,
#                     max_degree=MAX_DEGREE)

#     ustar , vstar = usol.vals, vsol.vals
#     Ustar = jnp.stack([ustar,vstar], axis=-1)

#     u_ = interpolate_field(ustar, cloud_vel, cloud_phi)
#     v_ = interpolate_field(vstar, cloud_vel, cloud_phi)

#     phisol_ = pde_solver_jit(diff_operator=diff_operator_phi,
#                     rhs_operator = rhs_operator_phi,
#                     rhs_args=[u_,v_], 
#                     cloud = cloud_phi, 
#                     boundary_conditions = bc_phi,
#                     rbf=RBF,
#                     max_degree=MAX_DEGREE)

#     p_ = phisol_.vals
#     gradphi_ = gradient_vec(cloud_phi.sorted_nodes, phisol_.coeffs, cloud_phi.sorted_nodes, RBF)

#     gradphi = interpolate_field(gradphi_, cloud_phi, cloud_vel)

#     U = Ustar - (gradphi*DT/RHO)
#     u, v = U[:,0], U[:,1]
#     vel = jnp.linalg.norm(U, axis=-1)

#     u_list.append(u)
#     v_list.append(v)
#     vel_list.append(vel)
#     p_list.append(p_)

# # %%

# print("\nSimulation complete. Saving all files to %s" % DATAFOLDER)

# renum_map_vel = jnp.array(list(cloud_vel.renumbering_map.keys()))
# renum_map_p = jnp.array(list(cloud_phi.renumbering_map.keys()))

# jnp.savez(DATAFOLDER+'u.npz', renum_map_vel, jnp.stack(u_list, axis=0))
# jnp.savez(DATAFOLDER+'v.npz', renum_map_vel, jnp.stack(v_list, axis=0))
# jnp.savez(DATAFOLDER+'vel.npz', renum_map_vel, jnp.stack(vel_list, axis=0))
# jnp.savez(DATAFOLDER+'p.npz', renum_map_p, jnp.stack(p_list, axis=0))

# # plt.show()



# # %%

# print("\nSaving complete. Now running visualisation ...")

# # pyvista_animation(DATAFOLDER, "u", duration=5)
# # pyvista_animation(DATAFOLDER, "v", duration=5)
# pyvista_animation(DATAFOLDER, "vel", duration=5)
# # pyvista_animation(DATAFOLDER, "p", duration=5)



























# %%

from tqdm import tqdm

import jax
import jax.numpy as jnp

from updec import *

EXPERIMENET_ID = "StepFlow"
DATAFOLDER = "./data/" + EXPERIMENET_ID +"/"
make_dir(DATAFOLDER)


facet_types_vel = {"Wall1":"d", "Wall2":"d","Wall3":"d","Wall4":"d", "Inflow":"d", "Outflow":"n"}
facet_types_phi = {"Wall1":"n", "Wall2":"n","Wall3":"n","Wall4":"n", "Inflow":"n", "Outflow":"d"}

cloud_vel = GmshCloud(filename="./meshes/channel_step.py", facet_types=facet_types_vel, mesh_save_location=DATAFOLDER)
cloud_phi = GmshCloud(filename=DATAFOLDER+"mesh.msh", facet_types=facet_types_phi, mesh_save_location=DATAFOLDER)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5,1.4*2), sharex=True)
cloud_vel.visualize_cloud(ax=ax1, s=0.5, title="Cloud for velocity", xlabel=False);
cloud_phi.visualize_cloud(ax=ax2, s=0.5, title=r"Cloud for $\phi$");

print("Total number of nodes:", cloud_vel.N)


RBF = polyharmonic
MAX_DEGREE = 1

NB_ITER = 100

Re = 20
RHO = 1
NU = 1
DT = 1e-6


# Pa = 101325.0
Pa = 0.0
beta = 0.

def diff_operator_u(x, center=None, rbf=None, monomial=None, fields=None):
    U_prev = jnp.array([fields[0], fields[1]])
    u_val = nodal_value(x, center, rbf, monomial)
    u_grad = nodal_gradient(x, center, rbf, monomial)
    u_lap = nodal_laplacian(x, center, rbf, monomial)
    return u_val + DT*jnp.dot(U_prev, u_grad) - DT*NU*u_lap

def rhs_operator_u(x, centers=None, rbf=None, fields=None):
    u_prev = value(x, fields[:, 0], centers, rbf)
    grad_px = gradient(x, fields[:, 1], centers, rbf)[0]
    return  u_prev - beta*(DT*grad_px/RHO)



def diff_operator_v(x, center=None, rbf=None, monomial=None, fields=None):
    U_prev = jnp.array([fields[0], fields[1]])
    v_val = nodal_value(x, center, rbf, monomial)
    v_grad = nodal_gradient(x, center, rbf, monomial)
    v_lap = nodal_laplacian(x, center, rbf, monomial)
    return v_val + DT*jnp.dot(U_prev, v_grad) - DT*NU*v_lap

def rhs_operator_v(x, centers=None, rbf=None, fields=None):
    v_prev = value(x, fields[:, 0], centers, rbf)
    grad_py = gradient(x, fields[:, 1], centers, rbf)[1]
    return  v_prev - beta*(DT *grad_py/RHO)



def diff_operator_phi(x, center=None, rbf=None, monomial=None, fields=None):
    return nodal_laplacian(x, center, rbf, monomial)

def rhs_operator_phi(x, centers=None, rbf=None, fields=None):
    return RHO * divergence(x, fields[:, :2], centers, rbf) / DT


## Initial states, all defined on cloud_vel
u = jnp.zeros((cloud_vel.N,))
v = jnp.zeros((cloud_vel.N,))

p_ = jnp.zeros((cloud_phi.N,))
out_nodes = jnp.array(cloud_phi.facet_nodes["Outflow"])
p_ = p_.at[out_nodes].set(Pa)


ones = jax.jit(lambda x: 1.)
zero = jax.jit(lambda x: 0.)
atmospheric = jax.jit(lambda x: Pa)

bc_u = {"Wall1":zero, "Wall2":zero, "Wall3":zero, "Wall4":zero, "Inflow":ones, "Outflow":zero}
bc_v = {"Wall1":zero, "Wall2":zero, "Wall3":zero, "Wall4":zero, "Inflow":zero, "Outflow":zero}
bc_phi = {"Wall1":zero, "Wall2":zero, "Wall3":zero, "Wall4":zero, "Inflow":zero, "Outflow":atmospheric}


u_list = []
v_list = []
vel_list = []
p_list = []

for i in tqdm(range(NB_ITER)):
    # print("Starting iteration %d" % i)

    p = interpolate_field(p_, cloud_phi, cloud_vel)

    usol = pde_solver_jit(diff_operator=diff_operator_u, 
                    diff_args=[u, v],
                    rhs_operator = rhs_operator_u, 
                    rhs_args=[u, p], 
                    cloud = cloud_vel, 
                    boundary_conditions = bc_u,
                    rbf=RBF,
                    max_degree=MAX_DEGREE)

    vsol = pde_solver_jit(diff_operator=diff_operator_v,
                    diff_args=[u, v],
                    rhs_operator = rhs_operator_v,
                    rhs_args=[v, p], 
                    cloud = cloud_vel, 
                    boundary_conditions = bc_v,
                    rbf=RBF,
                    max_degree=MAX_DEGREE)

    ustar , vstar = usol.vals, vsol.vals
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

    p_ = beta*p_ + phisol_.vals
    gradphi_ = gradient_vec(cloud_phi.sorted_nodes, phisol_.coeffs, cloud_phi.sorted_nodes, RBF)

    gradphi = interpolate_field(gradphi_, cloud_phi, cloud_vel)

    U = Ustar - (gradphi*DT/RHO)
    u, v = U[:,0], U[:,1]
    vel = jnp.linalg.norm(U, axis=-1)

    u_list.append(u)
    v_list.append(v)
    vel_list.append(vel)
    p_list.append(p_)

# filename = DATAFOLDER+'/video.mp4'
# cloud_vel.animate_fields([all_u, all_v, all_vel, all_p], filename=filename, cmaps=["jet", "jet", "jet", "magma"], titles=[r"$u$", r"$v$", "Velocity amplitude", "Pressure"], duration=5, figsize=(9.5,1.4*4));

# plt.show()


# %%

print("\nSimulation complete. Saving all files to %s" % DATAFOLDER)

renum_map_vel = jnp.array(list(cloud_vel.renumbering_map.keys()))
renum_map_p = jnp.array(list(cloud_phi.renumbering_map.keys()))

jnp.savez(DATAFOLDER+'u.npz', renum_map_vel, jnp.stack(u_list, axis=0))
jnp.savez(DATAFOLDER+'v.npz', renum_map_vel, jnp.stack(v_list, axis=0))
jnp.savez(DATAFOLDER+'vel.npz', renum_map_vel, jnp.stack(vel_list, axis=0))
jnp.savez(DATAFOLDER+'p.npz', renum_map_p, jnp.stack(p_list, axis=0))

# plt.show()



# %%

print("\nSaving complete. Now running visualisation ...")

# pyvista_animation(DATAFOLDER, "u", duration=5)
# pyvista_animation(DATAFOLDER, "v", duration=5)
pyvista_animation(DATAFOLDER, "vel", duration=5)
# pyvista_animation(DATAFOLDER, "p", duration=5)
















