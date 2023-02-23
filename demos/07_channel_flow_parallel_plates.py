
from tqdm import tqdm

import jax
import jax.numpy as jnp
from jax.tree_util import Partial

from updec import *


## Conda hook on windows: C:\Users\<username>\anaconda3\shell\condabin\conda-hook.ps1

facet_types_vel = {"Wall":"d", "Inflow":"d", "Outflow":"n"}
facet_types_phi = {"Wall":"n", "Inflow":"n", "Outflow":"d"}

cloud_vel = GmshCloud(filename="./demos/meshes/channel.py", facet_types=facet_types_vel, support_size="max")
cloud_phi = GmshCloud(filename="./demos/meshes/channel.msh", facet_types=facet_types_phi, support_size="max")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5,1.4*2), sharex=True)
cloud_vel.visualize_cloud(ax=ax1, s=12, title="Cloud for velocity", xlabel=False);
cloud_phi.visualize_cloud(ax=ax2, s=12, title=r"Cloud for $\phi$");

print("Total number of nodes:", cloud_vel.N)
print(cloud_vel.outward_normals)
plt.show()
exit(0)


RBF = polyharmonic      ## Can define which rbf to use
MAX_DEGREE = 4


Re = 200
RHO = 1          ## Water
# RHO = 1           ## air
# NU = 1e-3           ## water
NU = 1           ## water
# NU = 1e-5         ## air
# sigma = 0.07
DT = 1e-6


pa = 101325.0
# pa = 0.0
beta = 0.

@Partial(jax.jit, static_argnums=[2,3])
def diff_operator_u(x, center=None, rbf=None, monomial=None, fields=None):
    U_prev = jnp.array([fields[0], fields[1]])
    u_val = nodal_value(x, center, rbf, monomial)
    u_grad = nodal_gradient(x, center, rbf, monomial)
    u_lap = nodal_laplacian(x, center, rbf, monomial)
    return u_val + DT*jnp.dot(U_prev, u_grad) - DT*NU*u_lap

@Partial(jax.jit, static_argnums=[2])
def rhs_operator_u(x, centers=None, rbf=None, fields=None):
    u_prev = value(x, fields[:, 0], centers, rbf)
    grad_px = gradient(x, fields[:, 1], centers, rbf)[0]
    return  u_prev - beta*(DT*grad_px/RHO)



@Partial(jax.jit, static_argnums=[2,3])
def diff_operator_v(x, center=None, rbf=None, monomial=None, fields=None):
    U_prev = jnp.array([fields[0], fields[1]])
    v_val = nodal_value(x, center, rbf, monomial)
    v_grad = nodal_gradient(x, center, rbf, monomial)
    v_lap = nodal_laplacian(x, center, rbf, monomial)
    return v_val + DT*jnp.dot(U_prev, v_grad) - DT*NU*v_lap

@Partial(jax.jit, static_argnums=[2])
def rhs_operator_v(x, centers=None, rbf=None, fields=None):
    v_prev = value(x, fields[:, 0], centers, rbf)
    grad_py = gradient(x, fields[:, 1], centers, rbf)[1]
    return  v_prev - beta*(DT *grad_py/RHO)



@Partial(jax.jit, static_argnums=[2,3])
def diff_operator_phi(x, center=None, rbf=None, monomial=None, fields=None):
    return nodal_laplacian(x, center, rbf, monomial)

@Partial(jax.jit, static_argnums=[2])
def rhs_operator_phi(x, centers=None, rbf=None, fields=None):
    return RHO * divergence(x, fields[:, :2], centers, rbf) / DT
    # return RHO * jnp.min(jnp.array([divergence(x, fields[:, :2], centers, rbf), 0.001])) / DT


## Initial states, all defined on cloud_vel
u = jnp.zeros((cloud_vel.N,))
v = jnp.zeros((cloud_vel.N,))

p_ = jnp.zeros((cloud_phi.N,))       ## on cloud_phi        ##TODO set this to p_a on Outlet
out_nodes = jnp.array(cloud_phi.facet_nodes["Outflow"])
p_ = p_.at[out_nodes].set(pa)




# parabolic = jax.jit(lambda x: 1.5 - 6*(x[1]**2))
parabolic = jax.jit(lambda x: 1.)
atmospheric = jax.jit(lambda x: pa*(1. - beta))     ##TODO Carefull: beta and pa must never change
zero = jax.jit(lambda x: 0.0)

bc_u = {"Wall":zero, "Inflow":parabolic, "Outflow":zero}
bc_v = {"Wall":zero, "Inflow":zero, "Outflow":zero}
bc_phi = {"Wall":zero, "Inflow":zero, "Outflow":atmospheric}



nb_iter = 1
all_u = []
all_v = []
all_vel = []
all_p = []

for i in tqdm(range(nb_iter)):
    # print("Starting iteration %d" % i)

    ## TODO Interpolate p and gradphi onto cloud_vel
    p = interpolate_field(p_, cloud_phi, cloud_vel)

    usol = pde_solver(diff_operator=diff_operator_u, 
                    diff_args=[u, v],
                    rhs_operator = rhs_operator_u, 
                    rhs_args=[u, p], 
                    cloud = cloud_vel, 
                    boundary_conditions = bc_u,
                    rbf=RBF,
                    max_degree=MAX_DEGREE)

    vsol = pde_solver(diff_operator=diff_operator_v,
                    diff_args=[u, v],
                    rhs_operator = rhs_operator_v,
                    rhs_args=[v, p], 
                    cloud = cloud_vel, 
                    boundary_conditions = bc_v,
                    rbf=RBF,
                    max_degree=MAX_DEGREE)

    ustar , vstar = usol.vals, vsol.vals     ## Star
    Ustar = jnp.stack([ustar,vstar], axis=-1)

    ## TODO Interpolaate Ustar onto cloud_phi
    u_ = interpolate_field(ustar, cloud_vel, cloud_phi)
    v_ = interpolate_field(vstar, cloud_vel, cloud_phi)

    phisol_ = pde_solver(diff_operator=diff_operator_phi,
                    rhs_operator = rhs_operator_phi,
                    rhs_args=[u_,v_], 
                    cloud = cloud_phi, 
                    boundary_conditions = bc_phi,
                    rbf=RBF,
                    max_degree=MAX_DEGREE)

    p_ = beta*p_ + phisol_.vals
    gradphi_ = gradient_vec(cloud_phi.sorted_nodes, phisol_.coeffs, cloud_phi.sorted_nodes, RBF)        ## TODO use Pde_solver here instead ?

    ## TODO Interpolate p and gradphi onto cloud_vel
    gradphi = interpolate_field(gradphi_, cloud_phi, cloud_vel)

    U = Ustar - (gradphi*DT/RHO)
    u, v = U[:,0], U[:,1]
    vel = jnp.linalg.norm(U, axis=-1)

    # fig, ax = plt.subplots(4, 1, figsize=(9.5,1.4*4), sharex=True)
    # iter_str = " at iteration " + str(i+1)
    # cloud_vel.visualize_field(u, cmap="jet", title="Velocity along x"+iter_str, ax=ax[0], xlabel=False);
    # cloud_vel.visualize_field(v, cmap="jet", title="Velocity along y"+iter_str, ax=ax[1], xlabel=False);
    # cloud_vel.visualize_field(vel, cmap="jet", title="Velocity norm"+iter_str, ax=ax[2], xlabel=False);
    # cloud_phi.visualize_field(p_, cmap="jet", title="Pressure"+iter_str, ax=ax[3]);
    # plt.savefig('demos/temp/solutions_iter_'+str(i)+'.png')
    all_u.append(u)
    all_v.append(v)
    all_vel.append(vel)
    all_p.append(p_)


filename = 'demos/temp/video.mp4'
cloud_vel.animate_fields([all_u, all_v, all_vel, all_p], filename=filename, cmaps=["jet", "jet", "jet", "magma"], titles=[r"$u$", r"$v$", "Velocity amplitude", "Pressure"], duration=10, figsize=(9.5,1.4*4));

plt.show()
