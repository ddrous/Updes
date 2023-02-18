
from tqdm import tqdm

import jax
import jax.numpy as jnp
from jax.tree_util import Partial

from updec import *


facet_types_vel = {"Wall":"d", "Square":"d", "Inflow":"d", "Outflow":"n"}
facet_types_phi = {"Wall":"n", "Square":"n", "Inflow":"n", "Outflow":"d"}

cloud_vel = GmshCloud(filename="./demos/meshes/cylinder.msh", facet_types=facet_types_vel, support_size="max")       ## TODO do not hardcode this path
cloud_phi = GmshCloud(filename="./demos/meshes/cylinder.msh", facet_types=facet_types_phi, support_size="max")       ## TODO do not hardcode this path


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5,1.4*2), sharex=True)
cloud_vel.visualize_cloud(ax=ax1, s=12, title="Cloud for velocity", xlabel=False);
cloud_phi.visualize_cloud(ax=ax2, s=12, title=r"Cloud for $\phi$");

print("Total number of nodes:", cloud_vel.N)
# plt.show()
# exit(0)


RBF = polyharmonic      ## Can define which rbf to use
MAX_DEGREE = 4

parabolic = jax.jit(lambda x: 1.5 - 6*(x[1]**2))
zero = jax.jit(lambda x: 0.0)
one = jax.jit(lambda x: -0.25)

bc_u = {"Wall":zero, "Square":zero, "Inflow":parabolic, "Outflow":zero}
bc_v = {"Wall":zero, "Square":zero, "Inflow":zero, "Outflow":zero}
bc_phi = {"Wall":zero, "Square":zero, "Inflow":zero, "Outflow":zero}


Du = Re = 200
DeltaT = 0.0001

@Partial(jax.jit, static_argnums=[2,3])
def diff_operator_u(x, center=None, rbf=None, monomial=None, fields=None):
    u = nodal_value(x, center, rbf, monomial)
    # u = nodal_laplacian(x, center, rbf, monomial)
    return u

@Partial(jax.jit, static_argnums=[2])
def rhs_operator_u(x, centers=None, rbf=None, fields=None):
    u = value(x, fields[:, 0], centers, rbf) 
    v = value(x, fields[:, 1], centers, rbf) 
    U = jnp.array([u,v])

    grad_u = gradient(x, fields[:, 0], centers, rbf)
    grad_px = gradient(x, fields[:, 2], centers, rbf)[0]
    lap_u = laplacian(x, fields[:, 0], centers, rbf)

    return  u + DeltaT*(jnp.dot(U, grad_u) - grad_px + lap_u/Re)



@Partial(jax.jit, static_argnums=[2,3])
def diff_operator_v(x, center=None, rbf=None, monomial=None, fields=None):
    v = nodal_value(x, center, rbf, monomial)
    return  v

@Partial(jax.jit, static_argnums=[2])
def rhs_operator_v(x, centers=None, rbf=None, fields=None):
    u = value(x, fields[:, 0], centers, rbf) 
    v = value(x, fields[:, 1], centers, rbf) 
    U = jnp.array([u,v])

    grad_v = gradient(x, fields[:, 1], centers, rbf)
    grad_py = gradient(x, fields[:, 2], centers, rbf)[1]
    lap_v = laplacian(x, fields[:, 1], centers, rbf)

    return v + DeltaT*(jnp.dot(U, grad_v) - grad_py + lap_v/Re)



@Partial(jax.jit, static_argnums=[2,3])
def diff_operator_phi(x, center=None, rbf=None, monomial=None, fields=None):
    return nodal_laplacian(x, center, rbf, monomial)

@Partial(jax.jit, static_argnums=[2])
def rhs_operator_phi(x, centers=None, rbf=None, fields=None):
    return divergence(x, fields[:, :2], centers, rbf)


## Initial states, all defined on cloud_vel
u = jnp.zeros((cloud_vel.N,))
v = jnp.zeros((cloud_vel.N,))
p_ = jnp.zeros((cloud_phi.N,))       ## on cloud_phi        ##TODO set this to p_a on Outlet

nb_iter = 5
# plt.show()
# exit()

for i in tqdm(range(nb_iter)):
    # print("Starting iteration %d" % i)

    ## TODO Interpolate p and gradphi onto cloud_vel
    p = interpolate_field(p_, cloud_phi, cloud_vel)

    usol = pde_solver(diff_operator=diff_operator_u, 
                    # diff_args=[v], 
                    rhs_operator = rhs_operator_u, 
                    rhs_args=[u,v,p], 
                    cloud = cloud_vel, 
                    boundary_conditions = bc_u,
                    rbf=RBF,
                    max_degree=MAX_DEGREE)

    vsol = pde_solver(diff_operator=diff_operator_v,
                    # diff_args=[v],
                    rhs_operator = rhs_operator_v,
                    rhs_args=[u,v,p], 
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

    p_ = p_ + phisol_.vals
    gradphi_ = gradient_vec(cloud_phi.sorted_nodes, phisol_.coeffs, cloud_phi.sorted_nodes, RBF)        ## TODO use Pde_solver here instead ?

    ## TODO Interpolate p and gradphi onto cloud_vel
    gradphi = interpolate_field(gradphi_, cloud_phi, cloud_vel)

    U = Ustar - gradphi
    u, v = U[:,0], U[:,1]
    vel = jnp.linalg.norm(U, axis=-1)

    fig, ax = plt.subplots(4, 1, figsize=(9.5,1.4*4), sharex=True)
    iter_str = " at iteration " + str(i+1)
    cloud_vel.visualize_field(u, cmap="jet", title="Velocity along x"+iter_str, ax=ax[0], xlabel=False);
    cloud_vel.visualize_field(v, cmap="jet", title="Velocity along y"+iter_str, ax=ax[1], xlabel=False);
    cloud_vel.visualize_field(vel, cmap="jet", title="Velocity norm"+iter_str, ax=ax[2], xlabel=False);
    cloud_phi.visualize_field(p_, cmap="jet", title="Pressure"+iter_str, ax=ax[3]);
    plt.savefig('demos/temp/solutions_iter_'+str(i)+'.png')

plt.show()
