
# %%
import jax
import jax.numpy as jnp
from jax.tree_util import Partial
from functools import partial
from tqdm import tqdm
jax.config.update('jax_platform_name', 'cpu')           ## TODO Slow on GPU on Daffy Duck !

from updec import *


# EXPERIMENET_ID = random_name()
EXPERIMENET_ID = "ChannelAdjoint1"
DATAFOLDER = "./data/" + EXPERIMENET_ID +"/"
make_dir(DATAFOLDER)


# %%
### Constants for the problem

# EPS=1e-2
RBF = polyharmonic      ## Can define which rbf to use
# RBF = partial(inverse_multiquadric, eps=10.0)
# RBF = partial(gaussian, eps=1e2)z
# RBF = partial(thin_plate, a=3)

MAX_DEGREE = 4

Re = 100

Pa = 0.0

# NB_ITER = 12
NB_ITER = 2



# %%


facet_types_vel = {"Wall":"d", "Inflow":"d", "Outflow":"n", "Blowing":"d", "Suction":"d"}
facet_types_phi = {"Wall":"n", "Inflow":"n", "Outflow":"d", "Blowing":"n", "Suction":"n"}
# facet_types_vel = {"Wall":"d", "Inflow":"d", "Outflow":"n", "Cylinder":"d"}
# facet_types_phi = {"Wall":"n", "Inflow":"n", "Outflow":"d", "Cylinder":"n"}

cloud_vel = GmshCloud(filename="./meshes/channel_blow_suc.py", facet_types=facet_types_vel, mesh_save_location=DATAFOLDER)             ## TODO Pass the savelocation here
cloud_phi = GmshCloud(filename=DATAFOLDER+"mesh.msh", facet_types=facet_types_phi)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6,3*2), sharex=True)
cloud_vel.visualize_cloud(ax=ax1, s=6, title="Cloud for velocity", xlabel=False);
cloud_phi.visualize_cloud(ax=ax2, s=6, title=r"Cloud for $\phi$");

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.5*2,5))
cloud_vel.visualize_normals(ax=ax1, title="Normals for velocity")
cloud_phi.visualize_normals(ax=ax2,title="Normals for phi", zoom_region=(0.25,1.25,-0.1,1.1))



# %%
print(f"\nStarting RBF simulation with {cloud_vel.N} nodes\n")


@Partial(jax.jit, static_argnums=[2,3])
def diff_operator_u(x, center=None, rbf=None, monomial=None, fields=None):
    U_prev = jnp.array([fields[0], fields[1]])
    u_grad = nodal_gradient(x, center, rbf, monomial)
    u_lap = nodal_laplacian(x, center, rbf, monomial)
    return jnp.dot(U_prev, u_grad) - u_lap/Re

@Partial(jax.jit, static_argnums=[2])
def rhs_operator_u(x, centers=None, rbf=None, fields=None):
    grad_px = gradient(x, fields[:, 0], centers, rbf)[0]
    return -grad_px



@Partial(jax.jit, static_argnums=[2,3])
def diff_operator_v(x, center=None, rbf=None, monomial=None, fields=None):
    U_prev = jnp.array([fields[0], fields[1]])
    v_grad = nodal_gradient(x, center, rbf, monomial)
    v_lap = nodal_laplacian(x, center, rbf, monomial)
    return jnp.dot(U_prev, v_grad) - v_lap/Re

@Partial(jax.jit, static_argnums=[2])
def rhs_operator_v(x, centers=None, rbf=None, fields=None):
    grad_py = gradient(x, fields[:, 0], centers, rbf)[1]
    return  -grad_py



@Partial(jax.jit, static_argnums=[2,3])
def diff_operator_phi(x, center=None, rbf=None, monomial=None, fields=None):
    return nodal_laplacian(x, center, rbf, monomial)

@Partial(jax.jit, static_argnums=[2])
def rhs_operator_phi(x, centers=None, rbf=None, fields=None):
    return divergence(x, fields[:, :2], centers, rbf)


## Initial states, all defined on cloud_vel
u = jnp.zeros((cloud_vel.N,))
# in_nodes = jnp.array(cloud_phi.facet_nodes["Inflow"])
# u = u.at[in_nodes].set(1.)
v = jnp.zeros((cloud_vel.N,))

p_ = jnp.zeros((cloud_phi.N,))       ## on cloud_phi        ##TODO set this to p_a on Outlet
out_nodes = jnp.array(cloud_phi.facet_nodes["Outflow"])
p_ = p_.at[out_nodes].set(Pa)



parabolic = jax.jit(lambda x: 4*x[1]*(1.-x[1]))
blowing = jax.jit(lambda x: 0.3)
suction = jax.jit(lambda x: 0.3)
one = jax.jit(lambda x: 1.)
zero = jax.jit(lambda x: 0.)

# bc_u = {"Wall":zero, "Inflow":one, "Outflow":zero}
# bc_v = {"Wall":zero, "Inflow":zero, "Outflow":zero}
# bc_phi = {"Wall":zero, "Inflow":zero, "Outflow":zero}

bc_u = {"Wall":zero, "Inflow":parabolic, "Outflow":zero, "Cylinder":zero, "Blowing":zero, "Suction":zero}
bc_v = {"Wall":zero, "Inflow":zero, "Outflow":zero, "Cylinder":zero, "Blowing":blowing, "Suction":suction}
bc_phi = {"Wall":zero, "Inflow":zero, "Outflow":zero, "Cylinder":zero, "Blowing":zero, "Suction":zero}


u_list = [u]
v_list = [v]
vel_list = []
p_list = [p_]

for i in tqdm(range(NB_ITER)):
    # print("Starting iteration %d" % i)

    ## TODO Interpolate p and gradphi onto cloud_vel
    p = interpolate_field(p_, cloud_phi, cloud_vel)

    usol = pde_solver(diff_operator=diff_operator_u, 
                    diff_args=[u, v],
                    rhs_operator = rhs_operator_u, 
                    rhs_args=[p], 
                    cloud = cloud_vel, 
                    boundary_conditions = bc_u,
                    rbf=RBF,
                    max_degree=MAX_DEGREE)

    vsol = pde_solver(diff_operator=diff_operator_v,
                    diff_args=[u, v],
                    rhs_operator = rhs_operator_v,
                    rhs_args=[p], 
                    cloud = cloud_vel, 
                    boundary_conditions = bc_v,
                    rbf=RBF,
                    max_degree=MAX_DEGREE)

    ustar , vstar = usol.vals, vsol.vals     ## Star
    Ustar = jnp.stack([ustar,vstar], axis=-1)

    print("u star max:", jnp.max(ustar))
    print("u star max loc:", cloud_vel.sorted_nodes[jnp.argmax(ustar)])
    print("v star max:", jnp.max(vstar))
    print("v star max loc:", cloud_vel.sorted_nodes[jnp.argmax(vstar)])

    ## TODO Interpolate Ustar onto cloud_phi
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
    # gradphi_ = gradient_vec(cloud_phi.sorted_nodes, phisol_.coeffs, cloud_phi.sorted_nodes, RBF)        ## TODO use Pde_solver here instead ?
    # gradphi_ = enforce_cartesian_gradient_neumann(phisol_.vals, gradphi_, bc_phi, cloud_phi)
    gradphi_ = cartesian_gradient_vec(range(cloud_phi.N), phisol_.vals, cloud_phi)

    ## TODO Interpolate p and gradphi onto cloud_vel
    gradphi = interpolate_field(gradphi_, cloud_phi, cloud_vel)

    U = Ustar - gradphi
    u, v = U[:,0], U[:,1]
    vel = jnp.linalg.norm(U, axis=-1)

    print("u max:", jnp.max(u))
    print("u max loc:", cloud_vel.sorted_nodes[jnp.argmax(u)])
    print("v max:", jnp.max(v))
    print("v max loc:", cloud_vel.sorted_nodes[jnp.argmax(v)])

    u_list.append(u)
    v_list.append(v)
    vel_list.append(vel)
    p_list.append(p_)


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

## Best results are stored in folder 06719

print("\nSaving complete. Now running visualisation ...")

pyvista_animation(DATAFOLDER, "p", duration=5, vmin=0, vmax=1.1)


# %%
