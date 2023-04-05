
# %%
import jax
import jax.numpy as jnp
from jax.tree_util import Partial
from functools import partial
from tqdm import tqdm
jax.config.update('jax_platform_name', 'cpu')           ## TODO Slow on GPU on Daffy Duck !

from updec import *


# EXPERIMENET_ID = random_name()
EXPERIMENET_ID = "ChannelAdjoint2"
DATAFOLDER = "./data/" + EXPERIMENET_ID +"/"
make_dir(DATAFOLDER)


# %%
### Constants for the problem

RBF = polyharmonic      ## Can define which rbf to use
MAX_DEGREE = 4

Re = 100
Pa = 0.0

# NB_ITER = 12
NB_ITER = 12


# %%


facet_types_vel = {"Wall":"d", "Inflow":"d", "Outflow":"r", "Blowing":"d", "Suction":"d"}
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
def diff_operator_l1(x, center=None, rbf=None, monomial=None, fields=None):
    lambda_val = jnp.array([nodal_value(x, center, rbf, monomial), fields[0]])
    U_val = jnp.array([fields[1], fields[2]])
    U_grad_T = jnp.array([fields[3], fields[4]])
    lambda_grad = nodal_gradient(x, center, rbf, monomial)
    lambda_lap = nodal_laplacian(x, center, rbf, monomial)
    return jnp.dot(lambda_val, U_grad_T) + jnp.dot(U_val, lambda_grad) - lambda_lap/Re

@Partial(jax.jit, static_argnums=[2])
def rhs_operator_l1(x, centers=None, rbf=None, fields=None):
    grad_pi_x = gradient(x, fields[:, 0], centers, rbf)[0]
    return -grad_pi_x



@Partial(jax.jit, static_argnums=[2,3])
def diff_operator_l2(x, center=None, rbf=None, monomial=None, fields=None):
    lambda_val = jnp.array([fields[0], nodal_value(x, center, rbf, monomial)])
    U_val = jnp.array([fields[1], fields[2]])
    U_grad_T = jnp.array([fields[3], fields[4]])
    lambda_grad = nodal_gradient(x, center, rbf, monomial)
    lambda_lap = nodal_laplacian(x, center, rbf, monomial)
    return jnp.dot(lambda_val, U_grad_T) + jnp.dot(U_val, lambda_grad) - lambda_lap/Re

@Partial(jax.jit, static_argnums=[2])
def rhs_operator_l2(x, centers=None, rbf=None, fields=None):
    grad_pi_y = gradient(x, fields[:, 0], centers, rbf)[1]
    return  -grad_pi_y



@Partial(jax.jit, static_argnums=[2,3])
def diff_operator_mu(x, center=None, rbf=None, monomial=None, fields=None):
    return nodal_laplacian(x, center, rbf, monomial)

@Partial(jax.jit, static_argnums=[2])
def rhs_operator_mu(x, centers=None, rbf=None, fields=None):
    return divergence(x, fields[:, :2], centers, rbf)


## Initial states, all defined on cloud_vel
lambda1 = jnp.zeros((cloud_vel.N,))
# u = u.at[in_nodes].set(1.)
lambda2 = jnp.zeros((cloud_vel.N,))
out_nodes_lamb = jnp.array(cloud_vel.facet_nodes["Outflow"])

pi_ = jnp.zeros((cloud_phi.N,))       ## on cloud_phi        ##TODO set this to p_a on Outlet
out_nodes_pi = jnp.array(cloud_phi.facet_nodes["Outflow"])
pi_ = pi_.at[out_nodes_pi].set(Pa)


parabolic = jax.jit(lambda x: 4*x[1]*(1.-x[1]))
zero = jax.jit(lambda x: 0.)


## Quantities for Robin boundary condition
u = v = jnp.ones((cloud_vel.N))     ## TODO see this
u1 = u[out_nodes_lamb]      ## For robin BCs
u2 = v[out_nodes_lamb]
u_parab = jax.vmap(parabolic)(cloud_vel.sorted_nodes[out_nodes_pi])
pi_out = pi_[out_nodes_pi]


grad_u = cartesian_gradient_vec(range(cloud_vel.N), u, cloud_vel)
grad_v = cartesian_gradient_vec(range(cloud_vel.N), v, cloud_vel)
# temp = jnp.zeros(grad_uT.shape)
# temp[:, :] = grad_uT[:, :]
# grad_uT[:, 1] = grad_vT[:, 0]
# grad_vT[:, 0] = temp[:, 1]


l1_list = [lambda1]
l2_list = [lambda2]
lnorm_list = []
pi_list = [pi_]

for i in tqdm(range(NB_ITER)):
    # print("Starting iteration %d" % i)


    bc_u = {"Wall":zero, "Inflow":zero, "Outflow":(-u2*Re, u1*Re), "Cylinder":zero, "Blowing":zero, "Suction":zero}
    bc_v = {"Wall":zero, "Inflow":zero, "Outflow":((u1-u_parab+pi_out)*Re, u1*Re), "Cylinder":zero, "Blowing":zero, "Suction":zero}
    bc_phi = {"Wall":zero, "Inflow":zero, "Outflow":zero, "Cylinder":zero, "Blowing":zero, "Suction":zero}



    ## TODO Interpolate p and gradphi onto cloud_vel
    pi = interpolate_field(pi_, cloud_phi, cloud_vel)

    l1sol = pde_solver(diff_operator=diff_operator_l1, 
                    diff_args=[lambda2, u, v, grad_u[:,0], grad_v[:,0]],
                    rhs_operator = rhs_operator_l1, 
                    rhs_args=[pi], 
                    cloud = cloud_vel,
                    boundary_conditions = bc_u,
                    rbf=RBF,
                    max_degree=MAX_DEGREE)

    l2sol = pde_solver(diff_operator=diff_operator_l2,
                    diff_args=[lambda1, u, v, grad_u[:,1], grad_v[:,1]],
                    rhs_operator = rhs_operator_l2,
                    rhs_args=[pi], 
                    cloud = cloud_vel, 
                    boundary_conditions = bc_v,
                    rbf=RBF,
                    max_degree=MAX_DEGREE)

    l1star , l2star = l1sol.vals, l2sol.vals     ## Star
    Lstar = jnp.stack([l1star,l2star], axis=-1)

    ## TODO Interpolate Ustar onto cloud_phi
    l1_ = interpolate_field(l1star, cloud_vel, cloud_phi)
    l2_ = interpolate_field(l2star, cloud_vel, cloud_phi)

    musol_ = pde_solver(diff_operator=diff_operator_mu,
                    rhs_operator = rhs_operator_mu,
                    rhs_args=[l1_, l2_], 
                    cloud = cloud_phi, 
                    boundary_conditions = bc_phi,
                    rbf=RBF,
                    max_degree=MAX_DEGREE)

    pi_ = pi_ + musol_.vals
    # pi_ = pi_.at[out_nodes_pi].set(Pa)      ## TODO Rebmember this one up too

    gradmu_ = gradient_vec(cloud_phi.sorted_nodes, musol_.coeffs, cloud_phi.sorted_nodes, RBF)        ## TODO use Pde_solver here instead ?
    # gradmu_ = cartesian_gradient_vec(range(cloud_phi.N), musol_.vals, cloud_phi)

    ## TODO Interpolate p and gradphi onto cloud_vel
    gradmu = interpolate_field(gradmu_, cloud_phi, cloud_vel)

    L = Lstar - gradmu
    l1, l2 = L[:,0], L[:,1]
    lnorm = jnp.linalg.norm(L, axis=-1)

    print("u max:", jnp.max(l1))
    print("v max:", jnp.max(l2))

    l1_list.append(u)
    l2_list.append(v)
    lnorm_list.append(lnorm)
    pi_list.append(pi_)


# %%

print("\nSimulation complete. Saving all files to %s" % DATAFOLDER)


renum_map_vel = jnp.array(list(cloud_vel.renumbering_map.keys()))
renum_map_p = jnp.array(list(cloud_phi.renumbering_map.keys()))

jnp.savez(DATAFOLDER+'u.npz', renum_map_vel, jnp.stack(l1_list, axis=0))
jnp.savez(DATAFOLDER+'v.npz', renum_map_vel, jnp.stack(l2_list, axis=0))
jnp.savez(DATAFOLDER+'vel.npz', renum_map_vel, jnp.stack(lnorm_list, axis=0))
jnp.savez(DATAFOLDER+'p.npz', renum_map_p, jnp.stack(pi_list, axis=0))

# plt.show()


# %%

## Best results are stored in folder 06719

print("\nSaving complete. Now running visualisation ...")

pyvista_animation(DATAFOLDER, "v", duration=5, vmin=None, vmax=None)


# %%


grad_l1 = cartesian_gradient_vec(range(cloud_vel.N), l1, cloud_vel)
in_nodes_lamb = jnp.array(cloud_vel.facet_nodes["Inflow"])
in_nodes_pi = jnp.array(cloud_phi.facet_nodes["Inflow"])

gradJ = pi_[in_nodes_pi] - grad_l1[in_nodes_lamb, 0]/Re

print("gradient of cost objective J:", gradJ)

### Now let's compute gradient of cost
