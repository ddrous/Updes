
# %%
import jax
import jax.numpy as jnp
from jax.tree_util import Partial
from tqdm import tqdm

from updec import *

### Save location
EXPERIMENET_ID = random_name()
DATAFOLDER = "./data/" + EXPERIMENET_ID +"/"
make_dir(DATAFOLDER)

### Constants for the problem
# EPS = 10.0
# RBF = partial(gaussian, eps=EPS)
# RBF = polyharmonic      ## Can define which rbf to use
# MAX_DEGREE = 4

RBF = partial(thin_plate, a=3)
MAX_DEGREE = 2

a, b, c = 1.5, -2., 0.5
Re = 10
RHO=1.
NU=1/Re
Pa = 101325.
# Pa = 0.

DT = 1e-7

NB_ITER = 3
MAX_REFINEMENTS = 10
TOL = 1e-3



# %%


facet_types_vel = {"Wall":"d", "Inflow":"d", "Outflow":"n"}
facet_types_phi = {"Wall":"n", "Inflow":"n", "Outflow":"d"}
# facet_types_vel = {"Wall":"d", "Inflow":"d", "Outflow":"n", "Cylinder":"d"}
# facet_types_phi = {"Wall":"n", "Inflow":"n", "Outflow":"d", "Cylinder":"n"}

cloud_vel = GmshCloud(filename="./meshes/channel_2.py", facet_types=facet_types_vel, mesh_save_location=DATAFOLDER)    ## TODO Pass the savelocation here
cloud_phi = GmshCloud(filename=DATAFOLDER+"mesh.msh", facet_types=facet_types_phi)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5,1.4*2), sharex=True)
cloud_vel.visualize_cloud(ax=ax1, s=6, title="Cloud for velocity", xlabel=False);
cloud_phi.visualize_cloud(ax=ax2, s=6, title=r"Cloud for $\phi$");



# %%
print(f"\nStarting RBF simulations with {cloud_vel.N} nodes\n")


@Partial(jax.jit, static_argnums=[2,3])
def diff_operator_u(x, center=None, rbf=None, monomial=None, fields=None):
    U = jnp.array([fields[0], fields[1]])
    u_val = nodal_value(x, center, rbf, monomial)
    u_grad = nodal_gradient(x, center, rbf, monomial)
    u_lap = nodal_laplacian(x, center, rbf, monomial)
    return (RHO*a*u_val/DT) + RHO*jnp.dot(U, u_grad) - (NU*u_lap)

@Partial(jax.jit, static_argnums=[2])
def rhs_operator_u(x, centers=None, rbf=None, fields=None):
    u_rhs = value(x, fields[:, 0], centers, rbf)
    grad_px = gradient(x, fields[:, 1], centers, rbf)[0]
    return -(RHO*u_rhs/DT) - grad_px



@Partial(jax.jit, static_argnums=[2,3])
def diff_operator_v(x, center=None, rbf=None, monomial=None, fields=None):
    U = jnp.array([fields[0], fields[1]])
    v_val = nodal_value(x, center, rbf, monomial)
    v_grad = nodal_gradient(x, center, rbf, monomial)
    v_lap = nodal_laplacian(x, center, rbf, monomial)
    return (RHO*a*v_val/DT) + RHO*jnp.dot(U, v_grad) - (NU*v_lap)

@Partial(jax.jit, static_argnums=[2])
def rhs_operator_v(x, centers=None, rbf=None, fields=None):
    v_rhs = value(x, fields[:, 0], centers, rbf)
    grad_py = gradient(x, fields[:, 1], centers, rbf)[1]
    return -(RHO*v_rhs/DT) - grad_py



@Partial(jax.jit, static_argnums=[2,3])
def diff_operator_phi(x, center=None, rbf=None, monomial=None, fields=None):
    return nodal_laplacian(x, center, rbf, monomial)

@Partial(jax.jit, static_argnums=[2])
def rhs_operator_phi(x, centers=None, rbf=None, fields=None):
    return RHO*a*divergence(x, fields[:, :2], centers, rbf)/DT


## Initial states, all defined on cloud_vel
u_now = u_prev = jnp.zeros((cloud_vel.N,))
v_now = v_prev = jnp.zeros((cloud_vel.N,))
in_nodes = jnp.array(cloud_vel.facet_nodes["Inflow"])
u_prev = u_prev.at[in_nodes].set(1.)
u_now = u_now.at[in_nodes].set(1.)

# p_now_ = p_prev_ = jnp.zeros((cloud_phi.N,))
p_now_ = p_prev_ = jnp.ones((cloud_phi.N,))*Pa
out_nodes = jnp.array(cloud_phi.facet_nodes["Outflow"])
p_prev_ = p_prev_.at[out_nodes].set(Pa)
p_now_ = p_now_.at[out_nodes].set(Pa)



# parabolic = jax.jit(lambda x: 1.5 - 6*(x[1]**2))
ones = jax.jit(lambda x: 1.)
zero = jax.jit(lambda x: 0.)

bc_u = {"Wall":zero, "Inflow":ones, "Outflow":zero}
bc_v = {"Wall":zero, "Inflow":zero, "Outflow":zero}
bc_phi = {"Wall":zero, "Inflow":zero, "Outflow":zero}
# bc_u = {"Wall":zero, "Inflow":ones, "Outflow":zero, "Cylinder":zero}
# bc_v = {"Wall":zero, "Inflow":zero, "Outflow":zero, "Cylinder":zero}
# bc_phi = {"Wall":zero, "Inflow":zero, "Outflow":zero, "Cylinder":zero}



u_list = [u_prev, u_now]
v_list = [v_prev, v_now]
vel_list = []   ## Compute the original two norms
p_list = [p_prev_, p_now_]

for i in tqdm(range(NB_ITER)):
    # print("Starting iteration %d" % i)

    ## For all innner loop
    u_rhs = b*u_now + c*u_prev
    v_rhs = b*v_now + c*v_prev

    ## Only for k=0
    u_next_prev = u_now
    v_next_prev = v_now
    p_next_prev_ = p_now_

    for k in range(MAX_REFINEMENTS):
        ## TODO Interpolate p and gradphi onto cloud_vel
        p_next_prev = interpolate_field(p_next_prev_, cloud_phi, cloud_vel)
        U_next_prev = jnp.stack([u_next_prev, v_next_prev], axis=-1)

        usol = pde_solver(diff_operator=diff_operator_u,
                        diff_args=[u_next_prev, v_next_prev],
                        rhs_operator = rhs_operator_u,
                        rhs_args=[u_rhs, p_next_prev],
                        cloud = cloud_vel,
                        boundary_conditions = bc_u,
                        rbf=RBF,
                        max_degree=MAX_DEGREE)
        u_star = usol.vals

        vsol = pde_solver(diff_operator=diff_operator_v,
                        diff_args=[u_next_prev, v_next_prev],
                        rhs_operator = rhs_operator_v,
                        rhs_args=[v_rhs, p_next_prev], 
                        cloud = cloud_vel,
                        boundary_conditions = bc_v,
                        rbf=RBF,
                        max_degree=MAX_DEGREE)
        v_star = vsol.vals
        # print("v Max:", jnp.max(v_star))

        U_star = jnp.stack([u_star, v_star], axis=-1)
        U_star_ = interpolate_field(U_star, cloud_vel, cloud_phi)
        u_star_, v_star_ = U_star_[:,0], U_star_[:,1]

        phisol_ = pde_solver(diff_operator=diff_operator_phi,
                        rhs_operator = rhs_operator_phi,
                        rhs_args=[u_star_,v_star_],
                        cloud = cloud_phi,
                        boundary_conditions = bc_phi,
                        rbf=RBF,
                        max_degree=MAX_DEGREE)
        # print("phi Max:", jnp.max(phisol_.vals))

        p_next_now_ = p_next_prev_ + phisol_.vals

        p_next_now_ = p_next_now_.at[out_nodes].set(Pa)
        ## Also set the pressure at every other boundary point before correcting velocity

        gradphi_ = gradient_vec(cloud_phi.sorted_nodes, phisol_.coeffs, cloud_phi.sorted_nodes, RBF)        ## TODO use Pde_solver here instead ?
        gradphi = interpolate_field(gradphi_, cloud_phi, cloud_vel)
        U_next_now = U_star - DT*gradphi/(RHO*a)
        u_next_now, v_next_now = U_next_now[:,0], U_next_now[:,1]

        vel_change = jnp.linalg.norm(U_next_now-U_next_prev, axis=0).sum()
        if vel_change <= TOL:
            break

        # else:
        u_next_prev = u_next_now 
        v_next_prev = v_next_now 
        p_next_prev_ = p_next_now_

        print("Refinement iteration number:", k+1)
        print("Velocity change:", vel_change)
        print("u max:", jnp.max(u_star))
        print("v max:", jnp.max(v_star))
        print("u max location:", cloud_vel.sorted_nodes[jnp.argmax(u_star)])
        print("v max location:", cloud_vel.sorted_nodes[jnp.argmax(v_star)])
        print()

        if k==MAX_REFINEMENTS-1:
            print("Reached maximum number of refinements without !")

    u_prev = u_now.copy()
    v_pref = v_now.copy()
    p_prev_ = p_now_.copy()

    u_now = u_next_now.copy()
    v_now = v_next_now.copy()
    p_now_ = p_next_now_.copy()

    u_list.append(u_next_now)
    v_list.append(v_next_now)
    vel_list.append(jnp.linalg.norm(U_next_now, axis=-1))
    p_list.append(p_next_now_)



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

pyvista_animation(DATAFOLDER, "u", duration=5, vmin=None, vmax=None)


# %%
