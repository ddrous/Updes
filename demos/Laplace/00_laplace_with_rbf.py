# %%

"""
Test of the Updec package on the Laplace equation with RBFs
"""

import os
# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = "false"
import time

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update('jax_platform_name', 'cpu')           ## CPU is faster here !
jax.config.update("jax_enable_x64", True)

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context='notebook', style='ticks',
        font='sans-serif', font_scale=1, color_codes=True, rc={"lines.linewidth": 2})
plt.style.use("dark_background")

from updec import *
# key = jax.random.PRNGKey(13)
key = None

# from torch.utils.tensorboard import SummaryWriter
# import datetime

RUN_NAME = "TempFolder"
DATAFOLDER = "../data/" + RUN_NAME +"/"
make_dir(DATAFOLDER)

RBF = partial(polyharmonic, a=1)
MAX_DEGREE = 1

# EPS=3.0
# RBF = partial(gaussian, eps=EPS)
# MAX_DEGREE = 0

# RBF = partial(thin_plate, a=3)
# MAX_DEGREE = 3

# Nx = 70     ## 19 minutes
Nx = 30
Ny = Nx
SUPPORT_SIZE = "max"
# SUPPORT_SIZE = 20

# print(run_name)
# r = jnp.linspace(0,1.2,1001)
# plt.plot(r, partial(gaussian_func,eps=EPS)(r), label="thin_plate") 
# plt.legend()

facet_types={"South":"n", "West":"d", "North":"d", "East":"d"}
# facet_types={"south":"n", "west":"n", "north":"n", "east":"n"}
# facet_types={"south":"d", "west":"d", "north":"d", "east":"d"}
cloud = SquareCloud(Nx=Nx, Ny=Ny, facet_types=facet_types, noise_key=key, support_size=SUPPORT_SIZE)
# cloud = GmshCloud(filename="./meshes/unit_square.py", facet_types=facet_types, mesh_save_location=DATAFOLDER, support_size="max")

cloud.visualize_cloud()

# print(distance(cloud.nodes[2], cloud.nodes[6]))
# print(cloud.global_indices)
# print("New local support of node 0:", cloud.nodes[0], cloud.local_supports[0])


## Diffeerential operator
# @Partial(jax.jit, static_argnums=[2,3])
def my_diff_operator(x, center=None, rbf=None, monomial=None, fields=None):
    # a, b = args[0], args[1]   ## agrs is a array
    return nodal_laplacian(x, center, rbf, monomial)

# Right-hand side operator
# @Partial(jax.jit, static_argnums=[2])
def my_rhs_operator(x, centers=None, rbf=None, fields=None):
    return -0.0

d_north = lambda node: jnp.sin(jnp.pi * node[0])
d_zero = lambda node: 0.0
boundary_conditions = {"South":d_zero, "West":d_zero, "North":d_north, "East":d_zero}


# betas = 100*jnp.ones((len(cloud.facet_nodes["South"]), ))
# boundary_conditions = {"South":(d_zero, betas), "West":d_zero, "North":d_north, "East":d_zero}

start = time.time()
# solution_field = pde_solver(my_diff_operator, my_rhs_operator, cloud, boundary_conditions, RBF, MAX_DEGREE)
sol = pde_solver_jit(diff_operator=my_diff_operator, 
                rhs_operator = my_rhs_operator, 
                cloud = cloud, 
                boundary_conditions = boundary_conditions, 
                rbf=RBF,
                max_degree=MAX_DEGREE)
walltime = time.time() - start

minutes = walltime // 60 % 60
seconds = walltime % 60
print(f"Walltime: {minutes} minutes {seconds:.2f} seconds")

## RBF solution
rbf_sol = sol.vals
lap = laplacian_vec(cloud.sorted_nodes, sol.coeffs, cloud.sorted_nodes, RBF)
# print("Norm of laplacian inside the domain:", jnp.linalg.norm(lap[:cloud.Ni]))
# print("Laplacian inside the domain:", lap[:cloud.Ni])            ## Should be equal to RHS of PDE

## Exact solution
def laplace_exact_sol(coord):
    return jnp.sin(jnp.pi*coord[0])*jnp.cosh(jnp.pi*coord[1]) / jnp.cosh(jnp.pi)
laplace_exact_sol = jax.vmap(laplace_exact_sol, in_axes=(0,), out_axes=0)
exact_sol = laplace_exact_sol(cloud.sorted_nodes)

## Dangerous TODO think of removing this
internal_nodes = jnp.array(range(cloud.Ni))
south_nodes = jnp.array(cloud.facet_nodes["South"])
# print("South nodes:", south_nodes)
# rbf_sol = rbf_sol.at[south_nodes].set(jnp.nan)
# exact_sol = exact_sol.at[south_nodes].set(jnp.nan)

error = jnp.nan_to_num(jnp.abs(exact_sol-rbf_sol))
print("MSE total error:", jnp.mean(error**2))

error_neumann = jnp.nan_to_num(jnp.abs(exact_sol[south_nodes]-rbf_sol[south_nodes]))
print("MSE on Neumann boundary:", jnp.mean(error_neumann**2))

# error_dirichlet = jnp.mean(error**2) - jnp.mean(error_neumann**2) - jnp.mean(jnp.nan_to_num(jnp.abs(exact_sol[internal_nodes]-rbf_sol[internal_nodes]))**2) ##TODO this formula is wrong, but informative
# print("MSE error on Dirichlet boundaries:", error_dirichlet)

## JNP SAVE solutions
# cloud_shape = str(Nx)+"x"+str(Ny)
# jnp.save("./examples/temp/sol_laplace_"+cloud_shape+".npy", solution_field)
# jnp.save("./examples/temp/mse_error_laplace_"+cloud_shape+".npy", error)



### Visualisation
fig = plt.figure(figsize=(6*3,5))
ax1= fig.add_subplot(1, 3, 1, projection='3d')
ax2 = fig.add_subplot(1, 3, 2, projection='3d')
ax3 = fig.add_subplot(1, 3, 3, projection='3d')
cloud.visualize_field(rbf_sol, cmap="jet", projection="3d", title="RBF solution", ax=ax1);
cloud.visualize_field(exact_sol, cmap="jet", projection="3d", title="Analytical solution", ax=ax2);
# cloud.visualize_field(exact_sol, cmap="jet", projection="2d", title="Analytical solution", ax=ax2);
cloud.visualize_field(error, cmap="magma", projection="3d", title="MSE error", ax=ax3);
plt.show()


## Write stuff to tensorboard
# run_name = str(datetime.datetime.now())[:19]        ##For tensorboard
# writer = SummaryWriter("runs/"+run_name, comment='-Laplace')
# hparams_dict = {"rbf":RBF.__name__, "max_degree":MAX_DEGREE, "nb_nodes":Nx*Ny, "support_size":SUPPORT_SIZE}      ## TODO Add local support
# metrics_dict = {"metrics/mse_error":float(error), "metrics/wall_time":walltime}                                        ## TODO Add time
# writer.add_hparams(hparams_dict, metrics_dict, run_name="hp_params")
# writer.add_figure("plots", fig)
# writer.flush()
# writer.close()

# %%
