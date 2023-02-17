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
key = jax.random.PRNGKey(42)

# from torch.utils.tensorboard import SummaryWriter
# import datetime


RBF = polyharmonic
MAX_DEGREE = 4
Nx = 10
Ny = Nx
SUPPORT_SIZE = Nx*Ny-1

# print(run_name)

facet_types={"south":"n", "west":"d", "north":"d", "east":"d"}
# facet_types={"south":"n", "west":"n", "north":"n", "east":"n"}
# facet_types={"south":"d", "west":"d", "north":"d", "east":"d"}
cloud = SquareCloud(Nx=Nx, Ny=Ny, facet_types=facet_types, noise_key=key, support_size=SUPPORT_SIZE)



## Diffeerential operator
# @Partial(jax.jit, static_argnums=[2,3])
def my_diff_operator(x, center=None, rbf=None, monomial=None, fields=None):
    # a, b = args[0], args[1]   ## agrs is a array
    return nodal_laplacian(x, center, rbf, monomial)

# Right-hand side operator
# @Partial(jax.jit, static_argnums=[2])
def my_rhs_operator(x, centers=None, rbf=None, fields=None):
    return -1.0

d_north = lambda node: jnp.sin(jnp.pi * node[0])
d_zero = lambda node: 0.0
boundary_conditions = {"south":d_zero, "west":d_zero, "north":d_north, "east":d_zero}

start = time.time()
# solution_field = pde_solver(my_diff_operator, my_rhs_operator, cloud, boundary_conditions, RBF, MAX_DEGREE)
sol = pde_solver(diff_operator=my_diff_operator, 
                rhs_operator = my_rhs_operator, 
                cloud = cloud, 
                boundary_conditions = boundary_conditions, 
                rbf=RBF,
                max_degree=MAX_DEGREE)
walltime = time.time() - start

minutes = walltime // 60 % 60
seconds = walltime % 60
print(f"Walltime: {minutes} minutes {seconds:.2f} seconds")


## Exact solution
def laplace_exact_sol(coord):
    return jnp.sin(jnp.pi*coord[0])*jnp.cosh(jnp.pi*coord[1]) / jnp.cosh(jnp.pi)
laplace_exact_sol = jax.vmap(laplace_exact_sol, in_axes=(0,), out_axes=0)

exact_sol = laplace_exact_sol(cloud.sorted_nodes)
error = jnp.mean((exact_sol-sol.vals)**2)



## JNP SAVE solutions
# cloud_shape = str(Nx)+"x"+str(Ny)
# jnp.save("./examples/temp/sol_laplace_"+cloud_shape+".npy", solution_field)
# jnp.save("./examples/temp/mse_error_laplace_"+cloud_shape+".npy", error)



### Visualisation
fig = plt.figure(figsize=(6*3,5))
ax1= fig.add_subplot(1, 3, 1, projection='3d')
ax2 = fig.add_subplot(1, 3, 2, projection='3d')
ax3 = fig.add_subplot(1, 3, 3, projection='3d')
cloud.visualize_field(sol.vals, cmap="jet", projection="3d", title="RBF solution", ax=ax1);
cloud.visualize_field(exact_sol, cmap="jet", projection="3d", title="Analytical solution", ax=ax2);
cloud.visualize_field(jnp.abs(sol.vals-exact_sol), cmap="magma", projection="3d", title="MSE error", ax=ax3);
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
