import os
# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = "false"

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




RBF = polyharmonic
MAX_DEGREE = 4
Nx = 4
Ny = 4


facet_types={"south":"n", "west":"d", "north":"d", "east":"d"}
cloud = SquareCloud(Nx=Nx, Ny=Ny, facet_types=facet_types, noise_key=key, support_size=Nx*Ny-1)



## Diffeerential operator
def my_diff_operator(x, node=None, monomial=None, *args):
    return  nodal_laplacian(x, node, monomial, rbf=RBF)

# Right-hand side operator
def my_rhs_operator(x):
    # return -100.0
    return 0.0

d_north = lambda node: jnp.sin(jnp.pi * node[0])
d_zero = lambda node: 0.0
boundary_conditions = {"south":d_zero, "west":d_zero, "north":d_north, "east":d_zero}


solution_field = pde_solver(my_diff_operator, my_rhs_operator, cloud, boundary_conditions, RBF, MAX_DEGREE)


## Exact solution
def laplace_exact_sol(coord):
    return jnp.sin(jnp.pi*coord[0])*jnp.cosh(jnp.pi*coord[1]) / jnp.cosh(jnp.pi)
coords = cloud.sort_nodes_jnp()
laplace_exact_sol = jax.vmap(laplace_exact_sol, in_axes=(0,), out_axes=0)

exact_sol = laplace_exact_sol(coords)
error = jnp.sum((exact_sol-solution_field)**2)



## JNP SAVE solutions
# cloud_shape = str(Nx)+"x"+str(Ny)
# jnp.save("./examples/temp/sol_laplace_"+cloud_shape+".npy", solution_field)
# jnp.save("./examples/temp/mse_error_laplace_"+cloud_shape+".npy", error)



### Visualisation
fig = plt.figure(figsize=(6*2,5))
ax1= fig.add_subplot(1, 2, 1, projection='3d')
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
cloud.visualize_field(solution_field, cmap="jet", projection="3d", title="RBF solution", figsize=(6,5), ax=ax1);
cloud.visualize_field(exact_sol, cmap="jet", projection="3d", title="Analytical solution", figsize=(6,5), ax=ax2);

plt.show()
