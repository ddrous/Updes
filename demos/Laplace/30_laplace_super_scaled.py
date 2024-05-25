# %%

"""
Super-Scaled Updes on the Laplace equation with RBFs
"""

# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = "false"

import pstats
from updes import *

import time

import jax
import jax.numpy as jnp

# jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)

import matplotlib.pyplot as plt
import seaborn as sns

import cProfile


DATAFOLDER = "./data/TempFolder/"

RBF = partial(polyharmonic, a=1)
# RBF = partial(gaussian, eps=1e1)
# RBF = partial(thin_plate, a=3)
MAX_DEGREE = 1

Nx = Ny = 15
SUPPORT_SIZE = "max"
# SUPPORT_SIZE = 50*1
facet_types={"South":"d", "West":"d", "North":"d", "East":"d"}

start = time.time()

## benchmarking with cprofile
res = cProfile.run("cloud = SquareCloud(Nx=Nx, Ny=Ny, facet_types=facet_types, support_size=SUPPORT_SIZE)")

## Print results sorted by cumulative time
p = pstats.Stats(res)
p.sort_stats('cumulative').print_stats(10)


## Only print the top 10 high-level function
# p.print_callers(10)



# cloud = SquareCloud(Nx=Nx, Ny=Ny, facet_types=facet_types, support_size=SUPPORT_SIZE)



walltime = time.time() - start

print(f"Cloud generation walltime: {walltime:.2f} seconds")

# cloud.visualize_cloud(s=0.5, figsize=(7,6));

# %%

def my_diff_operator(x, center=None, rbf=None, monomial=None, fields=None):
    return nodal_laplacian(x, center, rbf, monomial)

def my_rhs_operator(x, centers=None, rbf=None, fields=None):
    return -0.0

d_north = lambda node: jnp.sin(jnp.pi * node[0])
d_zero = lambda node: 0.0
boundary_conditions = {"South":d_zero, "West":d_zero, "North":d_north, "East":d_zero}

start = time.time()
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



fig = plt.figure(figsize=(6*1,5))
ax= fig.add_subplot(1, 1, 1, projection='3d')
cloud.visualize_field(rbf_sol, cmap="jet", projection="3d", title="Laplace with RBFs", ax=ax);
plt.show()
## Savefig
fig.savefig(DATAFOLDER+"super_scaled.png", dpi=300)


# %%



# import jax
# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = "false"
# import jax.numpy as jnp
# import jax.random as jr
# import lineax as lx

# # size = 15000
# # matrix_key, vector_key = jr.split(jr.PRNGKey(0))
# # matrix = jr.normal(matrix_key, (size, size))
# # vector = jr.normal(vector_key, (size,))
# # operator = lx.MatrixLinearOperator(matrix)
# # solution = lx.linear_solve(operator, vector, solver=lx.QR())
# # solution.value

# # size = 8000
# # matrix_key, vector_key = jr.split(jr.PRNGKey(0))
# # matrix = jr.normal(matrix_key, (size, size))
# # vector = jr.normal(vector_key, (size,))
# # solution = jnp.linalg.solve(matrix, vector)
# # solution


# size = 15000
# matrix_key, vector_key = jr.split(jr.PRNGKey(0))
# matrix = jr.normal(matrix_key, (size, size))
# vector = jr.normal(vector_key, (size,))
# solution = jnp.linalg.lstsq(matrix, vector)

# %%


# ## Observing the sparsity patten of the matrices involved

# RBF = partial(gaussian, eps=1)

# Nx = Ny = 10
# SUPPORT_SIZE = "max"
# # SUPPORT_SIZE = 5
# facet_types={"South":"n", "West":"d", "North":"d", "East":"d"}

# cloud = SquareCloud(Nx=Nx, Ny=Ny, facet_types=facet_types, support_size=SUPPORT_SIZE)




# M = compute_nb_monomials(MAX_DEGREE, 2)
# A = assemble_A(cloud, RBF, M)
# mat1 = jnp.abs(A)

# inv_A = assemble_invert_A(cloud, RBF, M)
# mat2 = jnp.abs(inv_A)

# ## Matrix B for the linear system
# mat3 = sol.mat

# ## 3 figures
# fig, ax = plt.subplots(1, 3, figsize=(15,5))

# sns.heatmap(jnp.abs(mat1), ax=ax[0], cmap="grey", cbar=True, square=True, xticklabels=False, yticklabels=False)
# ax[0].set_title("Collocation Matrix")

# sns.heatmap(jnp.abs(mat2), ax=ax[1], cmap="grey", cbar=True, square=True, xticklabels=False, yticklabels=False)
# ax[1].set_title("Inverse of Collocation Matrix")

# sns.heatmap(jnp.abs(mat3), ax=ax[2], cmap="grey", cbar=True, square=True, xticklabels=False, yticklabels=False)
# ax[2].set_title("Linear System Matrix (B)")

# # plt.title("Sparsity Pattern of the Collocation Matrix")
# plt.show()


#%%

