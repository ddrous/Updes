# %%

"""
Control of Laplace equation with differentiable physics
"""

import jax
import jax.numpy as jnp
import optax

import matplotlib.pyplot as plt
from tqdm import tqdm
import tracemalloc, time

from updec import *

#%%


RUN_NAME = "LaplaceDiffPhys"
DATAFOLDER = "./data/" + RUN_NAME +"/"
make_dir(DATAFOLDER)

COMPFOLDER = "./data/" + "Comparison" +"/"
make_dir(COMPFOLDER)

RBF = polyharmonic
MAX_DEGREE = 1

Nx = 100
Ny = Nx

LR = 1e-2
GAMMA = 1       ### LR decay rate
EPOCHS = 500


facet_types={"North":"d", "South":"d", "West":"d", "East":"d"}
train_cloud = SquareCloud(Nx=Nx, Ny=Ny, facet_types=facet_types, noise_key=None, support_size="max")

train_cloud.visualize_cloud(s=0.1, title="Training cloud", figsize=(5,4));

#%%

start = time.process_time()
tracemalloc.start()


## For the cost function
north_ids = jnp.array(train_cloud.facet_nodes["North"])
xy_north = train_cloud.sorted_nodes[north_ids, :]
x_north = xy_north[:, 0]
q_cost = jax.vmap(lambda x: jnp.cos(2*jnp.pi * x))(x_north)


## Exact solution
def laplace_exact_sol(xy):
    PI = jnp.pi
    x, y = xy

    a = 0.5 * jnp.sin(2*PI*x) * (jnp.exp(2*PI*(y-1)) + jnp.exp(2*PI*(1-y))) / jnp.cosh(2*PI)
    b = jnp.cos(2*PI*x) * (jnp.exp(2*PI*y) + jnp.exp(-2*PI*y)) / (4*PI*jnp.cosh(2*PI))

    return a+b

def laplace_exact_control(x):
    PI = jnp.pi
    return (jnp.sin(2*PI*x)/jnp.cosh(2*PI)) + (jnp.cos(2*PI*x)*jnp.tanh(2*PI)/(2*PI))


exact_sol = jax.vmap(laplace_exact_sol)(train_cloud.sorted_nodes)
exact_control = jax.vmap(laplace_exact_control)(x_north)


#%%
def my_diff_operator(x, center=None, rbf=None, monomial=None, fields=None):
    return nodal_laplacian(x, center, rbf, monomial)

def my_rhs_operator(x, centers=None, rbf=None, fields=None):
    return 0.0



### Optimisation start ###
d_south = jax.jit(lambda x: jnp.sin(2*jnp.pi * x[0]))
d_east = jax.jit(lambda x: jnp.sinh(2*jnp.pi*x[1]) / (2*jnp.pi * jnp.cosh(2*jnp.pi)))
d_west = d_east

@jax.jit
def loss_fn(bcn):
    sol = pde_solver(diff_operator=my_diff_operator,
                    rhs_operator = my_rhs_operator,
                    cloud = train_cloud, 
                    boundary_conditions = {"South":d_south, "West":d_west, "North":bcn, "East":d_east},
                    rbf=RBF,
                    max_degree=MAX_DEGREE)

    grad_n_y = gradient_vec(xy_north, sol.coeffs, train_cloud.sorted_nodes, RBF)[...,1]

    loss_cost = (grad_n_y - q_cost)**2
    return jnp.trapz(loss_cost, x=x_north)


grad_loss_fn = jax.value_and_grad(loss_fn)


# %% 

optimal_bcn = jnp.zeros((north_ids.shape[0]))
history_cost = []
north_mse = []

scheduler = optax.piecewise_constant_schedule(init_value=LR,
                                            boundaries_and_scales={int(EPOCHS*0.5):0.1, int(EPOCHS*0.75):0.1})
optimiser = optax.adam(learning_rate=scheduler)
opt_state = optimiser.init(optimal_bcn)

for step in tqdm(range(1, EPOCHS+1)):

    loss, grad = grad_loss_fn(optimal_bcn)

    # learning_rate = LR * (GAMMA**step)
    # optimal_bcn = optimal_bcn - grad * learning_rate

    updates, opt_state = optimiser.update(grad, opt_state, optimal_bcn)
    optimal_bcn = optax.apply_updates(optimal_bcn, updates)

    north_error = jnp.mean((optimal_bcn-exact_control)**2)
    history_cost.append(loss)
    north_mse.append(north_error)

    if step<=3 or step%10==0:
        print("Epoch: %-5d  InitLR: %.4f    Loss: %.8f  TestError: %.6f" % (step, LR, loss, north_error))

mem_usage = tracemalloc.get_traced_memory()[1]
exec_time = time.process_time() - start

print("A few script details:")
print(" Peak memory usage: ", mem_usage, 'bytes')
print(' CPU execution time:', exec_time, 'seconds')

tracemalloc.stop()


### Visualisation at north
ax = plot(x_north, exact_control, "-", label="Analytical", x_label=r"$x$", figsize=(5,3), ylim=(-.2,.2))
plot(x_north, optimal_bcn, "--", label="Diff. Physics", ax=ax, title=f"Optimised north solution / MSE = {north_error:.4f}");


ax = plot(history_cost, label='Cost objective', x_label='epochs', title="Loss", y_scale="log");
plot(north_mse, label='Test Error at North', x_label='epochs', title="Loss", y_scale="log", ax=ax);


# %%

############# Just for fun ########## TODO do this outside the loop

optimal_conditions = {"South":d_south, "West":d_west, "North":optimal_bcn, "East":d_east}
sol = pde_solver(diff_operator=my_diff_operator,
                rhs_operator = my_rhs_operator,
                cloud = train_cloud,
                boundary_conditions = optimal_conditions,
                rbf=RBF,
                max_degree=MAX_DEGREE)

### Optional visualisation of whole solution
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(6*2,5))
train_cloud.visualize_field(sol.vals, cmap="jet", projection="2d", title="Optimized solution", ax=ax1, vmin=-1, vmax=1)
# test_cloud.visualize_field(exact_sol, cmap="jet", projection="3d", title="Analytical solution", ax=ax2, vmin=-1, vmax=1)
train_cloud.visualize_field(jnp.abs(sol.vals-exact_sol), cmap="magma", projection="2d", title="Absolute error", ax=ax2, vmin=0, vmax=1);
plt.savefig(DATAFOLDER+"solution_"+str(step)+".png", transparent=True)



# %%

## Save data for comparison

jnp.savez(COMPFOLDER+"dp", objective_cost=history_cost, north_mse=north_error, exact_control=exact_control, optimal_bcn=optimal_bcn, exact_solution=exact_sol, optimal_solution=sol.vals, mem_time=jnp.array([mem_usage, exec_time]))


# %%
