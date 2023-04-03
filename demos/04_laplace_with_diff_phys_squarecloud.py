# %%
import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt
from tqdm import tqdm

# from torch.utils.tensorboard import SummaryWriter

from updec import *

#%%

RBF = polyharmonic
MAX_DEGREE = 4


RUN_NAME = "LaplaceDiffPhys"
DATAFOLDER = "./data/" + RUN_NAME +"/"
make_dir(DATAFOLDER)
# writer = SummaryWriter("runs/"+RUN_NAME)
KEY = jax.random.PRNGKey(41)     ## Use same random points for all iterations

Nx = 30
Ny = Nx
LR = 1e-2
GAMMA = 1
EPOCHS = 5000



facet_types={"North":"d", "South":"d", "West":"d", "East":"d"}
train_cloud = SquareCloud(Nx=Nx, Ny=Ny, facet_types=facet_types, noise_key=None, support_size="max")

train_cloud.visualize_cloud(s=0.1, title="Training cloud", figsize=(5,4));

#%%

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
@Partial(jax.jit, static_argnums=[2,3])
def my_diff_operator(x, center=None, rbf=None, monomial=None, fields=None):
    return nodal_laplacian(x, center, rbf, monomial)

@Partial(jax.jit, static_argnums=[2])
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


for step in range(1, EPOCHS+1):

    ### Optimsation start ###
    loss, grad = grad_loss_fn(optimal_bcn)
    # print("calculated grad = ", grad)
    learning_rate = LR * (GAMMA**step)

    optimal_bcn = optimal_bcn - grad * learning_rate

    # writer.add_scalar('loss', float(loss), step)

    north_error = jnp.mean((optimal_bcn-exact_control)**2)
    history_cost.append(loss)
    north_mse.append(north_error)

    if step<=3 or step%100==0:
        print("Epoch: %-5d  LR: %.4f    Loss: %.8f  TestMSE: %.6f" % (step, learning_rate, loss, north_error))


### Visualisation at north
ax = plot(x_north, exact_control, "x-", label="Ideal/Analytical", x_label=r"$x$", figsize=(5,3), ylim=(-.2,.2))
plot(x_north, optimal_bcn, "o", label="Differentiable Physics", ax=ax, title=f"Optimised north solution / MSE = {north_error:.4f}");
# plt.savefig(DATAFOLDER+"bcn_"+str(step)+".png", transparent=True)


ax = plot(history_cost, label='Cost objective', x_label='epochs', title="Loss", y_scale="log");
plot(north_mse, label='Test MSE', x_label='epochs', title="Loss", y_scale="log", ax=ax);


# %%

############# Just for fun ########## TODO do this outside the loop

optimal_conditions = {"South":d_south, "West":d_west, "North":optimal_bcn, "East":d_east}
sol = pde_solver(diff_operator=my_diff_operator,
                rhs_operator = my_rhs_operator,
                cloud = train_cloud,
                boundary_conditions = optimal_conditions,
                rbf=RBF,
                max_degree=MAX_DEGREE)
# optimal_error = jnp.mean((exact_sol-sol.vals)**2)

# print("calculated sol = ", sol.vals)

### Optional visualisation of whole solution
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(6*2,5))
train_cloud.visualize_field(sol.vals, cmap="jet", projection="2d", title="Optimized solution", ax=ax1, vmin=-1, vmax=1)
# test_cloud.visualize_field(exact_sol, cmap="jet", projection="3d", title="Analytical solution", ax=ax2, vmin=-1, vmax=1)
train_cloud.visualize_field(jnp.abs(sol.vals-exact_sol), cmap="magma", projection="2d", title="Absolute error", ax=ax2, vmin=0, vmax=1);
plt.savefig(DATAFOLDER+"solution_"+str(step)+".png", transparent=True)

############# fun ends ##########



## Write to tensorboard
# hparams_dict = {"learning_rate":LR, "nb_epochs":EPOCHS, "rbf":RBF.__name__, "max_degree":MAX_DEGREE, "nb_nodes":cloud.N, "support_size":cloud.support_size}
# metrics_dict = {"metrics/mse_error_north":float(north_error)}
# writer.add_hparams(hparams_dict, metrics_dict, run_name="hp_params")
# writer.add_figure("plots", fig)
# writer.flush()
# writer.close()


# %%
