# %%
import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt
from tqdm import tqdm

# from torch.utils.tensorboard import SummaryWriter

from updec import *


RBF = polyharmonic
MAX_DEGREE = 4


RUN_NAME = random_name()
DATAFOLDER = "./data/" + RUN_NAME +"/"
make_dir(DATAFOLDER)
# writer = SummaryWriter("runs/"+RUN_NAME)



facet_types={"West":"d", "East":"d", "North":"d", "South":"n"}  ## North is lowest priority (we enforce 0 on boundaries)
cloud = GmshCloud(filename="./meshes/unit_square.py", facet_types=facet_types, mesh_save_location=DATAFOLDER, support_size="max")
cloud.visualize_cloud(s=6, title=r"Unit square")



@Partial(jax.jit, static_argnums=[2,3])
def my_diff_operator(x, center=None, rbf=None, monomial=None, fields=None):
    return nodal_laplacian(x, center, rbf, monomial)

@Partial(jax.jit, static_argnums=[2])
def my_rhs_operator(x, centers=None, rbf=None, fields=None):
    return 0.0


## Exact solution
def laplace_exact_sol(coords):
    return jnp.sin(jnp.pi*coords[:, 0])*jnp.cosh(jnp.pi*coords[:, 1]) / jnp.cosh(jnp.pi)
exact_sol = laplace_exact_sol(cloud.sorted_nodes)



### Optimisation start ###
bc_zero = jax.jit(lambda x: 0.0)

@jax.jit
def loss_fn(bcn):
    sol = pde_solver(diff_operator=my_diff_operator,
                    rhs_operator = my_rhs_operator,
                    cloud = cloud, 
                    boundary_conditions = {"South":bc_zero, "West":bc_zero, "North":bcn, "East":bc_zero},
                    rbf=RBF,
                    max_degree=MAX_DEGREE)
    return jnp.mean((exact_sol-sol.vals)**2)

grad_loss_fn = jax.value_and_grad(loss_fn)


LR = 15.
EPOCHS = 12

north_ids = jnp.array(cloud.facet_nodes["North"])
optimal_bcn = jnp.zeros((north_ids.shape[0]))

for step in tqdm(range(EPOCHS)):


    north_coords = cloud.sorted_nodes[north_ids]
    ideal_bcn = jnp.sin(jnp.pi * north_coords[:, 0])
    north_error = jnp.mean((optimal_bcn-ideal_bcn)**2)



    ### Visualisation at north
    fig = plt.figure(figsize=(6,2))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(north_coords[:, 0], ideal_bcn, "x", label="Ideal/Analytical")
    ax.plot(north_coords[:, 0], optimal_bcn, "o", label="Differentiable Physics")
    ax.set_xlabel(r"$x$")
    ax.set_title(f"Optimised north solution / MSE = {north_error:.4f}")
    # ax.legend()
    plt.savefig(DATAFOLDER+"bcn_"+str(step)+".png", transparent=True)


    ############# Just for fun ########## TODO do this at the end
    optimal_conditions = {"South":bc_zero, "West":bc_zero, "North":optimal_bcn, "East":bc_zero}
    sol = pde_solver(diff_operator=my_diff_operator,
                    rhs_operator = my_rhs_operator,
                    cloud = cloud,
                    boundary_conditions = optimal_conditions,
                    rbf=RBF,
                    max_degree=MAX_DEGREE)
    # optimal_error = jnp.mean((exact_sol-sol.vals)**2)

    ### Optional visualisation of whole solution
    fig2 = plt.figure(figsize=(6*3,5))
    ax1= fig2.add_subplot(1, 3, 1, projection='3d')
    ax2 = fig2.add_subplot(1, 3, 2, projection='3d')
    ax3 = fig2.add_subplot(1, 3, 3, projection='3d')
    cloud.visualize_field(sol.vals, cmap="jet", projection="3d", title="Optimized solution", ax=ax1, vmin=0, vmax=1)
    cloud.visualize_field(exact_sol, cmap="jet", projection="3d", title="Analytical solution", ax=ax2, vmin=0, vmax=1)
    cloud.visualize_field(jnp.abs(sol.vals-exact_sol), cmap="magma", projection="3d", title="Absolute error", ax=ax3, vmin=0, vmax=1)
    ############# fun ends ##########
    plt.savefig(DATAFOLDER+"solution_"+str(step)+".png", transparent=True)




    ### Optimsation start ###
    loss, grad = grad_loss_fn(optimal_bcn)

    optimal_bcn = optimal_bcn - LR*grad

    # writer.add_scalar('loss', float(loss), step)

    print(f"\nStep: {step} \t Loss: {loss}")

    ### Optimsation end ###


plt.show()



## Write to tensorboard
# hparams_dict = {"learning_rate":LR, "nb_epochs":EPOCHS, "rbf":RBF.__name__, "max_degree":MAX_DEGREE, "nb_nodes":cloud.N, "support_size":cloud.support_size}
# metrics_dict = {"metrics/mse_error_north":float(north_error)}
# writer.add_hparams(hparams_dict, metrics_dict, run_name="hp_params")
# writer.add_figure("plots", fig)
# writer.flush()
# writer.close()

