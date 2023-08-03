# %%

"""
Test of the Updec package on the Advection-Diffusion equation with RBFs: 
PDE here: https://en.wikipedia.org/wiki/Convection%E2%80%93diffusion_equation
"""

import time

import jax
import jax.numpy as jnp

jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)

from updec import *
# key = jax.random.PRNGKey(13)
key = None

# from torch.utils.tensorboard import SummaryWriter


RUN_NAME = "TempFolder"
DATAFOLDER = "./data/" + RUN_NAME +"/"
make_dir(DATAFOLDER)

RBF = partial(polyharmonic, a=1)
MAX_DEGREE = 1

DT = 1e-4
NB_TIMESTEPS = 10
PLOT_EVERY = 10

VEL = jnp.array([100.0, 0.0])
## Diffusive constant
K = 0.08

Nx = 30
Ny = Nx
SUPPORT_SIZE = "max"

facet_types={"South":"d", "West":"d", "North":"d", "East":"n"}
cloud = SquareCloud(Nx=Nx, Ny=Ny, facet_types=facet_types, noise_key=key, support_size=SUPPORT_SIZE)

cloud.visualize_cloud(s=0.1, figsize=(6,5))

# print("Local supports:", cloud.local_supports[0])


# %%

def my_diff_operator(x, center=None, rbf=None, monomial=None, fields=None):
    val = nodal_value(x, center, rbf, monomial)
    grad = nodal_gradient(x, center, rbf, monomial)
    lap = nodal_laplacian(x, center, rbf, monomial)
    return (val/DT) + jnp.dot(VEL, grad) - K*lap

def my_rhs_operator(x, centers=None, rbf=None, fields=None):
    return value(x, fields[:,0], centers, RBF) / DT     ## TODO value ?

d_zero = lambda x: 0.
boundary_conditions = {"South":d_zero, "West":d_zero, "North":d_zero, "East":d_zero}


## u0 is zero everywhere except at a point in the middle
u0 = jnp.zeros(cloud.N)
source_id = int(cloud.N*0.05)
source_neighbors = jnp.array(cloud.local_supports[source_id][:cloud.N//50])
u0 = u0.at[source_neighbors].set(0.95)

## Begin timestepping for 100 steps

# fig = plt.figure(figsize=(6,3))
# ax1= fig.add_subplot(1, 1, 1, projection='3d')
# ax = fig.add_subplot(1, 1, 1)


u = u0.copy()
ulist = [u]

start = time.time()

for i in range(1, NB_TIMESTEPS+1):
    ufield = pde_solver_jit(diff_operator=my_diff_operator,
                        rhs_operator = my_rhs_operator,
                        rhs_args=[u],
                        cloud = cloud,
                        boundary_conditions = boundary_conditions, 
                        rbf=RBF,
                        max_degree=MAX_DEGREE,)
    
    u = ufield.vals
    ulist.append(u)

    if i<=3 or i%PLOT_EVERY==0:
        print(f"Step {i}")
        # plt.cla()
        # cloud.visualize_field(u, cmap="jet", projection="3d", title=f"Step {i}")
        ax, _ = cloud.visualize_field(u, cmap="jet", title=f"Step {i}", vmin=0, vmax=1, figsize=(6,3),colorbar=False)
        # plt.draw()
        plt.show()


walltime = time.time() - start

minutes = walltime // 60 % 60
seconds = walltime % 60
print(f"Walltime: {minutes} minutes {seconds:.2f} seconds")



# %%

ax = plt.gca()
filename = DATAFOLDER + "advection_diffusion_rbf.png"
cloud.animate_fields([ulist], cmaps="jet", filename=filename, figsize=(6,3),colorbar=False, title=["Implicit Advection-Diffusion with RBFs"])



# %%


## Write stuff to tensorboard
# run_name = str(datetime.datetime.now())[:19]        ##For tensorboard
# writer = SummaryWriter("runs/"+run_name, comment='-Laplace')
# hparams_dict = {"rbf":RBF.__name__, "max_degree":MAX_DEGREE, "nb_nodes":Nx*Ny, "support_size":SUPPORT_SIZE}
# metrics_dict = {"metrics/mse_error":float(error), "metrics/wall_time":walltime}
# writer.add_hparams(hparams_dict, metrics_dict, run_name="hp_params")
# writer.add_figure("plots", fig)
# writer.flush()
# writer.close()

# %%
