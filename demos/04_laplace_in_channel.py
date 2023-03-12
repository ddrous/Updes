
# %%
import jax
import jax.numpy as jnp
from jax.tree_util import Partial
from tqdm import tqdm

from updec import *


### Constants for the problem

RBF = polyharmonic      ## Can define which rbf to use
MAX_DEGREE = 4

Pa = 101325.0
# Pa = 10.

EXPERIMENET_ID = random_name()
DATAFOLDER = "./data/" + EXPERIMENET_ID +"/"
make_dir(DATAFOLDER)



# %%


facet_types_phi = {"Wall":"n", "Inflow":"n", "Outflow":"d", "Cylinder":"n"}

cloud_phi = GmshCloud(filename="./meshes/channel_cylinder.py", facet_types=facet_types_phi, mesh_save_location=DATAFOLDER)    ## TODO Pass the savelocation here

cloud_phi.visualize_cloud(figsize=(8.5,1.4), s=6, title=r"Cloud for $\phi$");
cloud_phi.visualize_normals(title="Normals for phi", zoom_region=(-0.25,0.25,-0.25,0.25))



# %%
print(f"\nStarting RBF simulation with {cloud_phi.N} nodes\n")


@Partial(jax.jit, static_argnums=[2,3])
def diff_operator(x, center=None, rbf=None, monomial=None, fields=None):
    return -nodal_laplacian(x, center, rbf, monomial)

@Partial(jax.jit, static_argnums=[2])
def rhs_operator(x, centers=None, rbf=None, fields=None):
    return  10000.


# parabolic = jax.jit(lambda x: 1.5 - 6*(x[1]**2))
ones = jax.jit(lambda x: 1.)
atmospheric = jax.jit(lambda x: Pa)
zero = jax.jit(lambda x: 0.)

bc_phi = {"Wall":zero, "Inflow":zero, "Outflow":atmospheric, "Cylinder":zero}
# bc_phi = {"Wall":zero, "Inflow":zero, "Outflow":zero, "Cylinder":zero}

usol = pde_solver(diff_operator=diff_operator, 
                rhs_operator = rhs_operator, 
                cloud = cloud_phi, 
                boundary_conditions = bc_phi,
                rbf=RBF,
                max_degree=MAX_DEGREE)


# %%


print("\nSimulation complete. Now running visualisation ...")
cloud_phi.visualize_field(usol.vals, figsize=(10,2.5), cmap="jet")


# %%
