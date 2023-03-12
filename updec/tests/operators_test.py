# %%
import random
from updec import *
"A unit test that checks if the gradient of a constant field is zero"

seed = random.randint(0,100)
print("Runnin experiments with seed: %s" % seed)
print()

# facet_types = {"North":"d", "South":"d", "East":"d", "West":"d"}

# size = 22
# cloud = SquareCloud(Nx=size, Ny=size, facet_types=facet_types, support_size="max", noise_key=jax.random.PRNGKey(seed))       ## TODO do not hardcode this path

# cloud.visualize_cloud(figsize=(6,5), s=12, title="Cloud for testing");


EXPERIMENET_ID = random_name()
DATAFOLDER = "../../demos/data/" + EXPERIMENET_ID +"/"
make_dir(DATAFOLDER)

facet_types = {"Wall":"d", "Inflow":"d", "Outflow":"d", "Cylinder":"n"}
cloud = GmshCloud(filename="../../demos/meshes/channel_cylinder.py", facet_types=facet_types, mesh_save_location=DATAFOLDER)    ## TODO Pass the savelocation here

cloud.visualize_cloud(figsize=(8.5,2.5), s=6, title=r"Cloud for $\phi$");
cloud.visualize_normals(title="Normals for phi", zoom_region=(-0.25,0.25,-0.25,0.25))



# %%

RBF = polyharmonic      ## Can define which rbf to use
MAX_DEGREE = 4

const = lambda x: seed
# bc = {"North":const, "South":const, "East":const, "West":const}
bc = {"Wall":const, "Inflow":const, "Outflow":const, "Cylinder":const}



@Partial(jax.jit, static_argnums=[2,3])
def diff_operator(x, center=None, rbf=None, monomial=None, fields=None):
    return nodal_value(x, center, rbf, monomial)

@Partial(jax.jit, static_argnums=[2])
def rhs_operator(x, centers=None, rbf=None, fields=None):
    return seed

sol = pde_solver(diff_operator=diff_operator,
                rhs_operator = rhs_operator,
                cloud = cloud, 
                boundary_conditions = bc,
                rbf=RBF,
                max_degree=MAX_DEGREE)

# %%
# p_ = p_ + phisol_.vals
# grad = gradient(cloud.sorted_nodes[0], sol.coeffs, cloud.sorted_nodes, RBF)
grads = gradient_vec(cloud.sorted_nodes, sol.coeffs, cloud.sorted_nodes, RBF)        ## TODO use Pde_solver here instead ?
grads_norm = jnp.linalg.norm(grads, axis=-1)
print("Grads close to 0 ?", jnp.allclose(grads_norm, 0, atol=1e-05))

field_vec = jnp.stack([sol.coeffs, sol.coeffs], axis=-1)
divs = divergence_vec(cloud.sorted_nodes, field_vec, cloud.sorted_nodes, RBF)        ## TODO use Pde_solver here instead ?
print("Divs close to 0 ?", jnp.allclose(divs, 0, atol=1e-05))

cloud.visualize_field(sol.vals, figsize=(8.5,2.5), title="Constant field");
# cloud.visualize_field(grads[:,0], figsize=(6,4.5), title="Partial along x");
# cloud.visualize_field(grads[:,1], figsize=(6,4.5), title="Partial along y");
cloud.visualize_field(grads_norm, figsize=(8.5,2.5), title="Norm of gradient of const field");
cloud.visualize_field(divs, figsize=(8.5,2.5), title="Divergence of (const, const) vector field");

# plt.show()

# %%
