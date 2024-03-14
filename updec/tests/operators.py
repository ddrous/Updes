# %%
import random
from functools import partial
from updec import *
"A unit test that checks if the gradient of a constant field is zero"

seed = random.randint(0,100)
seed = 12

# EXPERIMENET_ID = random_name()
EXPERIMENET_ID = "TempFolder"
DATAFOLDER = "demos/NavierStokes/data/" + EXPERIMENET_ID + "/"
# make_dir(DATAFOLDER)


# %%
# facet_types = {"North":"d", "South":"d", "East":"d", "West":"d"}

# size = 22
# cloud = SquareCloud(Nx=size, Ny=size, facet_types=facet_types, support_size="max", noise_key=jax.random.PRNGKey(seed))       ## TODO do not hardcode this path

# cloud.visualize_cloud(figsize=(6,5), s=12, title="Cloud for testing");


# facet_types = {"Wall":"n", "Inflow":"n", "Outflow":"n"}
# facet_types = {"Wall":"n", "Inflow":"n", "Outflow":"n", "Cylinder":"n"}
# cloud = GmshCloud(filename="../../demos/meshes/channel_cylinder.py", facet_types=facet_types, mesh_save_location=DATAFOLDER)    ## TODO Pass the savelocation here

facet_types = {"Wall":"n", "Inflow":"n", "Outflow":"n", "Blowing":"n", "Suction":"n"}

cloud = GmshCloud(filename=DATAFOLDER+"mesh.msh", facet_types=facet_types, mesh_save_location=DATAFOLDER)

# cloud.visualize_cloud(figsize=(8.5,2.5), s=6, title=r"Cloud for $\phi$");
# fig, ax = plt.subplots(1, 2, figsize=(5.5*2,5))
# cloud.visualize_normals(ax=ax[0], title="Normals for phi", zoom_region=(-0.25,0.25,-0.25,0.25))
# cloud.visualize_normals(ax=ax[1], title="Normals for phi", zoom_region=(7.75,8.25,-0.25,0.25))



# %%

EPS = 10.0
RBF = partial(gaussian, eps=EPS)      ## Can define which rbf to use
# RBF = partial(polyharmonic, a=1)
# RBF = partial(thin_plate, a=3)
MAX_DEGREE = 1

# r = jnp.linspace(-10,10,1000)
# plt.plot(r, gaussian_func(r, eps=EPS))

const = lambda x: seed
zero = lambda x: 0.
# bc = {"North":const, "South":const, "East":const, "West":const}
# bc = {"Wall":const, "Inflow":const, "Outflow":const}
# bc = {"Wall":const, "Inflow":const, "Outflow":const, "Cylinder":const}
bc = {"Wall":const, "Inflow":const, "Outflow":const, "Blowing":const, "Suction":const}

for k in bc.keys():
    if facet_types[k] == "n":
        bc[k] = zero



@Partial(jax.jit, static_argnums=[2,3])
def diff_operator(x, center=None, rbf=None, monomial=None, fields=None):
    return nodal_value(x, center, rbf, monomial)
    # return nodal_laplacian(x, center, rbf, monomial)

@Partial(jax.jit, static_argnums=[2])
def rhs_operator(x, centers=None, rbf=None, fields=None):
    return seed
    # return 0

sol = pde_solver(diff_operator=diff_operator,
                rhs_operator = rhs_operator,
                cloud = cloud, 
                boundary_conditions = bc,
                rbf=RBF,
                max_degree=MAX_DEGREE)

# %%

print("Runnin experiments with seed: %s" % seed)
print()

# p_ = p_ + phisol_.vals
# grad = gradient(cloud.sorted_nodes[0], sol.coeffs, cloud.sorted_nodes, RBF)

# grads = gradient_vec(cloud.sorted_nodes, sol.coeffs, cloud.sorted_nodes, RBF)        ## TODO use Pde_solver here instead ?

grads = cartesian_gradient_vec(range(cloud.N), sol.vals, cloud)        ## TODO use Pde_solver here instead ?

grads_norm = jnp.linalg.norm(grads, axis=-1)
# print("Grads close to 0 ?", jnp.allclose(grads_norm, 0, atol=1e-05))
# print("Maximum of grad norm:", jnp.max(grads_norm))

# assert jnp.allclose(grads_norm, 0, atol=1e-05)

field_vec = jnp.stack([sol.coeffs, sol.coeffs], axis=-1)
divs = divergence_vec(cloud.sorted_nodes, field_vec, cloud.sorted_nodes, RBF)        ## TODO use Pde_solver here instead ?

# print("Divs close to 0 ?", jnp.allclose(divs, 0, atol=1e-05))
# print("Maximum of div norm:", jnp.max(divs))

def test():
    assert jnp.allclose(grads_norm, 0, atol=1e-05) == True
    assert jnp.allclose(divs, 0, atol=1e-05) == True

# cloud.visualize_field(sol.vals, projection="2d", figsize=(8.5,2.5), title="Constant field");
# # cloud.visualize_field(grads[:,0], figsize=(6,4.5), title="Partial along x");
# # cloud.visualize_field(grads[:,1], figsize=(6,4.5), title="Partial along y");
# cloud.visualize_field(grads_norm, figsize=(8.5,2.5), title="Norm of gradient of const field");
# cloud.visualize_field(divs, figsize=(8.5,2.5), title="Divergence of (const, const) vector field");

# plt.show()

# %%
