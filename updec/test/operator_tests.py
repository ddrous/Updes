from updec import *
"A unit test that checks if the gradient of a constant field is zero"


# facet_types = {"north":"n", "south":"n", "east":"n", "west":"n"}
facet_types = {"north":"d", "south":"d", "east":"d", "west":"d"}

size = 22
cloud = SquareCloud(Nx=size, Ny=size, facet_types=facet_types, support_size="max", noise_key=jax.random.PRNGKey(39))       ## TODO do not hardcode this path

cloud.visualize_cloud(figsize=(6,5), s=12, title="Cloud for testing");

RBF = polyharmonic      ## Can define which rbf to use
MAX_DEGREE = 2

const = lambda x: 1.0
bc = {"north":const, "south":const, "east":const, "west":const}



@Partial(jax.jit, static_argnums=[2,3])
def diff_operator(x, center=None, rbf=None, monomial=None, fields=None):
    return nodal_value(x, center, rbf, monomial)

@Partial(jax.jit, static_argnums=[2])
def rhs_operator(x, centers=None, rbf=None, fields=None):
    return 1.0

sol = pde_solver(diff_operator=diff_operator,
                rhs_operator = rhs_operator,
                cloud = cloud, 
                boundary_conditions = bc,
                rbf=RBF,
                max_degree=MAX_DEGREE)

# p_ = p_ + phisol_.vals
# grad = gradient(cloud.sorted_nodes[0], sol.coeffs, cloud.sorted_nodes, RBF)
grads = gradient_vec(cloud.sorted_nodes, sol.coeffs, cloud.sorted_nodes, RBF)        ## TODO use Pde_solver here instead ?
grads_norm = jnp.linalg.norm(grads, axis=-1)
print("Grads close to 0 ?", jnp.allclose(grads_norm, 0, rtol=1e-05))

field_vec = jnp.stack([sol.coeffs, sol.coeffs], axis=-1)
divs = divergence_vec(cloud.sorted_nodes, field_vec, cloud.sorted_nodes, RBF)        ## TODO use Pde_solver here instead ?
print("Divs close to 0 ?", jnp.allclose(divs, 0, rtol=1e-05))

cloud.visualize_field(sol.vals, figsize=(6,4.5), title="Constant field");
# cloud.visualize_field(grads[:,0], figsize=(6,4.5), title="Partial along x");
# cloud.visualize_field(grads[:,1], figsize=(6,4.5), title="Partial along y");
cloud.visualize_field(grads_norm, figsize=(6,4.5), title="Norm of gradient of const field");
cloud.visualize_field(divs, figsize=(6,4.5), title="Divergence of (const, const) vector field");

plt.show()
