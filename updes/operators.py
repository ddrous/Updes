from logging import warning
import jax
import jax.numpy as jnp
from jax.tree_util import Partial
import lineax as lx

from functools import cache

import updes.config as UPDES
from updes.utils import compute_nb_monomials, SteadySol,  make_all_monomials
from updes.cloud import Cloud, SquareCloud
from updes.assembly import assemble_B, assemble_q, core_compute_coefficients


@Partial(jax.jit, static_argnums=[2,3])
def nodal_value(x, center=None, rbf=None, monomial=None):
    """ Computes the rbf or polynomial value at position x

    Args:
        x (Float[Array, "dim"]): The position at which to evaluate the rbf or polynomial
        center (Float[Array, "dim"]): The centroid of the RBF if used. (Currently mandadatory, despite the signature.)
        rbf (callable): The rbf to use. (Currently mandadatory, despite the signature.)
        monomial (callable): The monomial to use. (Currently mandadatory, despite the signature.)

    Returns:
        float: The value of the rbf or polynomial at x
    """

    if center != None:          ## TODO else should be handled with grace
        return rbf(x, center)
    elif monomial != None:
        return monomial(x)

@cache
def core_gradient_rbf(rbf):
    return jax.grad(rbf)

@cache
def core_gradient_mon(monomial):
    return jax.grad(monomial)

def nodal_gradient(x, center=None, rbf=None, monomial=None):
    """ Computes the gradient of the rbf or polynomial at position x

    Args:
        x (Float[Array, "dim"]): The position at which to evaluate the gradient rbf or polynomial
        center (Float[Array, "dim"]): The centroid of the RBF if used. (Currently mandadatory, despite the signature.)
        rbf (callable): The rbf to use. (Currently mandadatory, despite the signature.)
        monomial (callable): The monomial to use. (Currently mandadatory, despite the signature.)

    Returns:
        Float[Array, "dim"]: The value of the gradient rbf or polynomial at x
    """
    ## TODO? Given the caching, does calling this too often cause problems ?

    if center != None:
        ## TODO: benchmark the line below to see the acertain the cost of avoiding NaNs
        return jnp.nan_to_num(core_gradient_rbf(rbf)(x, center), posinf=0., neginf=0.)
    elif monomial != None:
        return core_gradient_mon(monomial)(x)

@cache
def core_laplacian_rbf(rbf):
    return jax.jacfwd(jax.grad(rbf))    ## TODO: Try reverse mode AD

@cache
def core_laplacian_mon(monomial):
    return jax.jacfwd(jax.grad(monomial))

def nodal_laplacian(x, center=None, rbf=None, monomial=None):     ## TODO Jitt through this efficiently
    """ Computes the laplacian as the trace of the jacobian of the gradient of the rbf or polynomial at position x

    Args:
        x (Float[Array, "dim"]): The position at which to evaluate the laplacian rbf or polynomial
        center (Float[Array, "dim"]): The centroid of the RBF if used. (Currently mandadatory, despite the signature.)
        rbf (callable): The rbf to use. (Currently mandadatory, despite the signature.)
        monomial (callable): The monomial to use. (Currently mandadatory, despite the signature.)

    Returns:
        float: The value of the laplacian rbf or polynomial at x
    """
    if center != None:
        return jnp.nan_to_num(jnp.trace(core_laplacian_rbf(rbf)(x, center)), posinf=0., neginf=0.)
    elif monomial != None:
        return jnp.trace(core_laplacian_mon(monomial)(x))


def nodal_div_grad(x, center=None, rbf=None, monomial=None, args=None):
    """ Computes the laplacian as the divergence of the gradient of the rbf or polynomial at position x

    Args:
        x (Float[Array, "dim"]): The position at which to evaluate the laplacian rbf or polynomial
        center (Float[Array, "dim"]): The centroid of the RBF if used. (Currently mandadatory, despite the signature.)
        rbf (callable): The rbf to use. (Currently mandadatory, despite the signature.)
        monomial (callable): The monomial to use. (Currently mandadatory, despite the signature.)
        args (list): The values to pre-multiply the gradients before taking the divergence

    Returns:
        float: The value of the laplacian rbf or polynomial at x
    """

    ## Trick to duplicate the args into a matricx
    if isinstance(args, jnp.ndarray):       ## a 2D array
        matrix = jnp.stack((args, args), axis=-1)   
    else:                                   ## a tuple
        matrix = jnp.array((args, args))

    if center != None:
        return jnp.nan_to_num(jnp.trace(matrix * core_laplacian_rbf(rbf)(x, center)), posinf=0., neginf=0.)
    elif monomial != None:
        return jnp.trace(matrix * core_laplacian_mon(monomial)(x))


## Vectorized versions of the nodal_value to use for the rbf case
_nodal_value_rbf_vec = jax.vmap(nodal_value, in_axes=(None, 0, None, None), out_axes=0)


def value(x, field, centers, rbf=None, clip_val=None):
    """ Computes the value of the field (given by its coefficients) at position x

    Args:
        x (Float[Array, "dim"]): The position at which to conmpute the field value
        centers (Float[Array, "nb_centers, dim"]): The centroids of the RBF to use
        rbf (callable): The rbf to use. (Currently mandadatory, despite the signature.)
        clip_val (float, optional): The limit to which to clip the value to avoid blowup. Defaults to None.

    Returns:
        float: The value of the field at x
    """

    N = centers.shape[0]
    lambdas, gammas = field[:N], field[N:]

    val_rbf = _nodal_value_rbf_vec(x, centers, rbf, None)
    final_val = jnp.sum(jnp.nan_to_num(lambdas*val_rbf, posinf=0., neginf=0.), axis=0)

    all_monomials = make_all_monomials(gammas.shape[0])

    ## Now, compute the value of the field
    for j in range(gammas.shape[0]):
        polynomial_val = nodal_value(x, monomial=all_monomials[j])
        final_val = final_val + jnp.nan_to_num(gammas[j] * polynomial_val, posinf=0., neginf=0.)

    if clip_val:
        return jnp.clip(final_val, -clip_val, clip_val)
    else:
        return final_val

value_vec_ = jax.vmap(value, in_axes=(0, None, None, None), out_axes=0)
value_vec = jax.jit(value_vec_, static_argnums=[3])


_nodal_gradient_rbf_vec = jax.vmap(nodal_gradient, in_axes=(None, 0, None, None), out_axes=0)


def gradient(x, field, centers, rbf=None, clip_val=None):
    """ Computes the gradient of the field (given by its coefficients) at position x

    Args:
        x (Float[Array, "dim"]): The position at which to conmpute the gradient value
        centers (Float[Array, "nb_centers, dim"]): The centroids of the RBF to use
        rbf (callable): The rbf to use. (Currently mandadatory, despite the signature.)
        clip_val (float, optional): The limit to which to clip the value to avoid blowup. Defaults to None.

    Returns:
        Float[Array, "dim"]: The gradient of the field at x
    """
    N = centers.shape[0]
    lambdas, gammas = field[:N], field[N:]

    grads_rbf = jnp.nan_to_num(_nodal_gradient_rbf_vec(x, centers, rbf, None), posinf=0., neginf=0.)
    lambdas = jnp.stack([lambdas, lambdas], axis=-1)
    final_grad = jnp.sum(jnp.nan_to_num(lambdas*grads_rbf, posinf=0., neginf=0.), axis=0)

    all_monomials = make_all_monomials(gammas.shape[0])

    for j in range(gammas.shape[0]):
        polynomial_grad = nodal_gradient(x, monomial=all_monomials[j])
        final_grad = final_grad.at[:].add(jnp.nan_to_num(gammas[j] * polynomial_grad, posinf=0., neginf=0.))

    if clip_val:
        return jnp.clip(final_grad, -clip_val, clip_val)
    else:
        return final_grad

gradient_vec_ = jax.vmap(gradient, in_axes=(0, None, None, None), out_axes=0)
gradient_vec = jax.jit(gradient_vec_, static_argnums=[3])


def gradient_vals(x, field, cloud, rbf, max_degree):
    """ Computes the gradient of the field (given by its values) at position x

    Args:
        x (Float[Array, "dim"]): The position at which to conmpute the gradient value
        centers (Float[Array, "nb_centers, dim"]): The centroids of the RBF to use
        rbf (callable): The rbf to use. (Currently mandadatory, despite the signature.)
        clip_val (float, optional): The limit to which to clip the value to avoid blowup. Defaults to None.

    Returns:
        Float[Array, "dim"]: The gradient of the field at x
    """
    nb_monomials = compute_nb_monomials(max_degree, cloud.dim)
    coeffs = core_compute_coefficients(field, cloud, rbf, nb_monomials)

    return gradient(x, coeffs, cloud.sorted_nodes, rbf)

gradient_vals_vec_ = jax.vmap(gradient_vals, in_axes=(0, None, None, None, None), out_axes=0)
gradient_vals_vec = jax.jit(gradient_vals_vec_, static_argnums=[2,3,4])


def cartesian_gradient(node_id, field, cloud:Cloud, clip_val=None):
    """ Computes the gradient of the field (given by its values) at a specific node of a cartesian grid, using finite differences

    Args:
        node_id (int): The node at which to conmpute the gradient value
        field (Float[Array, "nb_grid_points"]): The field to use for the gradient computation
        cloud (Cloud): The cloud of points to use: must be a square grid
        clip_val (float, optional): The limit to which to clip the value to avoid blowup. Defaults to None.

    Returns:
        Float[Array, "dim"]: The gradient of the field at x
    """
    N = field.shape[0]
    i = node_id

    final_grad = jnp.array([0.,0.])
    for d, direction in [(0, [1., 0]), (1, [0., 1.])]:
        direction = jnp.array(direction)

        opposite_neighbours = []
        closest_neighbour = None
        closest_distance = 1e20

        for j in cloud.local_supports[i]:
            vec = cloud.nodes[j] - cloud.nodes[i]
            vec_norm = jnp.linalg.norm(vec)
            vec_scaled = vec / vec_norm
            if jnp.dot(direction, vec_scaled) + 1. <= 1e-1:    ##TODO Keep reducing the tolerance until we find one
                opposite_neighbours.append(j)
                if vec_norm <= closest_distance:
                    closest_neighbour = j
                    closest_distance = vec_norm
        
        if closest_neighbour == None:
            return final_grad

        final_grad = final_grad.at[d].set((field[i] - field[closest_neighbour]) / vec_norm)

    if clip_val:
        return jnp.clip(final_grad, -clip_val, clip_val)
    else:
        return final_grad

def cartesian_gradient_vec(node_ids, field, cloud:Cloud):
    """ Vectorised version of the cartesian_gradient to all nodes in the square grid """
    grad = jnp.zeros((len(node_ids), 2))  ## TODO handle zero case
    for node_id in node_ids:
        grad = grad.at[node_id, :].set(cartesian_gradient(node_id, field, cloud))
    return grad


def enforce_cartesian_gradient_neumann(field, grads, boundary_conditions, cloud, clip_val=None):
    """ Sets the gradient at every neumann node using catesian grid """

    for facet_id, facet_type in cloud.facet_types.items():
        if facet_type == "n":
            nm_nodes = cloud.facet_nodes[facet_id]

            for i in nm_nodes:
                normal = cloud.outward_normals[i]

                opposite_neighbours = []
                closest_neighbour = None
                closest_distance = 1e20

                for j in cloud.local_supports[i]:
                    vec = cloud.nodes[j] - cloud.nodes[i]
                    vec_norm = jnp.linalg.norm(vec)
                    vec_scaled = vec / vec_norm
                    if jnp.dot(normal, vec_scaled) + 1. <= 1e-1:
                        opposite_neighbours.append(j)
                        if vec_norm <= closest_distance:
                            closest_neighbour = j
                            closest_distance = vec_norm

                grads = grads.at[i].set((field[i] - field[closest_neighbour]) / closest_distance)

    if clip_val:
        return jnp.clip(grads, -clip_val, clip_val)
    else:
        return grads


def divergence(x, field, centers, rbf=None, clip_val=None):
    """ Computes the divergence of the vector field (given by its coefficients) at position x

    Args:
        x (Float[Array, "dim"]): The position at which to compute the divergence value
        centers (Float[Array, "nb_centers, dim"]): The centroids of the RBF to use
        rbf (callable): The rbf to use. (Currently mandadatory, despite the signature.)
        clip_val (float, optional): The limit to which to clip the value to avoid blowup. Defaults to None.

    Returns:
        float: The divergence of field at x
    """
    dfieldx_dx = gradient(x, field[...,0], centers, rbf)[0]
    dfieldy_dy = gradient(x, field[...,1], centers, rbf)[1]

    if clip_val:
        return jnp.clip(dfieldx_dx + dfieldy_dy, -clip_val, clip_val)
    else:
        return dfieldx_dx + dfieldy_dy

divergence_vec = jax.vmap(divergence, in_axes=(0, None, None, None), out_axes=0)


_nodal_laplacian_rbf_vec = jax.vmap(nodal_laplacian, in_axes=(None, 0, None, None), out_axes=0)

def laplacian(x, field, centers, rbf=None, clip_val=None):
    """ Computes the laplacian of the field (given by its coefficients) at position x

    Args:
        x (Float[Array, "dim"]): The position at which to compute the laplacian value
        centers (Float[Array, "nb_centers, dim"]): The centroids of the RBF to use
        rbf (callable): The rbf to use. (Currently mandadatory, despite the signature.)
        clip_val (float, optional): The limit to which to clip the value to avoid blowup. Defaults to None.

    Returns:
        float: The laplacian of field at x
    """
    N = centers.shape[0]
    lambdas, gammas = field[:N], field[N:]

    laps_rbf = _nodal_laplacian_rbf_vec(x, centers, rbf, None)
    rbf_lap = jnp.sum(jnp.nan_to_num(lambdas*laps_rbf, posinf=0., neginf=0.), axis=0)


    all_monomials = make_all_monomials(gammas.shape[0])

    mon_lap = jnp.array([0.])
    for j in range(gammas.shape[0]):    ## TODO use FOR_I LOOP
        poly_lap = nodal_laplacian(x, monomial=all_monomials[j])
        mon_lap = mon_lap.at[:].add(gammas[j] * poly_lap)

    if clip_val:
        return jnp.clip(rbf_lap + mon_lap[0], -clip_val, clip_val)
    else:
        return rbf_lap + mon_lap[0]


laplacian_vec = jax.vmap(laplacian, in_axes=(0, None, None, None), out_axes=0)


def laplacian_vals(x, field, cloud, rbf, max_degree):
    """ Computes the laplacian of the field (given by its values) at position x

    Args:
        x (Float[Array, "dim"]): The position at which to compute the laplacian value
        centers (Float[Array, "nb_centers, dim"]): The centroids of the RBF to use
        rbf (callable): The rbf to use. (Currently mandadatory, despite the signature.)

    Returns:
        float: The laplacian of field at x
    """
    nb_monomials = compute_nb_monomials(max_degree, cloud.dim)
    coeffs = core_compute_coefficients(field, cloud, rbf, nb_monomials)

    return laplacian(x, coeffs, cloud.sorted_nodes, rbf)

laplacian_vals_vec_ = jax.vmap(laplacian_vals, in_axes=(0, None, None, None, None), out_axes=0)
laplacian_vals_vec = jax.jit(laplacian_vals_vec_, static_argnums=[2,3,4])









def integrate_field(field, cloud, rbf, max_degree):
    """Integrate the field (given by its coefficients) over the 2D square cloud domain, using the midpoint rule:
    1. Identify the small squares in the domain (all identical)
    2. Get the field value at the center of each square
    3. Account for border and corner values: https://stackoverflow.com/a/62991037/8140182
    4. Compute the approximate integral

    Returns:
        float: the integral of the field over the domain
    """

    ## Assert we have a SquareCloud instance
    assert isinstance(cloud, SquareCloud), "The cloud must be a SquareCloud instance"

    ## Compute the number of squares in the domain
    nb_squares = (cloud.Nx - 1) * (cloud.Ny - 1)

    ## Compute the size of a square
    area = (1 / (cloud.Nx-1)) * (1 / (cloud.Ny-1))

    ## Get quad_points inside the domain
    quad_points_x = jnp.linspace(1/(1*(cloud.Nx-1)), 1-1/(1*(cloud.Nx-1)), cloud.Nx-1)
    quad_points_y = jnp.linspace(1/(1*(cloud.Ny-1)), 1-1/(1*(cloud.Ny-1)), cloud.Ny-1)
    quad_points = jnp.array(jnp.meshgrid(quad_points_x, quad_points_y)).reshape(nb_squares, 2)
    field_vals_in = value_vec(quad_points, field, cloud.sorted_nodes, rbf)

    ## Get quad_points exclusively on the left boundary
    quad_points_x = jnp.zeros((cloud.Ny-2))
    quad_points_y = jnp.linspace(1/(1*(cloud.Ny-1)), 1-1/(1*(cloud.Ny-1)), cloud.Ny-2)
    quad_points = jnp.stack((quad_points_x, quad_points_y), axis=-1)
    field_vals_left = value_vec(quad_points, field, cloud.sorted_nodes, rbf)

    ## Get quad_points exclusively on the right boundary
    quad_points_x = jnp.ones((cloud.Ny-2))
    quad_points_y = jnp.linspace(1/(1*(cloud.Ny-1)), 1-1/(1*(cloud.Ny-1)), cloud.Ny-2)
    quad_points = jnp.stack((quad_points_x, quad_points_y), axis=-1)
    field_vals_right = value_vec(quad_points, field, cloud.sorted_nodes, rbf)

    ## Get quad_points exclusively on the bottom boundary
    quad_points_x = jnp.linspace(1/(1*(cloud.Nx-1)), 1-1/(1*(cloud.Nx-1)), cloud.Nx-2)
    quad_points_y = jnp.zeros((cloud.Nx-2))
    quad_points = jnp.stack((quad_points_x, quad_points_y), axis=-1)
    field_vals_bottom = value_vec(quad_points, field, cloud.sorted_nodes, rbf)

    ## Get quad_points exclusively on the top boundary
    quad_points_x = jnp.linspace(1/(1*(cloud.Nx-1)), 1-1/(1*(cloud.Nx-1)), cloud.Nx-2)
    quad_points_y = jnp.ones((cloud.Nx-2))
    quad_points = jnp.stack((quad_points_x, quad_points_y), axis=-1)
    field_vals_top = value_vec(quad_points, field, cloud.sorted_nodes, rbf)

    ## Get the bottom left corner
    quad_points = jnp.array([0., 0.])
    field_vals_bl = value(quad_points, field, cloud.sorted_nodes, rbf)

    ## Get the bottom right corner
    quad_points = jnp.array([1., 0.])
    field_vals_br = value(quad_points, field, cloud.sorted_nodes, rbf)

    ## Get the top left corner
    quad_point = jnp.array([0., 1.])
    field_vals_tl = value(quad_points, field, cloud.sorted_nodes, rbf)

    ## Get the top right corner
    quad_point = jnp.array([1., 1.])
    field_vals_tr = value(quad_point, field, cloud.sorted_nodes, rbf)

    ## Average the field 
    field_avg = jnp.sum(field_vals_in) + 0.5*(jnp.sum(field_vals_left) + jnp.sum(field_vals_right) + jnp.sum(field_vals_bottom) + jnp.sum(field_vals_top)) + 0.25*(field_vals_bl + field_vals_br + field_vals_tl + field_vals_tr)

    return field_avg * area






def interpolate_field(field, cloud1, cloud2):
    """Interpolates field from cloud1 to cloud2 given that their nodes might be numbered differently

    Args:
        field (Float[Array, "dim"]): The field to interpolate
        cloud1 (Cloud): The cloud from which to interpolate
        cloud2 (Cloud): The cloud to which to interpolate. Must be same type as cloud1, but with different numbering of node, i.e. different boundary conditions

    Raises:
        AssertionError: The two clouds do not contain the same number of nodes

    Returns:
        Float[Array, "dim"]: The interpolated field
    """

    assert cloud1.N == cloud2.N, "the two clouds do not contain the same number of nodes"   ## TODO: Make sure only the renumbering differs

    sorted_map1 = sorted(cloud1.renumbering_map.items(), key=lambda x:x[0])
    indexer1 = jnp.array(list(dict(sorted_map1).values()))
    field_orig = field[indexer1]

    indexer2 = jnp.array(list(cloud2.renumbering_map.keys()))

    return field_orig[indexer2]


def apply_neumann_conditions(field, boundary_conditions, cloud:Cloud):
    """Enforces the Neumann boundary conditions to the field """

    for facet_id, facet_type in cloud.facet_types.items():
        if facet_type == "n":
            nm_nodes = cloud.facet_nodes[facet_id]

            for i in nm_nodes:
                normal = cloud.outward_normals[i]

                opposite_neighbours = []
                closest_neighbour = None
                closest_distance = 1e20

                for j in cloud.local_supports[i]:
                    vec = cloud.nodes[j] - cloud.nodes[i]
                    vec_norm = jnp.linalg.norm(vec)
                    vec_scaled = vec / vec_norm
                    if jnp.dot(normal, vec_scaled) + 1. <= 1e-1:
                        opposite_neighbours.append(j)
                        if vec_norm <= closest_distance:
                            closest_neighbour = j
                            closest_distance = vec_norm

                field = field.at[i].set(field[closest_neighbour])

    return field


def duplicate_robin_coeffs(boundary_conditions, cloud):
    """ Duplicate the Robin coefficients to the nodes of the facets they are applied to """

    robin_coeffs = {}
    new_boundary_conditions = {}

    for f_id, f_type in cloud.facet_types.items():

        if f_type == "r":
            node_ids = cloud.facet_nodes[f_id]

            if type(boundary_conditions[f_id]) == tuple:
                new_boundary_conditions[f_id] = boundary_conditions[f_id][0]
                betas = boundary_conditions[f_id][1]
                if callable(betas):
                    nodes = cloud.sorted_nodes[jnp.array(node_ids)]
                    betas = jax.vmap(betas)(nodes)
            else:
                warning.warn("Did not provide beta coefficients for Robin BC. Using zeros ...")
                new_boundary_conditions[f_id] = boundary_conditions[f_id]
                betas = jnp.zeros((len(node_ids)))

            for i in node_ids:
                ii = i - node_ids[0]     ## TODO: this assumes consistent ordering. Check this !
                robin_coeffs[i] = betas[ii]

        else:
            new_boundary_conditions[f_id] = boundary_conditions[f_id]

    return robin_coeffs, new_boundary_conditions




def zerofy_periodic_cond(boundary_conditions, cloud):
    """ Zero out the periodic boundary conditions (this is aplied before the PDE solve, to overwrite any value set by the user) """
    for f_id, f_type in cloud.facet_types.items():

        if f_type[0] == "p":
            ## TODO: If a function/value is given, it could amount to setting the difference from one boundary to another ... Tricky !
            node_ids = cloud.facet_nodes[f_id]
            boundary_conditions[f_id] = jnp.zeros((len(node_ids)))

    return boundary_conditions



def pde_solver( diff_operator:callable,
                rhs_operator:callable,
                cloud:Cloud, 
                boundary_conditions:dict, 
                rbf:callable,
                max_degree:int,
                diff_args = None,
                rhs_args = None):
    """Solves a PDE using radial basis functions

    Args:
        diff_operator (callable): The differential operator (the left-hand side of the PDE), evaluated at each internal point with respect to each RBF centroid seperately, i.e. *nodal* evaluation.
        rhs_operator (callable): The right-hand-side operator, evaluated at each node with respect to all centroids at once, i.e. *global* evaluation.
        cloud (Cloud): The cloud on which to solve the PDE
        boundary_conditions (dict): The boundary conditions to enforce, one for each facet given by an approiate vector
        rbf (callable): The radial basis function to use
        max_degree (int): The maximum degree of the polynomial to use in the RBF
        diff_args (list, optional): The arguments to pass to the differential operator. Defaults to None.
        rhs_args (list, optional): The arguments to pass to the right-hand-side operator. Defaults to None.

    Returns:
        SteadySol: A named tuple containing the values and coefficients of the solution, as well as the matrix used in the linear solve
    """

    ## Jit-compile the operators
    diff_operator = jax.jit(diff_operator, static_argnums=[2,3])
    rhs_operator = jax.jit(rhs_operator, static_argnums=2)

    ## Set the global variables for later use
    UPDES.RBF = rbf
    UPDES.MAX_DEGREE = max_degree
    UPDES.DIM = cloud.dim

    ## Build robin coeffs
    robin_coeffs, boundary_conditions = duplicate_robin_coeffs(boundary_conditions, cloud)

    ## Zero out periodic conditions
    boundary_conditions = zerofy_periodic_cond(boundary_conditions, cloud)

    ## Compute the number of monomials needed
    nb_monomials = compute_nb_monomials(max_degree, cloud.dim)

    ## Assemble the quantities required
    B1 = assemble_B(diff_operator, cloud, rbf, nb_monomials, diff_args, robin_coeffs)
    rhs = assemble_q(rhs_operator, boundary_conditions, cloud, rbf, nb_monomials, rhs_args)

    ## Solve the linear system using JAX's direct solver
    # sol_vals = jnp.linalg.solve(B1, rhs)

    ## Solve the linear system using Scipy's iterative solver
    # sol_vals = jax.scipy.sparse.linalg.gmres(B1, rhs, tol=1e-5)[0]

    ## Solve the linear system using Lineax
    operator = lx.MatrixLinearOperator(B1)
    sol_vals = lx.linear_solve(operator, rhs, solver=lx.QR()).value
    # sol_vals = lx.linear_solve(operator, rhs, solver=lx.GMRES(rtol=1e-3, atol=1e-3)).value

    sol_coeffs = core_compute_coefficients(sol_vals, cloud, rbf, nb_monomials)

    return SteadySol(sol_vals, sol_coeffs, B1)


pde_solver_jit_with_bc = jax.jit(pde_solver, static_argnames=["diff_operator",
                                                    "rhs_operator",
                                                    "cloud",
                                                    "rbf",
                                                    "max_degree"])


def boundary_conditions_func_to_arr(boundary_conditions, cloud):
    """ Convert the given boundary conditions from functions to an array """

    boundary_conditions_arr = {}

    for f_id, f_bc in boundary_conditions.items():
        if callable(f_bc):
            f_nodes = cloud.sorted_nodes[jnp.array(cloud.facet_nodes[f_id])]
            boundary_conditions_arr[f_id] = jax.vmap(f_bc)(f_nodes)
        elif type(f_bc) == tuple:
            f_nodes = cloud.sorted_nodes[jnp.array(cloud.facet_nodes[f_id])]
            boundary_conditions_arr[f_id] = f_bc
            if callable(f_bc[0]):
                boundary_conditions_arr[f_id] = (jax.vmap(f_bc[0])(f_nodes), f_bc[1])
            if callable(f_bc[1]):
                boundary_conditions_arr[f_id] = (f_bc[0], jax.vmap(f_bc[1])(f_nodes))
        else:
            boundary_conditions_arr[f_id] = f_bc

    return boundary_conditions_arr


def pde_solver_jit( diff_operator:callable,
                rhs_operator:callable,
                cloud:Cloud, 
                boundary_conditions:dict, 
                rbf:callable,
                max_degree:int,
                diff_args = None,
                rhs_args = None):
    """PDE solver just-in-time compiled with respect to the boundary conditions

    Args:
        diff_operator (callable): The differential operator (the left-hand side of the PDE), evaluated at each internal point with respect to each RBF centroid seperately, i.e. *nodal* evaluation.
        rhs_operator (callable): The right-hand-side operator, evaluated at each node with respect to all centroids at once, i.e. *global* evaluation.
        cloud (Cloud): The cloud on which to solve the PDE
        boundary_conditions (dict): The boundary conditions to enforce, one for each facet given by either a function or an approiate vector
        rbf (callable): The radial basis function to use
        max_degree (int): The maximum degree of the polynomial to use in the RBF
        diff_args (list, optional): The arguments to pass to the differential operator. Defaults to None.
        rhs_args (list, optional): The arguments to pass to the right-hand-side operator. Defaults to None.

    Returns:
        SteadySol: A named tuple containing the values and coefficients of the solution, as well as the matrix used in the linear solve
    """

    boundary_conditions_arr = boundary_conditions_func_to_arr(boundary_conditions, cloud)

    return pde_solver_jit_with_bc(diff_operator=diff_operator,
                                    rhs_operator=rhs_operator,
                                    cloud=cloud, 
                                    boundary_conditions=boundary_conditions_arr, 
                                    rbf=rbf,
                                    max_degree=max_degree,
                                    diff_args = diff_args,
                                    rhs_args = rhs_args)












def pde_multi_solver( diff_operators:list,
                rhs_operators:list,
                cloud:Cloud, 
                boundary_conditions:list, 
                rbf:callable,
                max_degree:int,
                nb_iters:int=10,
                tol:float=1e-6,
                diff_args:list=None,
                rhs_args:list=None):
    """Solves a system of (non-linear) PDEs using an iterative approach for radial basis functions (see pde_solver for details on scalar PDEs)

    Args:
        diff_operators (list[callable]): The (nodal) differential operators (the left-hand side of the PDEs)
        rhs_operator (list[callable]): The (global) right-hand-side operator
        cloud (Cloud): The same cloud on which to solve the PDEs
        boundary_conditionss (list[dict]): The boundary conditions to enforce, one for each PDE
        rbf (callable): The radial basis function to use
        max_degree (int): The maximum degree of the polynomial to use in the RBF
        nb_iters (int, optional): The maximum number of iterations to use in the solver. Defaults to 10.
        tol (float, optional): The tolerance to check for convergence. Defaults to 1e-6. (Currently not used, because of JIT-issues)
        diff_args (list[list], optional): The arguments to pass to each differential operator. Defaults to list of Nones.
        rhs_args (list[list], optional): The arguments to pass to each right-hand-side operator. Defaults to list of None.

    Raises:
        AssertionError: The number of differential operators must match the number of right-hand side operators

    Returns:
        list[SteadySol]: A list of named named tuples containing the solutions for each PDE
    """

    # Number of scalar fields in the PDE
    n = len(diff_operators)
    assert n == len(rhs_operators) == len(boundary_conditions), "The number of differential operators must match the number of right-hand side operators"

    diff_operators = [jax.jit(diff_op, static_argnums=[2,3]) for diff_op in diff_operators]
    rhs_operators = [jax.jit(rhs_op, static_argnums=2) for rhs_op in rhs_operators]

    UPDES.RBF = rbf
    UPDES.MAX_DEGREE = max_degree
    UPDES.DIM = cloud.dim


    ## Build robin coeffs
    new_boundary_conditions = []
    robin_coefs = []
    for bcs in boundary_conditions:
        rcs, new_bcs = duplicate_robin_coeffs(bcs, cloud)
        new_bcs = zerofy_periodic_cond(new_bcs, cloud)
        new_bcs = boundary_conditions_func_to_arr(new_bcs, cloud)

        robin_coefs.append(rcs)
        new_boundary_conditions.append(new_bcs)

    boundary_conditions = new_boundary_conditions

    sols_vals = [u for u in diff_args[0][:n]]
    for k in range(nb_iters):
        sols = [pde_solver_jit_with_bc(diff_operators[i],
                                        rhs_operators[i],
                                        cloud, 
                                        boundary_conditions[i], 
                                        rbf,
                                        max_degree,
                                        diff_args = sols_vals + diff_args[i][n:],
                                        rhs_args = rhs_args[i]) for i in range(n)]
        total_error = 0.
        for i in range(n):
            total_error += jnp.linalg.norm(sols[i].vals - sols_vals[i]) / (jnp.linalg.norm(sols_vals[i]) + 1e-14)

        # if total_error < tol:     ## TODO: Circumvent this JIT issue
        #     break

        sols_vals = [s.vals for s in sols]

    return sols





## Vectorized versions of the dot operator
dot_vec = jax.jit(jax.vmap(jnp.dot, in_axes=(0, 0), out_axes=0))
dot_mat = jax.jit(jax.vmap(lambda J, v: J@v, in_axes=(0,0), out_axes=0))
