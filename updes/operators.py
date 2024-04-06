import functools
import jax
import jax.numpy as jnp
from jax.tree_util import Partial, tree_map

from functools import cache, lru_cache, partial

# from updec.config import RBF, MAX_DEGREE, DIM
import updes.config as UPDEC
from updes.utils import compute_nb_monomials, SteadySol,  make_all_monomials
from updes.cloud import Cloud, SquareCloud
from updes.assembly import assemble_B, assemble_q, core_compute_coefficients


@Partial(jax.jit, static_argnums=[2,3])
def nodal_value(x, center=None, rbf=None, monomial=None):
    """ Computes the rbf and polynomial functions """
    """ x: gradient at position x 
        node: the node defining the rbf function (if for a monomial)
        rbf: rbf function to use
        monomial: the id of the monomial (if for a monomial)
    """
    ## Only one of node_j or monomial_j can be given
    if center != None:          ## TODO if else can be handled with grace in Jax
        # rbf = Partial(make_rbf, rbf=rbf)
        return rbf(x, center)
    elif monomial != None:
        # monomial = Partial(make_monomial, id=monomial)
        return monomial(x)


@cache
def core_gradient_rbf(rbf):
    return jax.grad(rbf)
    # return polyharmonic_grad

## LRU cache this
# @lru_cache(maxsize=32)
@cache
def core_gradient_mon(monomial):
    # monomial = Partial(make_monomial, id=monomial)
    return jax.grad(monomial)

## Does calling this all the time cause problems ?
def nodal_gradient(x, center=None, rbf=None, monomial=None):
    """ Computes the gradients of the RBF and polynomial functions """
    """ x: gradient at position x 
        node: the node defining the rbf function (if for a monomial)
        rbf: rbf function to use
        monomial: the id of the monomial (if for a monomial)
    """
    # cfg.RBFtest = gaussian
    ## Only one of node_j or monomial_j can be given
    if center != None:  ## TODO if alse can be handled with grace in Jax
        # nodal_rbf = Partial(make_nodal_rbf, rbf=rbf)
        # return jax.grad(nodal_rbf)(x, node)
        # return jnp.where(jnp.all(x==node), jax.grad(nodal_rbf)(x, node), jnp.array([0., 0.]))
        return jnp.nan_to_num(core_gradient_rbf(rbf)(x, center), posinf=0., neginf=0.)                    ## TODO: benchmark agains line above to see the cost of avoiding NaNs
    elif monomial != None:
        # monomial = Partial(make_monomial, id=monomial)
        # return jax.grad(monomial)(x)
        return core_gradient_mon(monomial)(x)

### N.B: """ No divergence for RBF and Polynomial functions, because they are scalars """

## LRU cache this
# @lru_cache(maxsize=32)
@cache
def core_laplacian_rbf(rbf):
    return jax.jacfwd(jax.grad(rbf))

## LRU cache this
# @lru_cache(maxsize=32)
@cache
def core_laplacian_mon(monomial):
    # monomial = Partial(make_monomial, id=monomial)
    return jax.jacfwd(jax.grad(monomial))

def nodal_laplacian(x, center=None, rbf=None, monomial=None):     ## TODO Jitt through this efficiently
    """ Computes the lapalcian of the RBF and polynomial functions """
    if center != None:
        # nodal_rbf = Partial(make_nodal_rbf, rbf=rbf)
        # nodal_rbf = rbf
        # return jnp.trace(core_laplacian_rbf(rbf)(x, center))      ## TODO: try reverse mode
        return jnp.nan_to_num(jnp.trace(core_laplacian_rbf(rbf)(x, center)), posinf=0., neginf=0.)      ## TODO: try reverse mode
    elif monomial != None:
        # monomial = Partial(make_monomial, id=monomial)
        # monomial = monomial
        return jnp.trace(core_laplacian_mon(monomial)(x))




def nodal_div_grad(x, center=None, rbf=None, monomial=None, args=None):
    """ Computes the lapalcian of the RBF and polynomial functions """
    matrix = jnp.stack((args, args), axis=-1)
    if center != None:
        return jnp.nan_to_num(jnp.trace(matrix * core_laplacian_rbf(rbf)(x, center)), posinf=0., neginf=0.)
    elif monomial != None:
        # return jnp.nan_to_num(jnp.trace(matrix * core_laplacian_mon(monomial)(x)), posinf=0., neginf=0.)
        return jnp.trace(matrix * core_laplacian_mon(monomial)(x))


_nodal_value_rbf_vec = jax.vmap(nodal_value, in_axes=(None, 0, None, None), out_axes=0)


def value(x, field, centers, rbf=None, clip_val=None):
    """ Computes the value of field quantity s at position x """
    ## Find coefficients for s on the cloud
    # if field.shpae[0] == centers.shape[0]:   ## If the field is defined by its values - get its coefficients


    ## Now, compute the value of the field
    N = centers.shape[0]
    lambdas, gammas = field[:N], field[N:]

    val_rbf = _nodal_value_rbf_vec(x, centers, rbf, None)
    final_val = jnp.sum(jnp.nan_to_num(lambdas*val_rbf, posinf=0., neginf=0.), axis=0)

    all_monomials = make_all_monomials(gammas.shape[0])

    for j in range(gammas.shape[0]):    ### TODO: Use FOR_I LOOP
        polynomial_val = nodal_value(x, monomial=all_monomials[j])
        final_val = final_val + jnp.nan_to_num(gammas[j] * polynomial_val, posinf=0., neginf=0.)

    # def mon_val_body(j, val):
    #     polynomial_val = nodal_value(x, monomial=all_monomials[j])
    #     return val + jnp.nan_to_num(gammas[j] * polynomial_val, posinf=0., neginf=0.)
    # final_val = jax.lax.fori_loop(0, gammas.shape[0], mon_val_body, final_val)

    # return final_val
    # return jnp.clip(final_val, -1, 1)
    if clip_val:
        return jnp.clip(final_val, -clip_val, clip_val)
    else:
        return final_val

value_vec_ = jax.vmap(value, in_axes=(0, None, None, None), out_axes=0)
value_vec = jax.jit(value_vec_, static_argnums=[3])


_nodal_gradient_rbf_vec = jax.vmap(nodal_gradient, in_axes=(None, 0, None, None), out_axes=0)

### ATEMPT TO VECTORIZE MONOMIALS THTOUGH TREE_MAP ###
# @partial(jax.vmap, in_axes=(None, 0), out_axes=0)
# def _nodal_gradient_mon(monomial, x):
#     return nodal_gradient(x, None, None, monomial)
# # _nodal_gradient_mon_vec = jax.vmap(nodal_gradient, in_axes=(None, None, None, 0), out_axes=0)
# @cache
# def _nodal_gradient_mon_vec(monomials):                     ## Vectorized using pytrees
#     return jnp.stack(tree_map(_nodal_gradient_mon, monomials))


# @Partial(jax.jit, static_argnums=3)
def gradient(x, field, centers, rbf=None, clip_val=None):
    """ Computes the gradient of field quantity at position x 
        The field is defined by its _coefficients_ in the RBF basis """

    N = centers.shape[0]
    lambdas, gammas = field[:N], field[N:]

    grads_rbf = jnp.nan_to_num(_nodal_gradient_rbf_vec(x, centers, rbf, None), posinf=0., neginf=0.)              ## TODO remove all NaNs
    # print(grads_rbf)
    lambdas = jnp.stack([lambdas, lambdas], axis=-1)                        ## TODO Why is Jax unable to broadcast below ?
    final_grad = jnp.sum(jnp.nan_to_num(lambdas*grads_rbf, posinf=0., neginf=0.), axis=0)

    all_monomials = make_all_monomials(gammas.shape[0])

    for j in range(gammas.shape[0]):                                                   ### TODO: Use FOR_I LOOP
        polynomial_grad = nodal_gradient(x, monomial=all_monomials[j])
        final_grad = final_grad.at[:].add(jnp.nan_to_num(gammas[j] * polynomial_grad, posinf=0., neginf=0.))

    # def mon_grad_body(j, grad):
    #     polynomial_grad = nodal_gradient(x, monomial=all_monomials[j])
    #     return grad + jnp.nan_to_num(gammas[j] * polynomial_grad, posinf=0., neginf=0.)
    # final_grad = jax.lax.fori_loop(0, gammas.shape[0], mon_grad_body, final_grad)

    # return final_grad
    # return jnp.clip(final_grad, -1, 1)
        
    if clip_val:
        return jnp.clip(final_grad, -clip_val, clip_val)
    else:
        return final_grad

gradient_vec_ = jax.vmap(gradient, in_axes=(0, None, None, None), out_axes=0)
gradient_vec = jax.jit(gradient_vec_, static_argnums=[3])


def gradient_vals(x, field, cloud, rbf, max_degree):
    """ Computes the gradient of field quantity at position x 
        The field is defined by its _values_ in the RBF basis """

    nb_monomials = compute_nb_monomials(max_degree, cloud.dim)
    coeffs = core_compute_coefficients(field, cloud, rbf, nb_monomials)

    return gradient(x, coeffs, cloud.sorted_nodes, rbf)

gradient_vals_vec_ = jax.vmap(gradient_vals, in_axes=(0, None, None, None, None), out_axes=0)
gradient_vals_vec = jax.jit(gradient_vals_vec_, static_argnums=[2,3,4])


def cartesian_gradient(node_id, field, cloud:Cloud, clip_val=None):
    """ Computes the gradient of field quantity at position x """

    N = field.shape[0]
    # i = int(node_id)
    # i = node_id[0]
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
            if jnp.dot(direction, vec_scaled) + 1. <= 1e-1:    ##TODO Keep reducing the tolerance untile you find one
                opposite_neighbours.append(j)
                if vec_norm <= closest_distance:
                    closest_neighbour = j
                    closest_distance = vec_norm
        
        if closest_neighbour == None:
            # print("Warning: couldn't find good neighbor !")
            # print("Current Neumann node:", i)
            # print("Opposite neighbours:", opposite_neighbours)
            # print("Closest neighbour:", closest_neighbour)
            return final_grad

        final_grad = final_grad.at[d].set((field[i] - field[closest_neighbour]) / vec_norm)

    # return final_grad
    if clip_val:
        return jnp.clip(final_grad, -clip_val, clip_val)
    else:
        return final_grad


# cartesian_gradient_vec = jax.jit(jax.vmap(cartesian_gradient, in_axes=(0, None, None), out_axes=0), static_argnums=0)
# cartesian_gradient_vec = jax.vmap(cartesian_gradient, in_axes=(0, None, None), out_axes=0)

def cartesian_gradient_vec(node_ids, field, cloud:Cloud):       ## TODO beacause JAX has issues with concretisation and tracing
    # N = field.shape[0]
    # grad = jnp.ones((len(node_ids), 2))
    grad = jnp.zeros((len(node_ids), 2))  ## TODO handle zero case
    for node_id in node_ids:
        grad = grad.at[node_id, :].set(cartesian_gradient(node_id, field, cloud))
    return grad


def enforce_cartesian_gradient_neumann(field, grads, boundary_conditions, cloud, clip_val=None):
    """ Set the gradient at every neumann node using catesian grid """

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
                    if jnp.dot(normal, vec_scaled) + 1. <= 1e-1:    ##TODO Keep reducing the tolerance untile you find one
                        opposite_neighbours.append(j)
                        if vec_norm <= closest_distance:
                            closest_neighbour = j
                            closest_distance = vec_norm

                # print("Current Neumann node:", i)
                # print("Opposite neighbours:", opposite_neighbours)
                # print("Closest neighbour:", closest_neighbour)

                grads = grads.at[i].set((field[i] - field[closest_neighbour]) / closest_distance)

    # return grads
    if clip_val:
        return jnp.clip(grads, -clip_val, clip_val)
    else:
        return grads


def divergence(x, field, centers, rbf=None, clip_val=None):
    """ Computes the divergence of vector quantity s at position x """
    dfieldx_dx = gradient(x, field[...,0], centers, rbf)[0]
    dfieldy_dy = gradient(x, field[...,1], centers, rbf)[1]

    # return dfieldx_dx + dfieldy_dy
    if clip_val:
        return jnp.clip(dfieldx_dx + dfieldy_dy, -clip_val, clip_val)
    else:
        return dfieldx_dx + dfieldy_dy

divergence_vec = jax.vmap(divergence, in_axes=(0, None, None, None), out_axes=0)


_nodal_laplacian_rbf_vec = jax.vmap(nodal_laplacian, in_axes=(None, 0, None, None), out_axes=0)

def laplacian(x, field, centers, rbf=None, clip_val=None):
    """ Computes the laplacian of quantity field at position x """

    ## Now, compute the laplacian of the field

    N = centers.shape[0]
    lambdas, gammas = field[:N], field[N:]

    laps_rbf = _nodal_laplacian_rbf_vec(x, centers, rbf, None)              ## TODO remove all NaNs
    rbf_lap = jnp.sum(jnp.nan_to_num(lambdas*laps_rbf, posinf=0., neginf=0.), axis=0)


    all_monomials = make_all_monomials(gammas.shape[0])

    mon_lap = jnp.array([0.])
    for j in range(gammas.shape[0]):    ## TODO use FOR_I LOOP
        # monomial = Partial(make_monomial, id=j)
        poly_lap = nodal_laplacian(x, monomial=all_monomials[j])
        mon_lap = mon_lap.at[:].add(gammas[j] * poly_lap)

    # def mon_lap_body(j, lap):
    #     poly_lap = nodal_laplacian(x, monomial=all_monomials[j])
    #     return lap + jnp.nan_to_num(gammas[j] * poly_lap, posinf=0., neginf=0.)
    # mon_lap = jax.lax.fori_loop(0, gammas.shape[0], mon_lap_body, jnp.array([0.]))

    # return rbf_lap + mon_lap[0]
    # return jnp.clip(rbf_lap + mon_lap[0], -5e-1, 5e-1)
    if clip_val:
        return jnp.clip(rbf_lap + mon_lap[0], -clip_val, clip_val)
    else:
        return rbf_lap + mon_lap[0]


laplacian_vec = jax.vmap(laplacian, in_axes=(0, None, None, None), out_axes=0)


def laplacian_vals(x, field, cloud, rbf, max_degree):
    """ Computes the gradient of field quantity at position x 
        The field is defined by its _values_ in the RBF basis """

    nb_monomials = compute_nb_monomials(max_degree, cloud.dim)
    coeffs = core_compute_coefficients(field, cloud, rbf, nb_monomials)

    return laplacian(x, coeffs, cloud.sorted_nodes, rbf)

laplacian_vals_vec_ = jax.vmap(laplacian_vals, in_axes=(0, None, None, None, None), out_axes=0)
laplacian_vals_vec = jax.jit(laplacian_vals_vec_, static_argnums=[2,3,4])









def integrate_field(field, cloud, rbf, max_degree):
    """ Integrate over the 2D square cloud domain, using the midpoint rule 
    1. Identify the small squares in the domain (all identical)
    2. Get the field value at the center of each square
    4. Account for border and corner values: https://stackoverflow.com/a/62991037/8140182
    3. Compute the approximate integral

    NB: the fiel is defined by its coefficients in the RBF basis
    """

    ## Assert we have a SquareCloud instance
    assert isinstance(cloud, SquareCloud), "The cloud must be a SquareCloud instance"

    ## Compute the number of squares in the domain
    nb_squares = (cloud.Nx - 1) * (cloud.Ny - 1)
    # print("Number of squares:", nb_squares)

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
    """ Interpolates field from cloud1 to cloud2 """

    assert cloud1.N == cloud2.N, "the two clouds do not contain the same number of nodes"   ## TODO: Make sure only the renumbering differs

    sorted_map1 = sorted(cloud1.renumbering_map.items(), key=lambda x:x[0])
    indexer1 = jnp.array(list(dict(sorted_map1).values()))
    field_orig = field[indexer1]

    indexer2 = jnp.array(list(cloud2.renumbering_map.keys()))

    return field_orig[indexer2]            ## TODO Think of a way to do this


def apply_neumann_conditions(field, boundary_conditions, cloud:Cloud):

    # print("Reverse inndices", cloud.global_indices_rev)
    # south_nodes = cloud.facet_nodes["South"]
    # for nid in south_nodes:
    #     i, j = cloud.global_indices_rev[nid]
    #     ii, jj = i, j+1
    #     neighbour_id = cloud.global_indices[ii, jj]
    #     sol_vals = sol_vals.at[nid].set(sol_vals[neighbour_id])

    for facet_id, facet_type in cloud.facet_types.items():
        if facet_type == "n":
            # nm_nodes = cloud.facet_nodes["Outflow"]
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
                    if jnp.dot(normal, vec_scaled) + 1. <= 1e-1:    ##TODO Keep reducing the tolerance untile you find one
                        opposite_neighbours.append(j)
                        if vec_norm <= closest_distance:
                            closest_neighbour = j
                            closest_distance = vec_norm

                # print("Current Neumann node:", i)
                # print("Opposite neighbours:", opposite_neighbours)
                # print("Closest neighbour:", closest_neighbour)

                field = field.at[i].set(field[closest_neighbour])

    return field


def duplicate_robin_coeffs(boundary_conditions, cloud):
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
                print("WARNING: Did not provide beta coefficients for Robin BC. Using zeros ...")
                new_boundary_conditions[f_id] = boundary_conditions[f_id]
                betas = jnp.zeros((len(node_ids)))

            for i in node_ids:
                ii = i - node_ids[0]     ## TODO Assumes ordering. OMG fix this, as well as providing nodes as arrays.
                robin_coeffs[i] = betas[ii]

        else:
            new_boundary_conditions[f_id] = boundary_conditions[f_id]

    return robin_coeffs, new_boundary_conditions




def zerofy_periodic_cond(boundary_conditions, cloud):
    for f_id, f_type in cloud.facet_types.items():

        if f_type[0] == "p":
            """ Check that no periodic condition is given """ ## If this was done, it would amoun to setting the difference from one boundary to another, plus the differential as well... Tricky !

            # if f_id in boundary_conditions:
            #     print(f"WARNING: Values given for periodic boundary condition at facet {f_id}. Ignoring it ...")

            node_ids = cloud.facet_nodes[f_id]
            boundary_conditions[f_id] = jnp.zeros((len(node_ids)))

    return boundary_conditions



## Devise different LU, LDL decomposition strategies make functions here
def pde_solver( diff_operator:callable,
                rhs_operator:callable,
                cloud:Cloud, 
                boundary_conditions:dict, 
                rbf:callable,
                max_degree:int,
                diff_args = None,
                rhs_args = None):
    """ Solve a PDE 
    cloud: the cloud of points on which the PDE is solved
    diff_operator: can take as input the coeffcients of a field
    diff_args: can be either coeffs or values of the fields. Ultimately, only coefficients will be passed the diff operators
    rhs_args: can be either coeffs or values of the fields. Ultimately, only coefficients will be passed the rhs operators
    """

    # nodal_operator = jax.jit(nodal_operator, static_argnums=2)
    diff_operator = jax.jit(diff_operator, static_argnums=[2,3])
    rhs_operator = jax.jit(rhs_operator, static_argnums=2)

    UPDEC.RBF = rbf
    ### For rememmering purposes
    UPDEC.MAX_DEGREE = max_degree
    UPDEC.DIM = cloud.dim


    ## Build robin coeffs
    robin_coeffs, boundary_conditions = duplicate_robin_coeffs(boundary_conditions, cloud)

    ## Zero out periodic conditions
    boundary_conditions = zerofy_periodic_cond(boundary_conditions, cloud)

    # TODO Here
    nb_monomials = compute_nb_monomials(max_degree, cloud.dim)

    B1 = assemble_B(diff_operator, cloud, rbf, nb_monomials, diff_args, robin_coeffs)
    rhs = assemble_q(rhs_operator, boundary_conditions, cloud, rbf, nb_monomials, rhs_args)

    sol_vals = jnp.linalg.solve(B1, rhs)
    sol_coeffs = core_compute_coefficients(sol_vals, cloud, rbf, nb_monomials)

    # sol_vals = apply_neumann_conditions(sol_vals, boundary_conditions, cloud)

    # return sol_vals, jnp.concatenate(sol_coeffs)         ## TODO: return an object like solve_ivp
    return SteadySol(sol_vals, sol_coeffs, B1)


# pde_solver_jit_without_bc = jax.jit(pde_solver, static_argnames=["diff_operator",
#                                                     "rhs_operator",
#                                                     "cloud",
#                                                     "boundary_conditions",           ## BCs are static here
#                                                     "rbf",
#                                                     "max_degree"])

pde_solver_jit_with_bc = jax.jit(pde_solver, static_argnames=["diff_operator",
                                                    "rhs_operator",
                                                    "cloud",
                                                    "rbf",
                                                    "max_degree"])


def boundary_conditions_func_to_arr(boundary_conditions, cloud):
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
    """ Jitted PDE solver """
    boundary_conditions_arr = boundary_conditions_func_to_arr(boundary_conditions, cloud)

    # pde_solver_jit_with_bc = jax.jit(pde_solver, static_argnames=["diff_operator",
    #                                                     "rhs_operator",
    #                                                     "cloud",
    #                                                     "rbf",
    #                                                     "max_degree"])

    return pde_solver_jit_with_bc(diff_operator=diff_operator,
                                    rhs_operator=rhs_operator,
                                    cloud=cloud, 
                                    boundary_conditions=boundary_conditions_arr, 
                                    rbf=rbf,
                                    max_degree=max_degree,
                                    diff_args = diff_args,
                                    rhs_args = rhs_args)












## Solve a non-scalar PDE via iteration
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
    """ Solve a PDE """

    ## One conditions to use this function: in diff args and rhs args, the first n arguments must be the scalar fields we are solving for


    # Number of scalar fields in the PDE
    n = len(diff_operators)
    assert n == len(rhs_operators) == len(boundary_conditions), "The number of differential operators must match the number of right-hand side operators"

    diff_operators = [jax.jit(diff_op, static_argnums=[2,3]) for diff_op in diff_operators]
    rhs_operators = [jax.jit(rhs_op, static_argnums=2) for rhs_op in rhs_operators]

    UPDEC.RBF = rbf
    ### For rememmering purposes
    UPDEC.MAX_DEGREE = max_degree
    UPDEC.DIM = cloud.dim


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

        if total_error < tol:
            break

        sols_vals = [s.vals for s in sols]

    return sols






## Solve a non-scalar PDE via iteration
def pde_multi_solver_unbounded( diff_operators:list,
                rhs_operators:list,
                cloud:Cloud, 
                boundary_conditions:list, 
                rbf:callable,
                max_degree:int,
                nb_iters:int=10,
                tol:float=1e-6,
                diff_args:list=None,
                rhs_args:list=None):
    """ Solve a PDE """

    ## One conditions to use this function: in diff args and rhs args, the first n arguments must be the scalar fields we are solving for


    # Number of scalar fields in the PDE
    n = len(diff_operators)
    assert n == len(rhs_operators) == len(boundary_conditions), "The number of differential operators must match the number of right-hand side operators"

    diff_operators = [jax.jit(diff_op, static_argnums=[2,3]) for diff_op in diff_operators]
    rhs_operators = [jax.jit(rhs_op, static_argnums=2) for rhs_op in rhs_operators]

    UPDEC.RBF = rbf
    ### For rememmering purposes
    UPDEC.MAX_DEGREE = max_degree
    UPDEC.DIM = cloud.dim


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

        sols_vals = [s.vals for s in sols]

    return sols


















dot_vec = jax.jit(jax.vmap(jnp.dot, in_axes=(0, 0), out_axes=0))
dot_mat = jax.jit(jax.vmap(lambda J, v: J@v, in_axes=(0,0), out_axes=0))