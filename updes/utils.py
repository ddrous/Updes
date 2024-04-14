import jax
import jax.numpy as jnp
from jax.tree_util import Partial

from collections import namedtuple
from functools import cache, partial
import os

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context='notebook', style='ticks',
        font='sans-serif', font_scale=1, color_codes=True, rc={"lines.linewidth": 2})
plt.style.use("dark_background")

import math
import random


def distance(node1, node2):
    """ Euclidean distance between two points. """
    diff = node1 - node2
    return jnp.sqrt(diff.T @ diff)


## Print each item of a dictionary in a new line
def print_line_by_line(dictionary):
    for k, v in dictionary.items():
        print("\t", k,":",v)

def multiquadric_func(r, eps):
    return jnp.sqrt(1 + (eps*r)**2)
@jax.jit
def multiquadric(x, center, eps=1.):
    """ Hardy's Multiquadric RBF """
    return multiquadric_func(distance(x, center), eps)

def inv_multiquadric_func(r, eps):
    return 1./ jnp.sqrt(1 + (eps*r)**2)
@jax.jit
def inverse_multiquadric(x, center, eps=1.):
    """ Inverse Multiquadric RBF """
    return inv_multiquadric_func(distance(x, center), eps)

def gaussian_func(r, eps):
    return jnp.exp(-(eps * r)**2)
def gaussian(x, center, eps=1.):
    """ Gaussian RBF """
    return gaussian_func(distance(x, center), eps)

def polyharmonic_func(r, a):
    return r**(2*a+1)
@jax.jit
def polyharmonic(x, center, a=1):
    """ Polyharmonic Spline RBF """
    return polyharmonic_func(distance(x, center), a)

## Gradient of Polyharmonic Spline RBF
# @jax.jit
# def polyharmonic_grad(x, center):
#     return 3 * distance(x, center) * (x - center) 

## Thin Plate Spline RBF
def thin_plate_func(r, a):
    # return jnp.log(r) * r**(2*a)
    return jnp.nan_to_num(jnp.log(r) * r**(2*a), neginf=0., posinf=0.)
@jax.jit
def thin_plate(x, center, a=1):
    """ Thin Plate Spline RBF """
    return thin_plate_func(distance(x, center), a)


@Partial(jax.jit, static_argnums=2)
def make_nodal_rbf(x, node, rbf):
    # """ Gives the tuned rbf function """
    """A function that returns the value of the RBF at a given point x, with respect to a given node. The RBF is tuned to the given node.

    Args:
        x (Float[Array, "dim"]): The point at which the RBF is to be evaluated.
        node (Float[Array, "dim"]): The centroid with respect to which the RBF is evaluated.
        rbf (Callable): The RBF function to be used, with signature rbf(r) where r is the Euclidean distance between the two points

    Returns:
        float: The scalar value of the RBF at the given point x, with respect to the given node.
    """
    if rbf==None:
        func = polyharmonic
    else:
        func = rbf
    return func(distance(x, node))


@Partial(jax.jit, static_argnums=1)
def make_monomial(x, id):
    """A function that returns the value of a monomial at a given point x.

    Args:
        x (Float[Array, "dim"]): The point at which the monomial is to be evaluated.
        id (int): The id of the monomial to be evaluated.

    Returns:
        float: The value of the monomial at the given point x.
    """
    if id == 0:
        return 1.0
    elif id == 1:
        return x[0]
    elif id == 2:
        return x[1]
    elif id == 3:
        return x[0]**2
    elif id == 4:
        return x[0]*x[1]
    elif id == 5:
        return x[1]**2
    elif id == 6:
        return x[0]**3
    elif id == 7:
        return (x[0]**2)*x[1]
    elif id == 8:
        return x[0]*(x[1]**2)
    elif id == 9:
        return x[1]**3
    elif id == 10:
        return x[0]**4
    elif id == 11:
        return (x[0]**3)*x[1]
    elif id == 12:
        return (x[0]**2)*(x[1]**2)
    elif id == 13:
        return x[0]*(x[1]**3)
    elif id == 14:
        return x[1]**4
    else:
        pass        ## TODO: support higher order monomials !

@cache
def make_all_monomials(nb_monomials):
    """A function that returns up to a certain number of monomials"""
    return [partial(make_monomial, id=j) for j in range(nb_monomials)]


def compute_nb_monomials(max_degree, problem_dimension):
    """Computes the number of monomials of dregree less than 'max_degree', in dimension 'problem_dimension'"""
    return math.comb(max_degree+problem_dimension, max_degree)


## This stores both the RBF coefficients, the values, and its matrix after solving a PDE
SteadySol = namedtuple('PDESolution', ['vals', 'coeffs', 'mat'])


def random_name(length=5):
    """Make up a random name to identify a run"""
    name = ""
    for _ in range(length):
        name += str(random.randint(0, 9))
    return name


def make_dir(path):
    "Make a directory if it doesn't exist"
    if not os.path.exists(path):
        os.mkdir(path)

def plot(*args, ax=None, figsize=(6,3.5), x_label=None, y_label=None, title=None, x_scale='linear', y_scale='linear', xlim=None, ylim=None, **kwargs):
    """Wrapper function for matplotlib and seaborn"""
    if ax==None: 
        _, ax = plt.subplots(1, 1, figsize=figsize)
    # sns.despine(ax=ax)
    if x_label:
        ax.set_xlabel(x_label)
    if y_label:
        ax.set_ylabel(y_label)
    if title:
        ax.set_title(title)
    ax.plot(*args, **kwargs)
    ax.set_xscale(x_scale)
    ax.set_yscale(y_scale)
    if "label" in kwargs.keys():
        ax.legend()
    if ylim:
        ax.set_ylim(ylim)
    if xlim:
        ax.set_xlim(xlim)
    plt.tight_layout()
    return ax

## A dataloader for mini-batch training if needed
def dataloader(array, batch_size, key):
    dataset_size = array.shape[0]
    indices = jnp.arange(dataset_size)
    perm = jax.random.permutation(key, indices)
    start = 0
    end = batch_size
    while end < dataset_size:
        batch_perm = perm[start:end]
        yield array[batch_perm]
        start = end
        end = start + batch_size



def RK4(fun, t_span, y0, *args, t_eval=None, subdivisions=1, **kwargs):
    """Numerical integration with RK4 and fixed-time stepping, but with fine subdivisions of the evaluation time intervals

    Args:
        fun (Callable): The function to be integrated.
        y0 (Float[Array]): The initial condition.
        t_span (Tuple): The time interval for which the integration is to be performed.
        t_eval (Float[Array]): The time points at which the solution is to be evaluated.
        subdivisions (int): To improve stability, each interval in t_eval is divided into this many subdivisions. Consider increasing this if you obtain NaNs.
        *args: Additional arguments to be passed to the function.
        **kwargs: Additional keyword arguments to be passed to the function.

    Raises:
        Warning: if t_span[0] is None.
        ValueError: if t_eval is None and t_span[1] is None.

    Returns:
        Float[Array, "nb_time_steps"]: The solution at the time points in t_eval.
    """
    if t_eval is None:
        if t_span[0] is None:
            t_eval = jnp.array([t_span[1]])
            raise Warning("t_span[0] is None. Setting t_span[0] to 0.")
        elif t_span[1] is None:
            raise ValueError("t_span[1] must be provided if t_eval is not.")
        else:
            t_eval = jnp.array(t_span)

    hs = t_eval[1:] - t_eval[:-1]
    t_ = t_eval[:-1, None] + jnp.arange(subdivisions)[None, :]*hs[:, None]/subdivisions
    t_solve = jnp.concatenate([t_.flatten(), t_eval[-1:]])
    eval_indices = jnp.arange(0, t_solve.size, subdivisions)

    def step(state, t):
        t_prev, y_prev = state
        h = t - t_prev
        k1 = h * fun(t_prev, y_prev, *args)
        k2 = h * fun(t_prev + h/2., y_prev + k1/2., *args)
        k3 = h * fun(t_prev + h/2., y_prev + k2/2., *args)
        k4 = h * fun(t + h, y_prev + k3, *args)
        y = y_prev + (k1 + 2*k2 + 2*k3 + k4) / 6.
        return (t, y), y

    _, ys = jax.lax.scan(step, (t_solve[0], y0), t_solve[:])
    return ys[eval_indices, :]
