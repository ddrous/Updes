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

## Euclidian distance
def distance(node1, node2):
    # diff = node1 - node2
    # return jnp.sum(diff*diff)      ## TODO Squared distance !!!!!!!!
    return jnp.linalg.norm(node1 - node2)       ## Carefull: not differentiable at 0


def print_line_by_line(dictionary):
    for k, v in dictionary.items():
        print("\t", k,":",v)


def multiquadric_func(r, eps):
    return jnp.sqrt(1 + (eps*r)**2)
@jax.jit
def multiquadric(x, center, eps=1.):
    return multiquadric_func(distance(x, center), eps)

def inv_multiquadric_func(r, eps):
    return 1./ jnp.sqrt(1 + (eps*r)**2)
@jax.jit
def inverse_multiquadric(x, center, eps=1.):
    return inv_multiquadric_func(distance(x, center), eps)

def gaussian_func(r, eps):
    return jnp.exp(-(eps * r)**2)
def gaussian(x, center, eps=1.):
    return gaussian_func(distance(x, center), eps)

def polyharmonic_func(r, a):
    return r**(2*a+1)
@jax.jit
def polyharmonic(x, center, a=1):
    return polyharmonic_func(distance(x, center), a)

def thin_plate_func(r, a):
    # return jnp.log(r) * r**(2*a)
    return jnp.nan_to_num(jnp.log(r) * r**(2*a), neginf=0., posinf=0.)
@jax.jit
def thin_plate(x, center, a=1):
    return thin_plate_func(distance(x, center), a)


# @jax.jit
@Partial(jax.jit, static_argnums=2)
def make_nodal_rbf(x, node, rbf):
    """ Gives the tuned rbf function """
    if rbf==None:
        func = polyharmonic
    else:
        func = rbf
    # return jnp.where(jnp.all(x==node), 0., func(distance(x, node)))     ## TODO Bad attempt to avoid differentiability
    return func(distance(x, node))


@Partial(jax.jit, static_argnums=1)
def make_monomial(x, id):
    """ Easy way to keep track of all monomials """
    ## x is a 2D vector
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
        pass        ## Higher order monomials not yet supported

@cache
def make_all_monomials(nb_monomials):
    # return jnp.array([Partial(make_monomial, id=j) for j in range(nb_monomials)])
    return [partial(make_monomial, id=j) for j in range(nb_monomials)]



def compute_nb_monomials(max_degree, problem_dimension):
    return math.comb(max_degree+problem_dimension, max_degree)


SteadySol = namedtuple('PDESolution', ['vals', 'coeffs'])



def random_name(length=5):
    "Make random names to identify runs"
    name = ""
    for _ in range(length):
        name += str(random.randint(0, 9))
    return name


def make_dir(path):
    "Make a directory if it doesn't exist"
    if os.path.exists(path):
        os.system("rm -rf " + path)
    os.mkdir(path)


# plt.style.use('bmh')
# sns.set(context='notebook', style='ticks',
#         font='sans-serif', font_scale=1, color_codes=True, rc={"lines.linewidth": 2})

## Wrapper function for matplotlib and seaborn
def plot(*args, ax=None, figsize=(6,3.5), x_label=None, y_label=None, title=None, x_scale='linear', y_scale='linear', **kwargs):
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
    if x_label:
        ax.legend()
    plt.tight_layout()
    return ax


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
