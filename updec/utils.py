import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns

import math
from functools import partial



## Euclidian distance
def distance(node1, node2):
    diff = node1 - node2
    # return jnp.sum(diff*diff)      ## TODO Squared distance !!!!!!!!
    return jnp.linalg.norm(node1 - node2)       ## Carefull: not differentiable at 0


def print_line_by_line(dictionary):
    for k, v in dictionary.items():
        print("\t", k,":",v)


def multiquadric(r):
    eps = 1
    return jnp.sqrt(1 + (eps*r)**2)


def polyharmonic(r):
    a = 1
    return r**(2*a+1)

def gaussian(r):
    eps = 0.1
    return jnp.exp(-r**2 / eps**2)

# @jax.jit
@partial(jax.jit, static_argnums=2)
def make_nodal_rbf(x, node, rbf):
    """ Gives the tuned rbf function """
    if rbf==None:
        func = polyharmonic
    else:
        func = rbf
    # return jnp.where(jnp.all(x==node), 0., func(distance(x, node)))     ## TODO Bad attempt to avoid differentiability
    return func(distance(x, node))


@partial(jax.jit, static_argnums=1)
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


def compute_nb_monomials(max_degree, problem_dimension):
    return math.comb(max_degree+problem_dimension, max_degree)


# plt.style.use('bmh')
# sns.set(context='notebook', style='ticks',
#         font='sans-serif', font_scale=1, color_codes=True, rc={"lines.linewidth": 2})

## Wrapper function for matplotlib and seaborn
def plot(*args, ax=None, figsize=(6,3.5), x_label=None, y_label=None, title=None, y_scale='linear', **kwargs):
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
    ax.set_yscale(y_scale)
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
