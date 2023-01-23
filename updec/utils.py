import jax.numpy as jnp
import math

# import matplotlib.pyplot as plt
# import seaborn as sns


## Euclidian distance
def distance(node1, node2):
    # return (node1 - node2)@(node1 - node2).T      ## Squared distance
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


def make_nodal_rbf(x, node, rbf):
    """ Gives the tuned rbf function """
    if rbf==None:
        func = polyharmonic
    else:
        func = rbf
    return func(distance(x, node))


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
    else:
        pass        ## Higher order monomials not yet supported

def compute_nb_monomials(max_degree, problem_dimension):
    return math.comb(max_degree+problem_dimension, max_degree)


# # plt.style.use('bmh')

# sns.set(context='notebook', style='ticks',
#         font='sans-serif', font_scale=1, color_codes=True, rc={"lines.linewidth": 2})

# ## Wrapper function for matplotlib and seaborn
# def plot(*args, ax=None, figsize=(6,3.5), x_label=None, y_label=None, title=None, y_scale='linear', **kwargs):
#     if ax==None: 
#         _, ax = plt.subplots(1, 1, figsize=figsize)
#     # sns.despine(ax=ax)
#     if x_label:
#         ax.set_xlabel(x_label)
#     if y_label:
#         ax.set_ylabel(y_label)
#     if title:
#         ax.set_title(title)
#     ax.plot(*args, **kwargs)
#     ax.set_yscale(y_scale)
#     ax.legend()
#     plt.tight_layout()
#     return ax
