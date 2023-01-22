import jax.numpy as jnp
from utils import distance
from functools import partial

def multiquadric(r):
    eps = 1
    return jnp.sqrt(1 + (eps*r)**2)

def polyharmonic(r):
    a = 1
    return r**(2*a+1)

def get_rbf_function(rbf="polyharmonic", **kwargs):
    """ Gives the tuned rbf function """

    if rbf=="multiquadric":
        return partial(multiquadric, args=kwargs)

    elif rbf=="polyharmonic":
        return polyharmonic

def nodal_rbf(x, node_j, rbf="polyharmonic", **kwargs):
    """ When the rbf is applied to a node, rather than a distance """
    rbf_func = get_rbf_function(rbf, kwargs)
    return rbf_func(distance(x, node_j))



def monomial_1(vector):
    return jnp.array([1.0])

def monomial_x(vector):
    return vector[0]

def monomial_y(vector):
    return vector[1]

def monomial_xx(vector):
    return vector[0]**2

def monomial_xy(vector):
    return vector[0]*vector[1]

def monomial_yy(vector):
    return vector[1]**2

def polynomial(i):
    """ Just a way to keep track of polynomials"""
    if i == 0:
        return monomial_1
    elif i == 1:
        return monomial_x
    elif i == 2:
        return monomial_y
    elif i == 3:
        return monomial_xx
    elif i == 4:
        return monomial_xy
    elif i == 5:
        return monomial_yy
