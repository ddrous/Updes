import jax.numpy as jnp



## Euclidian distance
def distance(node1, node2):
    return jnp.linalg.norm(node1 - node2)**2

def print_line_by_line(dictionary):
    for k, v in dictionary.items():
        print("\t", k,":",v)


def multiquadric(r):
    eps = 1
    return jnp.sqrt(1 + (eps*r)**2)


def polyharmonic(r):
    a = 1
    return r**(2*a+1)


def make_nodal_rbf(rbf, x, node):
    """ Gives the tuned rbf function """
    if rbf==None:
        func = polyharmonic
    else:
        func = rbf
    return func(distance(x, node))


def make_monomial(id, vector):
    """ Easy way to keep track of all monomials """
    if id == 0:
        return jnp.array([1.0])
    elif id == 1:
        return vector[0]
    elif id == 2:
        return vector[1]
    elif id == 3:
        return vector[0]**2
    elif id == 4:
        return vector[0]*vector[1]
    elif id == 5:
        return vector[1]**2
    else:
        pass        ## Higher order monomials not yet supported
