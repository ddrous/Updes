import jax.numpy as jnp

def multiquadric(r):
    eps = 1
    return jnp.sqrt(1 + (eps*r)**2)

def polyharmonic(r):
    a = 1
    return r**(2*a+1)

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
