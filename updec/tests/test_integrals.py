# %%
import jax
# jax.config.update("jax_debug_nans", True)

import random
from functools import partial
from updec import *
"A unit test that checks if the gradient of a constant field is zero"

seed = random.randint(0,100)
# seed = 12

# EXPERIMENET_ID = random_name()
DATAFOLDER = "updec/tests/data/"


# %%
# facet_types = {"North":"n", "South":"n", "East":"n", "West":"n"}
facet_types = {"North":"d", "South":"d", "East":"d", "West":"d"}

# noise_key = jax.random.PRNGKey(seed)
noise_key = None
size = 12
cloud = SquareCloud(Nx=size, Ny=size, facet_types=facet_types, support_size="max", noise_key=noise_key)

cloud.visualize_cloud(figsize=(6,5), s=12, title="Cloud for testing");



RBF = partial(polyharmonic, a=5)
# RBF = multiquadric
MAX_DEGREE = 3


## Field s is x**2 + y**2
xy = cloud.sorted_nodes
# s = xy[:,0]**2 + xy[:,1]**2

# s = (xy[:,0]**2 - xy[:,1]**2) / (xy[:,0]**2 + xy[:,1]**2)**2 ## https://math.stackexchange.com/questions/467663/double-integral-question-on-unit-square
# s = jnp.sqrt(xy[:,0]**2 + xy[:,1]**2) ## https://mathworld.wolfram.com/UnitSquareIntegral.html#:~:text=which%20give%20the%20average%20distances,and%20to%20the%20center%2C%20respectively.

# s = 1 / (xy[:,0]**2 + xy[:,1]**2 + 1)
s = xy[:,0]**2/ (1+xy[:,1]**2)

## Visualise s
cloud.visualize_field(s, figsize=(6,5), title="Field s");


## Recover s from a pde solve


# print(s)

## Compute coefficients ?
s_coeffs = get_field_coefficients(s, cloud, RBF, MAX_DEGREE)
# extended_coeffs = jnp.concatenate([s, jnp.zeros((3,))])
# print(extended_coeffs)
# print(jnp.isinf(assemble_invert_A(cloud, RBF, 3)))
# # print(assemble_invert_A(cloud, RBF, 3) @ extended_coeffs)

## Original field
print("Original field\n", s)
##
print("Reconstructed field")
s_ = value_vec(cloud.sorted_nodes, s_coeffs, cloud.sorted_nodes, RBF)
print(s_)
cloud.visualize_field(s_, figsize=(6,5), title="Field s reconstructed from coefficients");

## Print the error
error = np.mean(np.abs(s - s_))
print("Mean error in the reconstruction: ", error)

## Integral
# val = integrate_field(s, cloud, RBF, MAX_DEGREE)
val = integrate_field(s_coeffs, cloud, RBF, MAX_DEGREE)

print("Integral over the unit square is: ", val)
# print("Expected value is: ", (np.sqrt(2) + (1/np.sinh(1)))/ 3)
print("Expected value is: ", np.pi / 12)
# print("Expected value is: ", 0.63951)

def test_integral():
    assert np.abs(val - (np.pi / 12)) < 1e-1


# %%


# ## Do a concergence analysis, with different sizes of the cloud
# sizes = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18, 20, 25]
# errors = []
# for size in sizes:
#     cloud = SquareCloud(Nx=size, Ny=size, facet_types=facet_types, support_size="max", noise_key=noise_key)
#     xy = cloud.sorted_nodes
#     # s = jnp.sqrt(xy[:,0]**2 + xy[:,1]**2)
#     s = xy[:,0]**2/ (1+xy[:,1]**2)
#     s_coeffs = get_field_coefficients(s, cloud, RBF, MAX_DEGREE)
#     val = integrate_field(s_coeffs, cloud, RBF, MAX_DEGREE)
#     # val = integrate_field(s, cloud, RBF, MAX_DEGREE)
#     # errors.append(np.abs(val - (np.sqrt(2) + (1/np.sinh(1)))/ 3))
#     errors.append(np.abs(val - (np.pi / 12)))

# # %%
# plt.plot(sizes, errors)
# plt.xlabel("Size of the cloud")
# plt.ylabel("Error in the integral")
# plt.title("Convergence of the integral over the unit square")
# plt.yscale("log")
# plt.show()




# %%

# RBF = partial(polyharmonic, a=5)
# # RBF = multiquadric
# # RBF = gaussian
# MAX_DEGREE = 2

# ## Do a concergence analysis, with different sizes of the cloud
# # sizes = [5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30, 35, 40, 50]
# sizes = [5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30]
# # noise_key = jax.random.PRNGKey(seed)
# errors = []
# for size in sizes:
#     cloud = SquareCloud(Nx=size, Ny=size, facet_types=facet_types, support_size="max", noise_key=noise_key)
#     xy = cloud.sorted_nodes
#     # s = jnp.sqrt(xy[:,0]**2 + xy[:,1]**2)
#     s = xy[:,0]**2/ (1+xy[:,1]**2)
#     s_coeffs = get_field_coefficients(s, cloud, RBF, MAX_DEGREE)
#     s_ = value_vec(cloud.sorted_nodes, s_coeffs, cloud.sorted_nodes, RBF)
#     errors.append(np.mean(np.abs(s - s_)))

# # %%

# print(errors)

# plt.plot(sizes, errors)
# plt.xlabel("Size of the cloud")
# plt.ylabel("Error in the reconstruction")
# plt.title("Convergence of the reconstructed field from its coefficients")
# plt.yscale("log")
# plt.show()
# # %%
