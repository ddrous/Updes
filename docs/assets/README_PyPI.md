# 𝕌pdes

𝕌pdes is a general-purpose library for mesh-free PDE simulation and control.
- GitHub project: https://github.com/ddrous/Updes
- Documentation: https://ddrous.github.io/Updes/


## Features
𝕌pdes leverages Radial Basis Functions (RBFs) and JAX to provide the following features:
- User-centric design: no need to re-implement a solver for each new PDE
- Lightning fast mesh-free simulation via Radial Basis Functions
- Robust differentiable simulation via JAX, and portable across CPU, GPU, and TPU
- Support for Dirichlet, Neumann, Robin, and Periodic boundary conditions
- Automatic generation of normals from 2D GMSH meshes

𝕌pdes in incredibly extendable, with additional features added frequently.


## Getting started
The package is available on PyPi. You can install it with
```
pip install updes
```

The example below illustrates how to solve the Laplace equation with Dirichlet and Neumann boundary conditions:
```python
import updes
import jax.numpy as jnp

# Create a mesh-free cloud of points on a unit square
facet_types={"South":"n", "West":"d", "North":"d", "East":"d"}
cloud = updes.SquareCloud(Nx=30, Ny=20, facet_types=facet_types)

# Define the differential operator (left-hand side of the PDE)
def my_diff_operator(x, center, rbf, monomial, fields):
    return updes.nodal_laplacian(x, center, rbf, monomial)

# Define the right-hand side of the PDE
def my_rhs_operator(x, centers, rbf, fields):
    return 0.0

# Set a sin function as the Dirichlet BC on the North, and zero everywhere else
sine = lambda coord: jnp.sin(jnp.pi * coord[0])
zero = lambda coord: 0.0
boundary_conditions = {"South":zero, "West":zero, "North":sine, "East":zero}

# Solve the Laplace equation with a JIT-compiled solver
sol = updes.pde_solver_jit(diff_operator=my_diff_operator, 
                    rhs_operator=my_rhs_operator, 
                    cloud=cloud, 
                    boundary_conditions=boundary_conditions, 
                    rbf=updes.polyharmonic,
                    max_degree=1)

# Visualize the solution
cloud.visualize_field(sol.vals, cmap="jet", projection="3d", title="RBF solution")
```

𝕌pdes can handle much complicated cases with little to no modifications to the code above. Check out further notebooks and scripts in the [documentation](https://ddrous.github.io/Updes/) and the folder [`demos`](./demos)!


## Dependencies
- **Core**: JAX - GMSH - Matplotlib - Seaborn - Scikit-Learn
- **Optional**: PyVista - FFMPEG - QuartoDoc

See the `pyproject.toml` file the specific versions of the dependencies.


## Cite us !
If you use this software, please cite us with the following BibTeX entry:
```
@inproceedings{nzoyem2023comparison,
  title={A comparison of mesh-free differentiable programming and data-driven strategies for optimal control under PDE constraints},
  author={Nzoyem Ngueguin, Roussel Desmond and Barton, David AW and Deakin, Tom},
  booktitle={Proceedings of the SC'23 Workshops of The International Conference on High Performance Computing, Network, Storage, and Analysis},
  pages={21--28},
  year={2023}}
```
