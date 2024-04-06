# 𝕌pdec

𝕌pdec is a Python library for mesh-free PDE simulation and control. __𝕌pdec__ stands for Universal(__𝕌__) Partial Differential Equations (__PDE__) Controller(__C__).

There is no faster way to test your PDE than to use Updec !

It workes well for:
- 2D PDEs
- Linear
- Scalar
- Non evolving bcs

## Features
𝕌pdec is equipped with the following features:
- Lightning fast mesh-free simulation via Radial Basis Functions
- Automatic handling of all boundary conditions
- Robust differentiable simulation via JAX
- Automatic generation of normals from 2D GMSH meshes
- Performance portability to CPU, GPU, and TPU
Features will be added on-demand (see list of To-Dos below)


## Getting started
The package is available on PyPi. You can install it via:
```
pip install Updec
```
Check out the example notebooks and scripts in the folder [`demos`](./demos)!


## To-Dos
- Logo, licence, contributors guide, and developer documentation
- Better introductory examples and user documentation for outreach
    - Non-linear and multi-dimensional PDEs
    - Adjoint schemes for fluid flows
- Better cloud points generation with accurate geometry and normals: 
    - USD files
    - Gmsh tutorial
- Add support for 3D radial basis functions
- Improve accuracy for integration with diffrax


## Dependencies
### Core
    - JAX
    - GMSH
    - Matplotlib

### Optional
    - PyVista
    - FFMPEG
    - QuartoDoc

## Cite us
If you use this software, please cite us: SC2023 reference
