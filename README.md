# 𝕌pdec

𝕌pdec is a Python library for mesh-free PDE simulation and control. __𝕌pdec__ stands for Universal(__𝕌__) Partial Differential Equations (__PDE__) Controller(__C__).


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
- Review the [specifications for JOSS](https://joss.readthedocs.io/en/latest/submitting.html)
- Setup automated tests via GitHub actions CI/CD
- Add comments and generate Documentation (like [Equinox](https://docs.kidger.site/equinox/))
- Code more demos and animations of PDEs: 
    - diffusion and other cool linear equation 
    - the Navier-Stokes in a channel with a step, 
    - etc.
- Publish to social media (and to my website)
- Research dynamic point clouds for time-dependent problems
- Parallelise time-dependent problems in time
- Add support for USD files


## Dependencies
- JAX
- GMSH
- PyVista


## Cite us
If you use this software, please cite us 
