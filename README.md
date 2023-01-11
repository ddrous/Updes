# Neural Fluids

__Updec__ is a framework for scaling and comparing various control methods for PDEs. __Updec__ stands for Universal($\mathbb{U}$) Partial Differential Equations (__PDE__) Controller (__C__)



## Features

__Updec__ support the following methods:
- Direct Numerical simulation
- Differentiable Physics
- Physics-Informed Neural network

__Updec__ offers parallel-in-time parallelization accros CPUs and GPUs, allowing users to combine the above methods. 

__Updec__ scales and compares the methods in terms of:
- accuracy and generalization error
- parallel-in-time performance
- robustness to perturbation


## Getting started
Check out the example notebooks and scripts in [`demos`](./democs/)!


## Dependencies
- Phiflow: for differentiable physics
- Diffrax: for neural ODEs
- PyOMP: for shared-memory parallelization
- MPI4Jax (optional): for distributed-memory parallelization


## Installation
Installing MPI4Jax can be tricky. You might find it helpful to consider the following steps when installing MPI4JAX in a Conda environment:
- Make sure you have OpenMPI installed: on MAC, `brew install open-mpi`, or on Linux ...
- If you prefer the MPICH implementation: 
    - Don't install `mpich-mpicc` with Conda. If need be, uninstall it: `conda uninstall -c conda-forge mpich-mpicc`
    - Install MPICH system-wide (e.g. on Ubuntu): `sudo apt install mpich`
    - Remember to set the `MPICC` path to the system's:  `env MPICC=/usr/bin/mpicc`
- On Linux WSL, install `x86_64-linux-gnu` on Conda: `conda install -c anaconda gcc_linux-64` (for some reason, MPI4JAX doesn't compile with the default)
- Only then, you install and test `mpi4jax`
