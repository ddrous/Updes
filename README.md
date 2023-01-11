# __𝕌__pdec

𝕌pdec is a framework for scaling and comparing various control methods for PDEs. __Updec__ stands for 𝕌niversal __P__artial __D__ifferential __E__quations __C__ontroller.


## Features
𝕌pdec support the following methods:
- Direct Numerical simulation
- Differentiable Physics
- Physics-Informed Neural network
- More to come ...

𝕌pdec compares the methods in terms of:
- accuracy and generalization error
- performance and speed
- robustness to perturbation
- More to come ...

𝕌pdec offers parallel-in-time parallelization accros CPUs and GPUs, allowing users to scale and combine the above methods.


## Getting started
Check out the example notebooks and scripts in  the folder [`demos`](./demos)!


## Dependencies
- PhiFlow: for differentiable physics
- Diffrax: for neural ODEs
- PyOMP: for shared-memory parallelization
- MPI4Jax (optional): for distributed-memory parallelization
