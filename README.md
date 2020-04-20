RiemannianOptim.jl
========

This package is about optimization on specific Riemannian manifolds.

# TO DO
use mul!() see https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/

# Install
Steps to install this package in Julia:
1. start Julia REPL, then press the ```]``` key on your keyboard to enter package command mode.
2. Run the command ```add "https://gitlab.com/RoyCCWang/riemannianoptim"```
3. Run the command ```import RiemannianOptim``` to compile the package into cache storage.

If this repository is modified in the future, one needs to enter package mode in Julia REPL, and run the command ```update RiemannianOptim```.


# Current features
This repo currently solves the RKHS regularization problem subjected to the coefficients being positive. i.e.:

min_{a}. norm(Ka-y,2) + μ*dot(a,K,a), s.t. entries of a are positive.

This is a convex optimization problem, but the solver toolboxes I used to use sometimes give me problems due to the kernel matrix not being numerically positive-definite.

An example script for this is in ```./explore/optim.jl```.

The retractions test scripts are in ```./explore/retract.jl```, for general retractions that I have worked on, and ```./explore/Rp_retract.jl``` for the specific retraction I used for the RKHS regularization problem in this repo.

The trust-region subproblem for this optimization problem is investigated in ```./explore/retract.jl```.

Background material for numerical optimization on Riemannian manifolds are available in [1-2].

# Future direction
General cost functions over the space of positive real numbers, and general cost function over the space of the parameters of a multivariate normal distribution (mean, covariance matrix) will be tested.

Extensions of the non-gradient / heuristic search methods reported in [2] will be explored, if a particularly troublesome cost function with multiple modes is to be implemented.

# References
[1]. P.-A. Absil, R. Mahony, and R. Sepulchre, Optimization Algorithms on Matrix Manifolds. Princeton
University Press, 2007.

[2]. R. C. C. Wang, "Adaptive Kernel Functions and Optimization Over a Space of Rank-One Decompositions", PhD thesis, University of Ottawa, 2017.
