# Physics-Informed Neural Networks (PINNs) in JAX 

PINNs are **coordinate networks** trained to be the solution to an
initial-boundary value problem. The optimization problem is based on minimizing
the residuum of the PDE in the domain, as well as residuums on the boundary and
initial conditions. Additional supervised data (i.e., labeled point-wise
solution values) are optional. All involved derivative information can be
readily obtained using the automatic differentiation capabilities of the
underlying deep learning framework (here: JAX).

This repository contains a simple PINN example for the 1d Poisson equation with
homogeneous Dirichlet boundary conditions. Check out the intro part of the
Jupyter notebook for more details. You might also find the accompanying [YouTube
video](https://youtu.be/-dQFrxNuxys) helpful to code along.

**Generally interested in Scientific Machine Learning?** Check out my [YouTube
channel](https://www.youtube.com/@machinelearningsimulation) and the
corresponding [GitHub
repository](https://github.com/Ceyron/machine-learning-and-simulation).
