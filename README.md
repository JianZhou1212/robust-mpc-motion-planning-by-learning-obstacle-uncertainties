# Robust Predictive Motion Planning by Learning Obstacle Uncertainty
This is the Python code for the article
```
@article{zhou2024robust,
  title={Robust Predictive Motion Planning by Learning Obstacle Uncertainty},
  author={Jian Zhou, Yulong Gao, Ola Johansson, Bj\"orn Olofsson, and Erik Frisk},
  year={2024},
  pages={},
  doi={ }} 
```

The authors are from the Department of Electrical Engineering, Linköping University, Sweden, Department of Electrical and Electronic
Engineering, Imperial College London, United Kingdom, and the Department of Automatic Control, Lund University, Sweden.

## Packages for running the code
To run the code you need to install the following key packages:

**CasADi**: https://web.casadi.org/

**HSL Solver**: https://licences.stfc.ac.uk/product/coin-hsl

**pytope**: https://pypi.org/project/pytope/

Note: Installing the HSL package can be a bit comprehensive, but the solvers just speed up the solutions. You can comment out the places where the HSL solver is used, i.e., "ipopt.linear_solver": "ma57", and just use the default linear solver of CasADi. 

## Introduction to the files
(1) `main.ipynb` is the main file for simulation.

(2) `ModelingSVTrue.py` defines the nonlinear MPC controller for simulating the SV.

(3) `Planner_D.py` defines the deterministic MPC (DMPC) planner.

(4) `Planner_R.py` defines the robust MPC (RMPC) planner.

(5) `Planner_N.py` defines the proposed MPC planner.

The code for the other case studies will be published soon, while the other case studies are implemented by the same methods.



