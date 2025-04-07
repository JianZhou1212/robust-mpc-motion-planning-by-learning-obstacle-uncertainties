# Robust Predictive Motion Planning by Learning Obstacle Uncertainty
This is the code for the article
```
@article{zhou2025robust,
  title={Robust predictive motion planning by learning obstacle uncertainty},
  author={Zhou, Jian and Gao, Yulong and Johansson, Ola and Olofsson, Bj{\"o}rn and Frisk, Erik},
  journal={IEEE Transactions on Control Systems Technology},
  year={2025},
  pages={},
  doi={10.1109/TCST.2025.3533378},
  publisher={IEEE}
}
```

**Jian Zhou** and **Erik Frisk** are with the Department of Electrical Engineering, Linköping University, Sweden.

**Yulong Gao** is with the Department of Electrical and Electronic
Engineering, Imperial College London, United Kingdom.

**Björn Olofsson** is with the Department of Automatic Control, Lund University, Sweden.

Contact: zjzb1212@qq.com

## Packages for running the code
The programming is by Python. To run the code you need to install the following key packages:

**CasADi**: https://web.casadi.org/

**HSL Solver**: https://licences.stfc.ac.uk/product/coin-hsl

**pytope**: https://pypi.org/project/pytope/

Note: To install the HSL package can be a bit comprehensive, but the solvers just speed up the solutions. You can comment out the places where the HSL solver is used, i.e., "ipopt.linear_solver": "ma57", and just use the default linear solver (`mumps`) of CasADi. 

## Introduction to the files
I. In `1_Reach_Avoid` folder:  
&ensp; (1) `main.ipynb` is the main file for simulation.  
&ensp; (2) `ModelingSVTrue.py` models the SV.  
&ensp; (3) `Planner_P.py` is the Proposed method.  
&ensp; (4) `Planner_D.py` is the Deterministic method.  
&ensp; (5) `Planner_P.py` is the Robust method.  
&ensp; (6) The data is saved in the folder `plot` for reproduction of the results in the paper.

II. In `2_Active_Evasion` folder:  
&ensp; (1) `main.ipynb` is the main file for simulation.  
&ensp; (2) `ModelingSVTrue.py` models the SV.  
&ensp; (3) `Planner_P.py` is the Proposed method for motion planning of the EV.


III. In `3_Highspeed_Overtaking` folder:  
&ensp; (1) `main.ipynb` is the main file for simulation.  
&ensp; (2) `ModelingSVTrue.py` models the SV.  
&ensp; (3) `Planner_P.py` is the Proposed method for motion planning of the EV.
&ensp; (4) `cdf_f.mat` and `cdf_x.mat` save the distribution information of the SV trained from a real-world dataset. This information is used to model the SV in the overtaking scenario.

IV. In `4_Encounter_Scenario_With_Same_Control` folder:  
&ensp; (1) `main.ipynb` is the main file for simulation.  
&ensp; (2) `ModelingSVTrue.py` models the SV.  
&ensp; (3) `Planner_EV.py` is the Proposed method for motion planning of the EV.
&ensp; (3) `Planner_SV.py` is the Proposed method for motion planning of the SV.
&ensp; (4) In this case both the EV and SV use the same strategy.

V. In `5_rounD_Scenario` folder:  
&ensp; (1)`main.ipynb` is the main file for simulation.  
&ensp; (2)`Planner_P.py` is the Proposed method for motion planning of the EV.  
&ensp; (3) `EV_Data.npy`, `SV0_Data.npy`, and `SV1_Data.npy` save the data of the involved vehicles in the rounD dataset scenario.

## Remarks
I. The code for generating the animations has been removed as a result of version compatibility of Python.  

II. The code for hardware experiments can be easily designed based on the published code here.




