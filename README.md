# LS4LSimulations

Simulations to accompany the LS4L algorithm design described in Gonzalez et al (2026)

These files are the workhorses for the simulation and are specific to a specific algorithm (LS4L/simple/complicated) and data generation setting (extreme1 = no effect modifications, extreme2 = effect modification).

- simulations_alg_LS4L_mechanism_extreme1.py
- simulations_alg_LS4L_mechanism_extreme2.py
- simulations_alg_simple_mechanism_extreme1.py
- simulations_alg_simple_mechanism_extreme2.py
- simulations_alg_complicated_mechanism_extreme1.py
- simulations_alg_complicated_mechanism_extreme2.py

Auxillary files:

- data_generation_seeds.csv provides the seeds that were used to generate each set of patient covariates used in the simulations
- pymc3_seeds.csv provides the random seeds that were passed to pymc3 to fit bayesian logistic regression
- true_beta_eval_extreme1_g838.csv provides the true model coefficients for setting 1: no effect modification
- true_beta_eval_extreme2_g33g.csv provides the truth model coefficients for setting 2: effect modification
- sim_dat_*.csv provides the contextual covariates at each decision point for each person in the simulation. These are a starting point for the algorithms to run the simulation.

Analysis files:

- compute_regret.R compares the simulation results to the truth and creates the average regret figure from the manuscript

To run the simulations, you will need to set up a batch submission script that submitts each .py script 50 times. Each .py file uses a system argument to identify which dataset to use as a starting point for the simulation. The system argument for each setting is a number 1-50 that corresponds to a dataset and random seeds for simulation.

