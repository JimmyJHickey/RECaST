# RECaST code repository

This directory contains the code to reproduce the simulation results in the manuscript.
"Transfer Learning with Uncertainty Quantification:
Random Effect Calibration of Source to Target (RECaST)"
Please contact Jimmy Hickey at `jhickey@ncsu.edu` for any help or.
questions.

## File descriptions

See workflow.sh for line-by-line Unix code for reproducing the results.


`colmeans_missing.jl`
- Contains a function to take column means of arrays with missing values.

`continuous_nn.jl`
- Contains a function to build a continuous neural network in `Flux`.

`discrete_nn.jl`
- Contains a function to build a discrete neural network in `Flux`.

`expit.jl`
- Contains functions to take the expit and logit of a number.

`glm_regression.jl`
- Contains function to fit linear of logistic regression for source models.

`make_directories.jl`
- Contains functions to build the output directory structure.

`mse.jl`
- Contains a function to calculate the MSE between a vector of true values and a vector of predicted values.

`pipeline.sh`
- `bash` file that runs the whole simulated data analysis.

`posterior_predictive.jl`
- Contains a function to calculate the posterior predictive prediction metrics (RMSE and AUC) as well as the continuous coverage.

`prepare_simulated_data.jl`
- Contains a function that generates simulated data for given sample size and noise.

`recast_binary_coverage.jl`
- Contains a function to calculate the coverage for a binary response.

`roc.jl`
- Contains a function to calculate the AUC and ROC given predicted probabilities.

`run_file.jl`
- Runs the entire pipeline including source models, target only models, and both RECaST models.

`run_wiens_glmnet.jl`
- Contains a wrapper function that runs the `wiens_method_glmnet.R` script.

`theta_S.csv`
- Saved source covariates for data generation.

`train_nn.jl`
- Contains a function to train neural networks using `Flux`.

`wiens_method_glmnet.R`
- Contains a function to penalized logistic regression using the `glmnet` package in `R`.
