# Condensity.jl Documentation

*A package for fitting conditional density estimation models using [MLJ](https://alan-turing-institute.github.io/MLJ.jl/dev/).*

## Purpose

This package provides methods for fitting both conditional density estimation (CDE) models and conditional density ratio estimation (CDRE) models in the [MLJ](https://alan-turing-institute.github.io/MLJ.jl/dev/) machine learning framework. 

Typical supervised learning regression models, such as linear regression, random forests, gradient boosters, and neural networks estimate all seek to estimate $E(Y|X)$, the conditional expectation of some target $Y$. Conditional density estimators, on the other hand, seek to estimate $f(Y|X)$, the entire density of $Y$ given $X$ (not just the mean). 

Conditional density estimators (and their ratios) are useful in a variety of settings, including propensity score estimation in causal inference, constructing prediction intervals, domain adaptation in machine learning, importance sampling, and other methods.

## Installation
CausalTables.jl can be installed using the Julia package manager.
From the Julia REPL, type `]` to enter the Pkg REPL mode and run

```
Pkg> add Condensity
```

## Quick Start
Condensity.jl is designed to work with the [MLJ](https://alan-turing-institute.github.io/MLJ.jl/dev/) machine learning framework. A typical MLJ workflow proceeds as follows:

1. Define the model and its hyperparameters.
2. Construct a `machine` that binds the model to the data.
3. Fit the model to the data using the `fit!` method.
4. Make predictions using the `predict` method.

The models defined in Condensity.jl are compatible with many (but not all) MLJ functionalities, including hyperparameter tuning, cross-validation, and model composition.

Here's a simple example of how to fit a conditional density estimator using Condensity.jl:

```jldoctest; output = false, filter = r"(?<=.{17}).*"s
using Condensity
using MLJ
using MLJLinearModels

# generate regression model data
n = 50
X = randn(n)
y = 4 .+ 2 .* X .+ randn(n)

# put data in Tables.jl-compliant format
X = (X = X,)
y = (y = y,)

# define and fit the model
location_model = LinearRegressor()
scale_model = ConstantRegressor()
density_model = KDE(0.001)

r = range(density_model, :bandwidth, lower=0.001, upper=0.5)
lse_model = LocationScaleDensity(location_model, scale_model, density_model, 
                                r, CV(nfolds=10))

# bind the model to the data and fit
lse_mach = machine(lse_model, X, y) |> fit!

# make predictions
data = merge(X, y) # must collect data into a single table
predict(lse_mach, data)

# output
50-element Vector
```

## Where to go from here

- If your goal is to estimate a conditional density of the form $p(Y|X)$, read through the various models in [Conditional Density Estimation](man/density.md).
- If your goal is to estimate a conditional density ratio of the form $p(Y'|X)/p(Y|X)$ (for example, for estimating a *generalized propensity score*), please refer to the models listed in [Conditional Density Ratio Estimation](man/density-ratio.md)
- If you want to simulate data and obtain the "underlying true conditional density" for the purpose of testing your models, please refer to the [Oracle Density Estimation using `CausalTables.jl`](man/oracle.md).
- If you are new to MLJ, please refer to the [MLJ documentation](https://alan-turing-institute.github.io/MLJ.jl/dev/) for more information on how to generally fit models within the MLJ framework.


