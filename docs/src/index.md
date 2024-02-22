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

