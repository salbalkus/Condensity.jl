module Condensity

import MLJModelInterface
import MLJTuning

import MLJBase: train_test_pairs, Machine, machine, fit!, predict, categorical
using Distributions
import Distributions: convolve, pdf

import StatsBase: mean
using KernelDensity
using Tables
using TableOperations
using DataAPI
using CausalTables

const PKG = "Condensity"          # substitute model-providing package name
const MMI = MLJModelInterface
const MT = MLJTuning

# Define abstract data types
abstract type DensityEstimator <: MMI.Deterministic end
abstract type ConDensityEstimator <: MMI.Deterministic end
abstract type ConDensityRatioEstimator <: MMI.Deterministic end
abstract type ConDensityRatioEstimatorAdaptive <: ConDensityRatioEstimator end
abstract type ConDensityRatioEstimatorFixed <: ConDensityRatioEstimator end


### Includes ###
include("utilities.jl")

include("density/oracle.jl")
include("density/kde.jl")
include("density/location_scale.jl")

include("density_ratio/density_ratio_plugin.jl")
include("density_ratio/probabilistic_classifier.jl")



### Exports ###

# utilities.jl
export negmeanloglik
export meanloglik
export reject
export merge_tables
export concat_tables

# oracle_density.jl
export OracleDensityEstimator

# location_scale.jl
export KDE
export LocationScaleDensity

# density_ratio_propensity.jl
export DensityRatioPlugIn

# probabilistic_classifier.jl
export DensityRatioClassifier

# golden_section_search.jl

# general
export fit
export predict

end
