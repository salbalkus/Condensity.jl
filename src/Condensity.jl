module Condensity

import MLJModelInterface
import MLJTuning

import MLJBase: train_test_pairs, Machine, machine, fit!, predict, categorical
using Distributions
import Distributions: convolve, pdf

import StatsBase: mean
using KernelDensity
using Tables
using TableTransforms
using DataAPI
using CausalTables
using DensityRatioEstimation

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
include("density_ratio/density_ratio_kliep.jl")
include("density_ratio/density_ratio_kernel.jl")
include("density_ratio/density_ratio_kmm.jl")


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

# density_ratio_kliep.jl
export DensityRatioKLIEP

# density_ratio_kernel.jl
export DensityRatioKernel

# density_ratio_kmm.jl
export DensityRatioKMM

# golden_section_search.jl

# general
export fit
export predict

end
