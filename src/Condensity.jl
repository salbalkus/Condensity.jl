module Condensity

import MLJModelInterface
import MLJTuning
using MLJBase

using Distributions
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

### Includes ###
include("utilities.jl")
include("density/oracle.jl")
include("density/kde.jl")
include("density/location_scale.jl")


### Exports ###

# utilities.jl
export negmeanloglik
export meanloglik
export reject

# oracle_density.jl
export OracleDensityEstimator

# location_scale.jl
export KDE
export LocationScaleDensity

# golden_section_search.jl

# general
export fit
export predict

end
