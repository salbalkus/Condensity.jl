module Condense

import MLJModelInterface
import Distributions
import InvertedIndices

const PKG = "Condense"          # substitute model-providing package name
const MMI = MLJModelInterface

# Define abstract data types
abstract type ConDensityEstimator <: MMI.Supervised end
abstract type ConDensityRatioEstimator <: MMI.Supervised end

### Includes ###
include("density/oracle.jl")

### Exports ###

# oracle_density.jl
export OracleDensity

# general
export fit
export predict

end
