
@doc raw"""
OracleDensityEstimator(dgp::DataGeneratingProcess)

An "oracle" density estimator is a model that returns the *true* conditional density of a given variable. 
Oracles are typically used for simulation and testing of methods that estimate conditional densities as nuisance parameters, in order to evaluate their performance when the underlying truth is known.
They can also be used to compute a conditional density when the true underlying distribution of the data is known.

Constructing an OracleDensityEstimator requires defining `DataGeneratingProcess` from the CausalTables package. 
This object describes the true data generating process that is used to compute the true conditional density.
Check out [CausalTables](https://salbalkus.github.io/CausalTables.jl/dev/) for more information on how to define a `DataGeneratingProcess`, 
and how to use it to conveniently generate data for statistical simulations.

# Important Notes:
- When fitting the model, the data on which to condition is provided as the first input `X`, and the target variable is provided as the second input `y`.
- Unlike other MLJ models, both `X` and `y` must be in Table form when input to `machine`.
- The predict method requires input as a Table containing the column names of both `X` and `y` that were provided during fitting.

# Arguments
- `dgp::DataGeneratingProcess`: The data generating process used by the estimator, from the CausalTables package.

"""

mutable struct OracleDensityEstimator <: ConDensityEstimator
    scm::StructuralCausalModel
end

"""
    MMI.fit!(model::OracleDensityEstimator, verbosity, X, y)

Fit the OracleDensityEstimator model to the given data. Assumes y is a Table with only one column.

# Arguments
- `model::OracleDensityEstimator`: The OracleDensityEstimator model to fit.
- `verbosity`: The verbosity level of the fitting process.
- `X`: The input data Table.
- `y`: The target data Table.

# Returns
- `fitresult`: A `NamedTuple` containing the fit result.
- `cache`: The cache object.
- `report`: The report object.

"""
function MMI.fit(model::OracleDensityEstimator, verbosity, X, y)
    fitresult = (; targetname = Tables.columnnames(y)[1],)
    cache = nothing
    report = nothing
    return fitresult, cache, report
end

"""
    MMI.predict(model::OracleDensityEstimator, verbosity, X)

Predict the density using the OracleDensityEstimator model.

# Arguments
- `model::OracleDensityEstimator`: The OracleDensityEstimator model to use for transformation.
- `verbosity`: The verbosity level.
- `X`: The input data table.

# Returns
- `density`: A `Vector` of density values.

"""
function MMI.predict(model::OracleDensityEstimator, fitresult, Xy)
    y = Tables.getcolumn(Xy, fitresult.targetname)
    return pdf.(condensity(model.scm, Xy, fitresult.targetname), y)
end

