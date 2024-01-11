

"""
    struct OracleDensityEstimator <: Unsupervised

A struct representing an Oracle Density Estimator model.

# Fields
- `dgp::DataGeneratingProcess`: The data generating process used by the estimator.

"""
struct OracleDensityEstimator <: ConDensityEstimator
    dgp::DataGeneratingProcess
end

"""
    MMI.fit!(model::OracleDensityEstimator, verbosity, X, y)

Fit the OracleDensityEstimator model to the given data.

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
    fitresult = (; targetnames = Tables.columnnames(y),)
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
function MMI.predict(model::OracleDensityEstimator, fitresult, X)
    
    # initialize the density
    density = 1.0

    # Iterate through each target column and multiply by its density,
    # conditional on all other variables and subsequent targets
    for (i, targetname) in enumerate(fitresult.targetnames)
        # BUG: reject function not doing what we want here
        y = Tables.getcolumn(X, targetname)
        density = density .* pdf.(condensity(model.dgp, X, targetname), y)
    end

    return density
end

