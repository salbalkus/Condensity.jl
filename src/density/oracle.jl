

"""
    struct OracleDensityEstimator <: Unsupervised

A struct representing an Oracle Density Estimator model.

# Fields
- `dgp::DataGeneratingProcess`: The data generating process used by the estimator.

"""
struct OracleDensityEstimator <: Unsupervised
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
function MMI.fit!(model::OracleDensityEstimator, verbosity, X, y)
    fitresult = (; targetnames = columnnames(y),)
    cache = nothing
    report = nothing
    return fitresult, cache, report
end

"""
    MMI.transform(model::OracleDensityEstimator, verbosity, X)

Transform the input data using the OracleDensityEstimator model.

# Arguments
- `model::OracleDensityEstimator`: The OracleDensityEstimator model to use for transformation.
- `verbosity`: The verbosity level of the transformation process.
- `X`: The input data table.

# Returns
- `density`: A `Vector` of transformed density values.

"""
function MMI.transform(model::OracleDensityEstimator, verbosity, X)
    
    # initialize the density
    density = ones(nrow(X))

    # Iterate through each target column and multiply by its density,
    # conditional on all other variables and subsequent targets
    for (i, targetname) in enumerate(fitresult.targetnames)
        Xsub = select(X, Not(targetnames[1:i]))
        density = @. density * pdf(condensity(model.dgp, Xsub, targetname))
    end

    return density
end

