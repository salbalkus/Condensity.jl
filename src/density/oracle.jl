
mutable struct OracleDensityEstimator <: ConDensityEstimator
    dgp::DataGeneratingProcess

    @doc raw"""
    mutable struct OracleDensityEstimator <: ConDensityEstimator

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

    # Example:

    ```@example
    using CausalTables

    # Define a DataGeneratingProcess (DGP)
    dgp = DataGeneratingProcess([
    :X => (; O...) -> Normal(0, 1), 
    :y => (; O...) -> Normal(3 * O[:X] + 3, 1)])

    # Construct an OracleDensityEstimator from the DGP
    oracle = OracleDensityEstimator(dgp)

    # Construct data for testing
    data = rand(dgp, 100)
    X = (X = Tables.getcolumn(data, :X),)
    y = (y = Tables.getcolumn(data, :y),)

    # Fit the OracleDensityEstimator to the data
    # Note that `data` contains both the columns of `X` and `y`
    mach = machine(oracle, data, y) |> fit!

    # Get predictions
    pred = predict(mach, data)

    ````

    # Fields:
    - `dgp::DataGeneratingProcess`: The data generating process used by the estimator, from the CausalTables package.

    """
    function OracleDensityEstimator(dgp::DataGeneratingProcess)
        new(dgp)
    end
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

    if length(fitresult.targetnames) == 0
        error("No target columns to be predicted by the OracleDensityEstimator")
    end

    # Iterate through each target column and multiply by its density,
    # conditional on all other variables and subsequent targets
    for (i, targetname) in enumerate(fitresult.targetnames)
        y = Tables.getcolumn(X, targetname)
        density = density .* pdf.(condensity(model.dgp, X, targetname), y)
    end

    return density
end

