
mutable struct DensityRatioPlugIn <: ConDensityRatioEstimator
    density_estimator::ConDensityEstimator
    truncate::Bool

    @doc raw"""
        DensityRatioPlugIn(density_estimator::ConDensityEstimator, truncate::Bool = false)

    This model estimates a conditional density ratio by directly fitting a conditional density model. DensityRatioPlugIn wraps the estimator and uses the fitted model to compute both the numerator and the denominator of the ratio. 
    That is, the model performs the following steps:

    1. Estimate ``p_n(Y|X)`` using a conditional density estimator.
    2. Directly compute the density ratio ``H_n(X) = \hat{p}_n(Y'|X) / \hat{p}_n(Y|X)`` by plugging in predictions from the fitted ``\hat{p}_n(Y|X)``.

    # Arguments
    - `density_estimator::ConDensityEstimator`: The density estimator used to compute the numerator and denominator of the density ratio.
    - `truncate::Bool`: Whether to truncate the density estimates to avoid numerical instability. Default: `false`.

    # Example
    ```jldoctest; output = false, filter = r"(?<=.{17}).*"s
    using Condensity
    using MLJ
    using MLJLinearModels

    # generate regression model data
    n = 50
    X = randn(n)
    Y = 4 .+ 2 .* X .+ randn(n)

    # put data in Tables.jl-compliant format
    X = (X = X,)
    y = (y = Y,)

    # define the conditional density model
    location_model = LinearRegressor()
    scale_model = ConstantRegressor()
    density_model = KDE(0.001)

    r = range(density_model, :bandwidth, lower=0.001, upper=0.5)
    lse_model = LocationScaleDensity(location_model, scale_model, density_model, 
                                    r, CV(nfolds=10))
    
    
    # plug in conditional density model and fit
    dr_model = DensityRatioPlugIn(lse_model)
    dr_mach = machine(dr_model, X, y) |> fit!

    # collect X and y for prediction
    denominator = merge(X, y) 
    numerator = merge(X, (y = Y .- 0.1,))

    predict(dr_mach, numerator, denominator)
    

    # output
    50-element Vector
    ```
    """
    function DensityRatioPlugIn(density_estimator::ConDensityEstimator, truncate::Bool = false)
        new(density_estimator, truncate)
    end
end

function MMI.fit(model::DensityRatioPlugIn, verbosity, X, y)
    density_mach = machine(model.density_estimator, X, y) |> fit!
    fitresult = (density_mach = density_mach,)
    cache = nothing
    report = nothing
    return fitresult, cache, report
end

function MMI.predict(model::DensityRatioPlugIn, fitresult, Xy_nu, Xy_de)

    g_nu = MMI.predict(fitresult.density_mach, Xy_nu)
    g_de = MMI.predict(fitresult.density_mach, Xy_de)

    # We need to keep (gn .> 0) for the shifted version of Hn
    #b = maximum([1/length(gn), 0.0001])
    #n = length(g_nu)

    # TODO: Decide which n to use for bounding in the cross-validated setting
    # Should we use full data, the train fold, or the test fold?
    #b = 1 / (sqrt(model.n_bound) * log(model.n_bound)) # from Dudoit and vdL
    
    if model.truncate
        bound!(g_de; lower = 0.001)
    end
    Hn = g_nu ./ g_de

    #Hn = (gδinvn ./ bound(gn, lower = b)) .* (gn .> 0) .+ (gδn .== 0)
    #Hn = ifelse.(gn .> 0, gδinvn ./ gn, 0) .+ (gδn .== 0)
    return Hn
end