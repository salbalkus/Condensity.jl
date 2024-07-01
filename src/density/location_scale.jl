
mutable struct LocationScaleDensity <: ConDensityEstimator
    location_model::MMI.Supervised
    scale_model::MMI.Supervised
    density_model::DensityEstimator
    r_density
    resampling::MT.ResamplingStrategy

    @doc raw"""
        LocationScaleDensity(location_model::MMI.Supervised, 
                             scale_model::MMI.Supervised, 
                             density_model::DensityEstimator, 
                             r_density, 
                             resampling::MT.ResamplingStrategy
                             )

    A conditional density estimator that models the distribution of a target variable given a set of features. 
    Crucially, this model assumes that the conditional distribution depends only on the covariates through the distribution's first two moments: that is, the mean and variance.
    It works by:

    1. Fitting a conditional mean (location) model and a conditional variance (scale) model for the target variable ``Y`` given the features ``X``.
    2. Standardizing the target ``Y`` by subtracting the conditional mean and dividing by the square root of the conditional variance.
    3. Performing kernel density estimation on the standardized residuals to estimate the conditional density.

    Mathematically, this can be represented as the following steps:

    1. Estimate ``\mu(X) = E[Y|X]`` using a supervised MLJ model.
    2. Estimate ``\sigma^2(X) = Var[Y|X]`` using a supervised MLJ model.
    3. Estimate density ``\rho(X)`` of ``(Y - \hat{\mu}(X))^2 / \hat{\sigma}^2(X)`` using kernel smoothing.
    4. Estimate conditional density as ``p_n(Y|X) = \hat{\rho}((Y - \hat{\mu}(X))) / \hat{\sigma}(X)``

    Hence, constructing a LocationScaleDensity model requires defining three sub-models: a location model, a scale model, and a density model. In addition, a range object is required to tune the density model, and a resampling strategy is required to fit the sub-models.

    # Example
    ```jldoctest; output = false, filter = r"(?<=.{17}).*"s
    using Condensity
    using MLJ
    using MLJLinearModels

    # generate regression model data
    n = 50
    X = randn(n)
    y = 4 .+ 2 .* X .+ randn(n)

    # put data in Tables.jl-compliant format
    X = (X = X,)
    y = (y = y,)

    # define and fit the model
    location_model = LinearRegressor()
    scale_model = ConstantRegressor()
    density_model = KDE(0.001)

    r = range(density_model, :bandwidth, lower=0.001, upper=0.5)
    lse_model = LocationScaleDensity(location_model, scale_model, density_model, 
                                    r, CV(nfolds=10))
    lse_mach = machine(lse_model, X, y) |> fit!

    # Get predictions
    data = merge(X, y) # must collect data into a single table
    predict(lse_mach, data)

    # output
    50-element Vector
    ```

    ## Arguments
    - `location_model::MMI.Supervised`: The location model used to estimate the conditional mean.
    - `scale_model::MMI.Supervised`: The scale model used to estimate the conditional variance.
    - `density_model::DensityEstimator`: The density model used to estimate the conditional density.
    - `r_density`: An MLJ range object over which `density_model` will be tuned.
    - `resampling::MT.ResamplingStrategy`: The resampling strategy used during model fitting.

    """
    function LocationScaleDensity(location_model::MMI.Supervised, scale_model::MMI.Supervised, density_model::DensityEstimator, r_density, resampling::MT.ResamplingStrategy)
        new(location_model, scale_model, density_model, r_density, resampling)
    end
end

function fit_density(model::LocationScaleDensity, verbosity, X, y)
    
    # Fit the location model
    location_mach = machine(model.location_model, X, y) |> fit!
    μ = MMI.predict_mean(location_mach, X)

    # Fit the scale model
    ε = @. y - μ
    ε2 = @. ε^2
    min_obs_ε2 = 2*minimum(ε2)
    scale_mach = machine(model.scale_model, X, ε2) |> fit!

    # Fit the density model
    σ2 = MMI.predict_mean(scale_mach, X)
    σ2[σ2 .< 0] .= min_obs_ε2
    ε = @. ε / sqrt(σ2)

    tuned_density_model = MT.TunedModel(
        # TODO: Pick better default bandwidth?
        model = model.density_model,
        # TODO: Choose better MT.TuningStrategy
        tuning = MT.Grid(resolution = 100),
        resampling = model.resampling,
        measure = negmeanloglik,
        operation = MMI.predict,
        range = model.r_density
        )
    
    density_mach = machine(tuned_density_model, (ε = ε,), zeros(length(ε))) |> fit!

    return location_mach, scale_mach, density_mach, min_obs_ε2
end

function MMI.fit(model::LocationScaleDensity, verbosity, X, y)

    targetname = Tables.columnnames(y)[1]
    y_vec = Tables.getcolumn(y, targetname)
    location_mach, scale_mach, density_mach, min_obs_ε2 = fit_density(model, verbosity, X, y_vec)
    
    fitresult = (location_mach = location_mach,  
                 scale_mach = scale_mach, 
                 density_mach = density_mach, 
                 min_obs_ε2 = min_obs_ε2, 
                 targetname = targetname
                 )
    cache = nothing
    report = nothing
    return fitresult, cache, report
end

function predict_density(location_mach, scale_mach, density_mach, min_obs_ε2, X, y)

    # Get residual model predictions
    μ = MMI.predict_mean(location_mach, X)
    σ2 = MMI.predict_mean(scale_mach, X)
    σ2[σ2 .<= 0] .= min_obs_ε2
    rootσ2 = @. sqrt(σ2)

    # Return density of standardized residual 
    ε = @. (y - μ) / rootσ2
    return MMI.predict(density_mach, (ε = ε,)) ./ rootσ2
end

function MMI.predict(model::LocationScaleDensity, fitresult, Xy) 

    # split off y
    y_cur = Tables.getcolumn(Xy, fitresult.targetname)
    Xy_cur = Xy |> TableTransforms.Reject(fitresult.targetname)
    # compute density
    density = predict_density(fitresult.location_mach, 
                                              fitresult.scale_mach, 
                                              fitresult.density_mach, 
                                              fitresult.min_obs_ε2, 
                                              Xy_cur, y_cur)
    return density
end

