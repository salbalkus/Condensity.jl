
struct LocationScaleDensity <: ConDensityEstimator
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

    # Example:
    ```@example
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

function fit_factorized_density(model::LocationScaleDensity, verbosity, X, y)
    
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
    
    location_machs = Vector{Machine}(undef, DataAPI.ncol(y))
    scale_machs = Vector{Machine}(undef, DataAPI.ncol(y))
    density_machs = Vector{Machine}(undef, DataAPI.ncol(y))
    min_obs_ε2s = Vector{Float64}(undef, DataAPI.ncol(y))

    Xy = merge_tables(X, y)
    target_names = Tables.columnnames(y)
    for (i, target) in enumerate(target_names)
        y_cur = Tables.getcolumn(y, target)
        Xy_cur = reject(Xy, target_names[1:i]...) |> Tables.columntable
        location_machs[i], scale_machs[i], density_machs[i], min_obs_ε2s[i] = fit_factorized_density(model, verbosity, Xy_cur, y_cur)
    end
    
    fitresult = (location_machs = location_machs,  
                 scale_machs = scale_machs, 
                 density_machs = density_machs, 
                 min_obs_ε2s = min_obs_ε2s, 
                 target_names = target_names
                 )
    cache = nothing
    report = nothing
    return fitresult, cache, report
end

function predict_factorized_density(location_mach, scale_mach, density_mach, min_obs_ε2, X, y)

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

    density = 1.0

    for (i, target) in enumerate(fitresult.target_names)
        y_cur = Tables.getcolumn(Xy, target)
        Xy_cur = reject(Xy, fitresult.target_names[1:i]...) |> Tables.columntable
        density = density .* predict_factorized_density(fitresult.location_machs[i], 
                                              fitresult.scale_machs[i], 
                                              fitresult.density_machs[i], 
                                              fitresult.min_obs_ε2s[i], 
                                              Xy_cur, y_cur)
    end

    return density
end

