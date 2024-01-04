"""
    struct LocationScaleDensity <: ConDensityEstimator

The `LocationScaleDensity` struct represents a conditional density estimator that models the conditional distribution of a target variable given a set of features. It combines a location model, a scale model, and a density model to estimate the conditional density.

## Fields
- `location_model::MMI.Supervised`: The location model used to estimate the conditional mean.
- `scale_model::MMI.Supervised`: The scale model used to estimate the conditional variance.
- `density_model::DensityEstimator`: The density model used to estimate the conditional density.
- `r_density`: An MLJ range object over which `density_model` will be tuned.
- `resampling::MT.ResamplingStrategy`: The resampling strategy used during model fitting.

"""

struct LocationScaleDensity <: ConDensityEstimator
    location_model::MMI.Supervised
    scale_model::MMI.Supervised
    density_model::DensityEstimator
    r_density
    resampling::MT.ResamplingStrategy
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

    Xy_cur = merge_tables(X, y)

    for (i, target) in enumerate(Tables.columnnames(y))
        y_cur = Tables.getcolumn(y, target)
        Xy_cur = reject(Xy_cur, target) |> Tables.columntable
        location_machs[i], scale_machs[i], density_machs[i], min_obs_ε2s[i] = fit_factorized_density(model, verbosity, Xy_cur, y_cur)
    end
    
    fitresult = (location_machs = location_machs,  
                 scale_machs = scale_machs, 
                 density_machs = density_machs, 
                 min_obs_ε2s = min_obs_ε2s, 
                 target_names = Tables.columnnames(y)
                 )
    cache = nothing
    report = nothing
    return fitresult, cache, report
end

function predict_factorized_density(location_mach, scale_mach, density_mach, min_obs_ε2, X, y)

    # Get residual model predictions
    μ = MMI.predict_mean(location_mach, X)
    σ2 = MMI.predict_mean(scale_mach, X)
    σ2[σ2 .< 0] .= min_obs_ε2
    rootσ2 = @. sqrt(σ2)

    # Return density of standardized residual 
    ε = @. (y - μ) / rootσ2
    return MLJBase.predict(density_mach, (ε = ε,)) ./ rootσ2
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

