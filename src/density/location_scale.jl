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

function MMI.fit(model::LocationScaleDensity, verbosity, X, y)
    
    yvec = Tables.getcolumn(y, 1) # convert Table to Vector

    # Fit the location model
    location_mach = machine(model.location_model, X, yvec) |> fit!
    μ = MMI.predict_mean(location_mach, X)

    # Fit the scale model
    ε = @. yvec - μ
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

    fitresult = (location_mach = location_mach,  
                 scale_mach = scale_mach, 
                 density_mach = density_mach, 
                 min_obs_ε2 = min_obs_ε2, 
                 target_name = Tables.columnnames(y)[1]
                 )
    cache = nothing
    report = (ε = ε,
                )
    return fitresult, cache, report

end

function MMI.predict(model::LocationScaleDensity, fitresult, Xy)

    # Split table into vectors
    X = reject(Xy, fitresult.target_name) |> Tables.columntable
    y = Tables.getcolumn(Xy, fitresult.target_name)

    # Get residual model predictions
    μ = MMI.predict_mean(fitresult.location_mach, X)
    σ2 = MMI.predict_mean(fitresult.scale_mach, X)
    σ2[σ2 .< 0] .= fitresult.min_obs_ε2
    rootσ2 = @. sqrt(σ2)

    # Return density of standardized residual 
    ε = @. (y - μ) / rootσ2
    return MLJBase.predict(fitresult.density_mach, (ε = ε,)) ./ rootσ2
end

