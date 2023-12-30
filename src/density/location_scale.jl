### Density ###

using MLJ
using MLJBase
using Distributions
using KernelDensity

mutable struct KDE <: MLJBase.Deterministic
    bandwidth::Float64
    kernel
end

function MLJBase.fit(model::KDE, verbosity, X, y)

    ρ_fit = KernelDensity.InterpKDE(KernelDensity.kde(X; bandwidth = model.bandwidth, kernel = model.kernel))

    fitresult = (ρ_fit = ρ_fit,)
    cache = nothing
    report = nothing
    return fitresult, cache, report
end

MLJBase.fit(model::KDE, verbosity, X) = MLJBase.fit(model::KDE, verbosity, X, nothing)
MLJBase.predict(model::KDE, fitresult, X) = pdf(fitresult.ρ_fit, X)


### Conditional Density ###

struct LocationScaleDensity <: MLJBase.Deterministic
    location_model::MLJ.Supervised
    scale_model::MLJ.Supervised
    r
    resampling::MLJBase.ResamplingStrategy
end

# Implement the necessary methods
function MLJBase.fit(model::LocationScaleDensity, X, y)
    
    yvec = getcolumn(y, 1)
    location_mach = machine(model.location_model, X, yvec) |> fit!
    μ = MLJBase.predict_mean(location_mach, X)
    ε = @. yvec - μ
    ε2 = @. ε^2
    
    min_obs_ε2 = 2*minimum(ε2)
    scale_mach = machine(model.scale_model, X, ε2) |> fit!

    σ2 = MLJBase.predict_mean(scale_mach, X)
    σ2[σ2 .< 0] .= min_obs_ε2

    ε = @. ε / sqrt(σ2)

    density_model = TunedModel(
        # TODO: Pick better default bandwidth?
        model = KDE(1.06 / length(ε)^(0.2), Epanechnikov), # optimal MISE with σ = 1
        tuning = Grid(resolution = 10),
        resampling = resampling,
        measure = negmeanloglik,
        operation = MLJBase.predict,
        range = r
        )
    
    density_mach = machine(density_model, X, zeros(length(X))) |> fit!

    fitresult = (location_mach = location_mach,  scale_mach = scale_mach, density_mach = density_mach, min_obs_ε2 = min_obs_ε2)
    cache = nothing
    report = nothing
    return fitresult, cache, report

end

function MLJBase.predict(model::LocationScaleDensity, fitresult, X)
    μ = MLJBase.predict_mean(location_mach, X)
    σ2 = MLJBase.predict_mean(scale_mach, X)
    σ2[σ2 .< 0] .= fitresult.min_obs_ε2
    rootσ2 = @. sqrt(σ2)

    ε = @. (y - μ) / rootσ2
    return MLJBase.predict(density_mach, ε) ./ rootσ2
end

# Implement Golden Section Search

mutable struct GoldenSectionSearch <: TuningStrategy
	goal::Union{Nothing,Int}
	resolution::Int
	rng::Random.AbstractRNG
end

# Constructor with keywords
GoldenSectionSearch(; goal=nothing, resolution=10, shuffle=true,
	 rng=Random.GLOBAL_RNG) =
	Grid(goal, resolution, MLJBase.shuffle_and_rng(shuffle, rng)...)

# Setup methods
MLJTuning.setup(tuning::MyTuningStrategy, model, range::RangeType1, n, verbosity) = ...
MLJTuning.setup(tuning::MyTuningStrategy, model, range::RangeType2, n, verbosity) = ...

# Generate model batches to evaluate
MLJTuning.models(tuning::MyTuningStrategy, model, history, state, n_remaining, verbosity)
	-> vector_of_models, new_state