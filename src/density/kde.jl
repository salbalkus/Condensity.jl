"""
    mutable struct KDE <: DensityEstimator

The `KDE` struct represents a Kernel Density Estimator.

# Fields
- `bandwidth::Float64`: The bandwidth parameter for the KDE.
- `kernel`: The kernel function used for the KDE.

"""
mutable struct KDE <: DensityEstimator
    bandwidth::Float64
    kernel
end

KDE(bandwidth::Float64) = KDE(bandwidth, Epanechnikov)

function MMI.fit(model::KDE, verbosity, X, y)

    if DataAPI.ncol(X) > 1
        error("KDE only supports univariate data")
    else
        x = Tables.getcolumn(X, 1)
    end

    ρ_fit = KernelDensity.InterpKDE(KernelDensity.kde(x; bandwidth = model.bandwidth, kernel = model.kernel))

    fitresult = (ρ_fit = ρ_fit,)
    cache = nothing
    report = nothing
    return fitresult, cache, report
end

MMI.fit(model::KDE, verbosity, X) = MMI.fit(model::KDE, verbosity, X, nothing)

function MMI.predict(model::KDE, fitresult, X)

    if DataAPI.ncol(X) > 1
        error("KDE only supports univariate data")
    else
        x = Tables.getcolumn(X, 1)
    end

    dens = pdf(fitresult.ρ_fit, x)

    # In case KernelDensity.jl returns very small negative values
    dens[dens .< 0] .= 0 

    return dens
end