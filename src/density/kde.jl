
mutable struct KDE <: DensityEstimator
    bandwidth::Float64
    kernel
    
    @doc raw"""
        KDE(bandwidth::Float64, kernel::Type)

    Wraps the Kernel Density Estimator from [KernelDensity.jl](https://github.com/JuliaStats/KernelDensity.jl) as an MLJ object. 
    This model defines a density estimator that does not condition on any features, and is used to estimate the marginal density of a target variable.
    It is mostly used as a component of a `LocationScaleDensity` model, though it can be used on its own.

    For more information on Kernel Density Estimation, [see here](https://en.wikipedia.org/wiki/Kernel_density_estimation).

    # Arguments
    - `bandwidth::Float64`: The bandwidth parameter for the KDE.
    - `kernel`: The kernel function used for the KDE. Default: Epanechnikov.

    # Example
    ```jldoctest; output = false, filter = r"(?<=.{17}).*"s
    using Condensity
    using MLJ

    # Generate data in Tables.jl-compliant format
    X = (X = randn(10),)

    # define and fit the model
    kde = KDE(1.0, Epanechnikov)
    kde_mach = machine(kde, X) |> fit!
    predict(kde_mach, X)

    # output
    10-element Vector
    ```
    """
    function KDE(bandwidth::Float64, kernel::Type)
        if bandwidth <= 0
            error("Bandwidth must be positive")
        end
        new(bandwidth, kernel)
    end
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