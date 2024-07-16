mutable struct DensityRatioKMM <: ConDensityRatioEstimatorFixed 
    σ
    λ
end

DensityRatioKMM(; σ = 1.0, λ = 1.0) = DensityRatioKMM(σ, λ)



function MMI.fit(dre::DensityRatioKMM, verbosity, Xy_nu, Xy_de)
    fitresult = (model = uKMM(σ = dre.σ, λ = dre.λ),)
    cache = nothing
    report = nothing
    return fitresult, cache, report
end

function MMI.predict(model::DensityRatioKMM, fitresult, Xy_nu, Xy_de)

    # Convert data into a vector of tuples for use by DensityRatioEstimation
    x_nu = rowtable(Xy_nu)
    x_de = rowtable(Xy_de)

    # Need to switch the numerator and the denominator, because apparently DensityRatioEstimation.jl works like that...
    return densratio(x_de, x_nu, fitresult.model)
end