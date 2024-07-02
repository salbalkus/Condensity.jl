mutable struct DensityRatioKLIEP<: ConDensityRatioEstimatorFixed 
    σ::Array{Float64,1}
    b::Array{Int,1}
end

function MMI.fit(model::DensityRatioKLIEP, verbosity, Xy_de, Xy_nu)

    # Convert data into a vector of tuples for use by DensityRatioEstimation
    x_nu = Tables.rowtable(Xy_nu)
    x_de = Tables.rowtable(Xy_de)

    # Optimize hyperparameters for the KLIEP method
    dre = DensityRatioEstimation.fit(KLIEP, x_nu, x_de, LCV((σ=model.σ,b=model.b)))

    fitresult = (; dre = dre,)
    cache = nothing
    report = nothing
    return fitresult, cache, report
end

function MMI.predict(::DensityRatioKLIEP, fitresult, Xy_de, Xy_nu)

    # Convert data into a vector of tuples for use by DensityRatioEstimation
    x_nu = rowtable(Xy_nu)
    x_de = rowtable(Xy_de)

    return densratio(x_nu, x_de, fitresult.dre)
end