mutable struct DensityRatioKernel<: ConDensityRatioEstimatorFixed 
    dre
end

function MMI.fit(::DensityRatioKernel, verbosity, Xy_nu, Xy_de)
    fitresult = nothing
    cache = nothing
    report = nothing
    return fitresult, cache, report
end

function MMI.predict(::DensityRatioKernel, fitresult, Xy_nu, Xy_de)

    # Convert data into a vector of tuples for use by DensityRatioEstimation
    x_nu = rowtable(Xy_nu)
    x_de = rowtable(Xy_de)

    return densratio(x_nu, x_de, model.dre)
end