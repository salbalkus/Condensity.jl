
mutable struct OracleDensity <: ConDensityEstimator
    density::Function
end

function MMI.fit(model::OracleDensity, verbosity::Int, X, y)
    fitresult = (; treatment_names = propertynames(y),)
    cache = nothing
    report = nothing
    return fitresult, cache, report
end

function MMI.predict(model::OracleDensity, fitresult, Xnew)
    y = MMI.matrix(Xnew[fitresult.treatment_names])
    X = MMI.matrix(Xnew[InvertedIndices.Not(fitresult.treatment_names)])

    # Get the propensity from the oracle assigned to the fitted treatment label
    return Distributions.pdf.(model.density(X), y)
end

