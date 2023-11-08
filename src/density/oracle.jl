"""
    OracleDensity <: ConDensityEstimator

A mutable structure representing an oracle density estimator.
"""
mutable struct OracleDensity <: ConDensityEstimator
    density::Function
end

"""
    MMI.fit(model::OracleDensity, verbosity::Int, X, y)

Fit the model to the data. Returns the fit result, cache, and report.
"""
function MMI.fit(model::OracleDensity, verbosity::Int, X, y)
    fitresult = (; treatment_names = propertynames(y),)
    cache = nothing
    report = nothing
    return fitresult, cache, report
end

"""
    MMI.predict(model::OracleDensity, fitresult, Xnew)

Predict the propensity from the oracle assigned to the fitted treatment label.
"""
function MMI.predict(model::OracleDensity, fitresult, Xnew)
    y = MMI.matrix(Xnew[fitresult.treatment_names])
    X = MMI.matrix(Xnew[InvertedIndices.Not(fitresult.treatment_names)])

    return Distributions.pdf.(model.density(X), y)
end