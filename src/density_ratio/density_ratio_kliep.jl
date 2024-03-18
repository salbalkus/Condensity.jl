mutable struct DensityRatioKLIEP<: ConDensityRatioEstimatorFixed end

function MMI.fit(model::DensityRatioKLIEP, verbosity, Xy_nu, Xy_de)

    fitresult = (classifier_machine = classifier_mach, n_ratio = n_de / n_nu)
    cache = nothing
    report = nothing
    return fitresult, cache, report
end

function MMI.predict(::DensityRatioClassifier, fitresult, Xy_nu, Xy_de)
    pred = predict(fitresult.classifier_machine, Xy_nu)
    prob_orig = pdf.(pred, false)
    prob_shift = pdf.(pred, true)
    return fitresult.n_ratio .* prob_orig ./ prob_shift
end