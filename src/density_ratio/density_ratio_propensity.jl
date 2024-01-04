mutable struct DensityRatioPropensity <: ConDensityRatioEstimator
    propensity_scorer::ConDensityEstimator
end

function MMI.fit(model::DensityRatioPropensity, verbosity, X, y)
    propensity_mach = machine(model.propensity_scorer, X, y) |> fit!
    fitresult = (propensity_mach = propensity_mach,)
    cache = nothing
    report = nothing
    return fitresult, cache, report
end

function MMI.predict(model::DensityRatioPropensity, fitresult, Xy_nu, Xy_de)

    g_nu = MMI.predict(fitresult.propensity_mach, Xy_nu)
    g_de = MMI.predict(fitresult.propensity_mach, Xy_de)

    # We need to keep (gn .> 0) for the shifted version of Hn
    #b = maximum([1/length(gn), 0.0001])
    #n = length(g_nu)

    # TODO: Decide which n to use for bounding in the cross-validated setting
    # Should we use full data, the train fold, or the test fold?
    #b = 1 / (sqrt(model.n_bound) * log(model.n_bound)) # from Dudoit and vdL
    bound!(g_de; lower = 0.001)
    Hn = g_nu ./ g_de

    #Hn = (gδinvn ./ bound(gn, lower = b)) .* (gn .> 0) .+ (gδn .== 0)
    #Hn = ifelse.(gn .> 0, gδinvn ./ gn, 0) .+ (gδn .== 0)
    return Hn
end

# TODO: Make these functions more general, able to take any argument
#MLJBase.transform(mach::Machine, LA_nu::AbstractNode, LA_de::AbstractNode) = node((LA_nu, LA_de) -> MLJBase.transform(mach, LA_nu, LA_de), LA_nu, LA_de)
#MLJBase.transform(mach::Machine, LA_nu::DataFrame, LA_de::DataFrame) = MLJBase.transform(mach.model, mach.fitresult, LA_nu, LA_de)
