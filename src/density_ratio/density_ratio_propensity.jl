mutable struct DensityRatioPropensity <: Unsupervised
    propensity_scorer
end

function MLJBase.fit(model::DensityRatioPropensity, verbosity, L, A)
    int_treatment_names = propertynames(A)
    LA = hcat(L, A)
    mach_propensity_vec = Vector{Machine}(undef, length(int_treatment_names))
    mach_propensity_vec[1] = machine(model.propensity_scorer, LA, int_treatment_names[1]) |> fit!

    for i in 1:(length(int_treatment_names)-1)
        mach_propensity_vec[i+1] = machine(model.propensity_scorer, DataFrames.select(LA, Not(int_treatment_names[1:i])), int_treatment_names[i+1]) |> fit!
    end

    fitresult = (int_treatment_names = int_treatment_names, mach_propensity_vec = mach_propensity_vec)
    cache = nothing
    report = nothing
    return fitresult, cache, report
end

function MLJBase.transform(model::DensityRatioPropensity, fitresult, LA_nu, LA_de)

    g_nu = MLJBase.transform(fitresult.mach_propensity_vec[1], LA_nu)
    g_de = MLJBase.transform(fitresult.mach_propensity_vec[1], LA_de)

    for i in 1:(length(fitresult.int_treatment_names)-1)

        g_nu = g_nu .* MLJBase.transform(fitresult.mach_propensity_vec[i+1], DataFrames.select(LA_nu, Not(fitresult.int_treatment_names[1:i])))
        g_de = g_de .* MLJBase.transform(fitresult.mach_propensity_vec[i+1], DataFrames.select(LA_de, Not(fitresult.int_treatment_names[1:i])))

    end

    # We need to keep (gn .> 0) for the TMLE shifted version of Hn
    #b = maximum([1/length(gn), 0.0001])
    #n = length(g_nu)

    # TODO: Decide which n to use for bounding in the cross-validated setting
    # Should we use full data, the train fold, or the test fold?
    #b = 1 / (sqrt(model.n_bound) * log(model.n_bound)) # from Dudoit and vdL
    Hn = g_nu ./ bound(g_de; lower = 0.001)

    #Hn = (gδinvn ./ bound(gn, lower = b)) .* (gn .> 0) .+ (gδn .== 0)
    #Hn = ifelse.(gn .> 0, gδinvn ./ gn, 0) .+ (gδn .== 0)
    return Hn
end

# TODO: Make these functions more general, able to take any argument
#MLJBase.transform(mach::Machine, LA_nu::AbstractNode, LA_de::AbstractNode) = node((LA_nu, LA_de) -> MLJBase.transform(mach, LA_nu, LA_de), LA_nu, LA_de)
#MLJBase.transform(mach::Machine, LA_nu::DataFrame, LA_de::DataFrame) = MLJBase.transform(mach.model, mach.fitresult, LA_nu, LA_de)
