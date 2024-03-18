@doc raw"""
DensityRatioClassifier(classifier::MMI.Supervised, resampling::MT.ResamplingStrategy)

This model estimates a conditional density ratio using probabilistic classification. 
An augmented dataset is created by concatenating the data provided to the numerator and denominator data; that is, if the ratio is ``p_n(Y'|X) / p_d(Y|X)``, then the augmented dataset is ``(X, y, \Lambda)`` where ``\Lambda`` is a binary variable indicating whether the observation is from the numerator.
Then, the provided supervised classifier is trained to predict ``P(\Lambda = 1 | Y, X)``.
When predict is called, Bayes' Theorem is used to compute the density ratio ``H_n(X) = \frac{p_n(Y'|X)}{p_n(Y|X)}`` as ``H_n(X) = \frac{P(\Lambda = 1 | Y, X)}{1 - P(\Lambda = 1 | Y, X)}`` using the classifier's predictions.
In essence, this machine wraps a classifier model and transforms its output to a density ratio.

# Arguments
- `classifier::MMI.Probabilistic`: The underlying supervised classifier used for density ratio estimation.

# Example
```jldoctest; output = false, filter = r"(?<=.{17}).*"s
using Condensity
using MLJ
using MLJLinearModels

n = 50
X = randn(n)
Y = 4 .+ 2 .* X .+ randn(n)

X = (X = X,)
y = (y = Y,)

classifier_model = LogisticClassifier()
drc_model = DensityRatioClassifier(classifier_model)    

# collect X and y for predicting the ratio between 
# densities of a shifted and unshifted dataset
denominator = merge(X, y) 
numerator = merge(X, (y = Y .- 0.1,))

dr_mach = machine(drc_model, numerator, denominator) |> fit!
predict(dr_mach, numerator, denominator)

# output
50-element Vector
```

"""
mutable struct DensityRatioClassifier{T <: MMI.Probabilistic} <: ConDensityRatioEstimatorFixed
    classifier::T
end


function MMI.fit(model::DensityRatioClassifier, verbosity, Xy_nu, Xy_de)

    # Concatenate the numerator and denominator data
    Xy = concat_tables(Xy_nu, Xy_de)

    n_nu = DataAPI.nrow(Xy_nu)
    n_de = DataAPI.nrow(Xy_de)

    # Record which observations are from the numerator versus the denominator
    indicators = categorical(vcat(falses(n_nu), trues(n_de)))

    # Initialize the classifier machine
    classifier_mach = machine(model.classifier, Xy, indicators) |> fit!

    fitresult = (classifier_machine = classifier_mach, n_ratio = n_de / n_nu)
    cache = nothing
    report = nothing
    return fitresult, cache, report
end

function MMI.predict(::DensityRatioClassifier, fitresult, Xy_nu, Xy_de)
    pred = predict(fitresult.classifier_machine, Xy_nu)
    prob_orig = pdf.(pred, true)
    prob_shift = pdf.(pred, false)
    return fitresult.n_ratio .* prob_orig ./ prob_shift
end