
mutable struct DensityRatioClassifier <: ConDensityRatioEstimator
    classifier::MMI.Probabilistic
    resampling::MT.ResamplingStrategy

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
    drc_model = DensityRatioClassifier(classifier_model, CV(nfolds = 10))

    # note that no data is needed for the machine, since training the model is only required for inference
    
    dr_mach = machine(drc_model, nothing, nothing) |> fit!

    # collect X and y for predicting the ratio between 
    # densities of a shifted and unshifted dataset
    denominator = merge(X, y) 
    numerator = merge(X, (y = Y .- 0.1,))

    predict(dr_mach, numerator, denominator)

    # output
    50-element Vector
    ```

    """
    function DensityRatioClassifier(classifier::MMI.Supervised, resampling::MT.ResamplingStrategy)
        new(classifier, resampling)
    end
end

function MMI.fit(model::DensityRatioClassifier, verbosity, Xy_nu, Xy_de)

    # Check input dimensions
    if DataAPI.nrow(Xy_nu) != DataAPI.nrow(Xy_de)
        throw(DimensionMismatch("Xy_nu and Xy_de must have the same number of rows"))
    end

    # Concatenate the numerator and denominator data
    Xy = concat_tables(Xy_nu, Xy_de)

    # Record which observations are from the numerator versus the denominator
    indicators = categorical(vcat(falses(DataAPI.nrow(Xy_nu)), trues(DataAPI.nrow(Xy_de))))

    # Initialize the classifier machine
    classifier_mach = machine(model.classifier, Xy, indicators) |> fit!

    fitresult = (classifier_machine = classifier_mach,)
    cache = nothing
    report = nothing
    return fitresult, cache, report
end

function MMI.predict(::DensityRatioClassifier, fitresult, Xy_nu)
    pred = predict(fitresult.classifier_machine, Xy_nu)
    prob_orig = pdf.(pred, false)
    prob_shift = pdf.(pred, true)
    return prob_orig ./ prob_shift
end