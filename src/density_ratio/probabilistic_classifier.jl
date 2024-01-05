mutable struct DensityRatioClassifier <: ConDensityRatioEstimator
    classifier::MMI.Supervised
    resampling::MT.ResamplingStrategy
end

function MMI.fit(model::DensityRatioClassifier, verbosity, X, y)
    fitresult = nothing
    cache = nothing
    report = nothing
    return fitresult, cache, report
end

function MMI.predict(model::DensityRatioClassifier, fitresult, Xy_nu, Xy_de)

    # Check input dimensions
    if DataAPI.nrow(Xy_nu) != DataAPI.nrow(Xy_de)
        throw(DimensionMismatch("Xy_nu and Xy_de must have the same number of rows"))
    end

    # Initialize the output
    Hn = Vector{Float64}(undef, DataAPI.nrow(Xy_nu))

    # Concatenate the numerator and denominator data
    Xy = concat_tables(Xy_nu, Xy_de)

    # Record which observations are from the numerator versus the denominator
    indicators = categorical(vcat(falses(DataAPI.nrow(Xy_nu)), trues(DataAPI.nrow(Xy_de))))

    # Initialize the classifier machine
    classifier_mach = machine(model.classifier, Xy, indicators)

    # Proceed with cross-fitting
    tt_pairs = train_test_pairs(model.resampling, 1:DataAPI.nrow(Xy_nu))

    for (train, test) in tt_pairs
        # Make sure each resampled Xy_nu is matched up with its analogous Xy_de for fitting
        extra_train_rows = vcat(train, train .+ DataAPI.nrow(Xy_nu))
        fit!(classifier_mach, rows=extra_train_rows)

        # Compute and store the out-of-fold predicted odds ratio
        pred = predict(classifier_mach, Tables.subset(Xy_nu, test))
        prob_orig = pdf.(pred, false)
        prob_shift = pdf.(pred, true)
        Hn[test] = prob_orig ./ prob_shift
    end
    return Hn
end