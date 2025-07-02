using Test
using Condensity
using CausalTables
using Distributions
using MLJBase
using MLJLinearModels
using MLJModels
using Tables
using TableTransforms
using DensityRatioEstimation
using Optim

XGBoostRegressor = @load XGBoostRegressor pkg=XGBoost
XGBoostClassifier = @load XGBoostClassifier pkg=XGBoost

# Extras
using Graphs

using Random

Random.seed!(1);

# NOTE: If you have multiple interventions, i.e. a summarized A and A,
# you MUST put the summarized column first in the y input table. 
# Otherwise it will not work

σ = 1.0
scm = StructuralCausalModel(
    @dgp(
        L1 ~ Beta(3,2),
        L2 ~ Poisson(100),
        L3 ~ Gamma(2, 4),
        L4 ~ Bernoulli(0.6),
        n = length(L1),
        G ≈ Graphs.adjacency_matrix(erdos_renyi(n, 4/n)),
        F $ Friends(:G),
        L1o $ AllOrderStatistics(:L1, :G),
        L2o $ AllOrderStatistics(:L2, :G),
        L3o $ AllOrderStatistics(:L3, :G),
        L4o $ AllOrderStatistics(:L4, :G),

        L1step = (@. -(L1 > 0.2) - (L1 > 0.4) + 2*(L1 > 0.6) + (L1 > 0.8)),
        L2step = (@. -(L2 > 85) - (L2 > 95) + 2*(L2 > 105) + (L2 > 115)),
        L3step = (@. -(L3 > 2) - (L3 > 4) + 2*(L3 > 8) + (L3 > 16)),

        nonlin = (@. (L4 + L4 * L1step + L1step + L2step + L3step)),
        A ~ Normal.(nonlin .+ 4, 1.0),
        As $ Sum(:A, :G),
        Y ~ (@. truncated(Normal(
                        0.5 * nonlin + (-0.2 * (A > 1) - 0.2 * (A > 2) - 0.1 * (A > 3) - 0.1 * (A > 4) + 0.4 * (A > 5) + 
                        2 * (As > 0) + 6 * (As > 10) + 6 * (As > 15) + 2 * (As > 20) + 4 * (As > 25)), σ), 
                        0.5 * nonlin + (-0.2 * (A > 1) - 0.2 * (A > 2) - 0.1 * (A > 3) - 0.1 * (A > 4) + 0.4 * (A > 5) + 
                        2 * (As > 0) + 6 * (As > 10) + 6 * (As > 15) + 2 * (As > 20) + 4 * (As > 25)) - (6*σ), 
                        0.5 * nonlin + (-0.2 * (A > 1) - 0.2 * (A > 2) - 0.1 * (A > 3) - 0.1 * (A > 4) + 0.4 * (A > 5) + 
                        2 * (As > 0) + 6 * (As > 10) + 6 * (As > 15) + 2 * (As > 20) + 4 * (As > 25)) + (6*σ)))
    ),  
    treatment = :A,
    response = :Y
)
dat = rand(scm, 2500)

@testset "KDE" begin
    n = 100
    kde = KDE(1.0, Epanechnikov)
    X = MLJBase.table((X = rand(Normal(), n),))
    kde_mach = machine(kde, X) |> fit!
    prediction = predict(kde_mach, X)
    true_density = pdf(Normal(), X.X)
    
    @test prediction isa Array{Float64,1}
    @test all(@. prediction > 0 && prediction < 1)
    @test mean(true_density .- prediction) .< 0.1
end

@testset "OracleDensity" begin
    condensity_model = Condensity.OracleDensityEstimator(scm)

    X = treatmentparents(dat)
    y = treatment(dat)
    condensity_mach = machine(condensity_model, X, y) |> fit!
    prediction = predict(condensity_mach, dat)
    @test prediction isa Array{Float64,1}
    @test all(@. prediction > 0 && prediction < 1)
    
    true_density = propensity(scm, dat, :A)
    @test all(@. prediction > 0 && prediction < 1)
    true_density
    prediction
    @test all(prediction .== true_density)
end

@testset "LocationScaleDensity" begin
    location_model = XGBoostRegressor(objective = "reg:squarederror", tree_method = "exact",
                    num_round = 40, eta = 0.1, max_depth = 4, min_child_weight = 40)
    scale_model = XGBoostRegressor(objective = "reg:squarederror", tree_method = "exact",
                    num_round = 40, eta = 0.1, max_depth = 4, min_child_weight = 40)

    density_model = KDE(0.001, Epanechnikov)

    r = range(density_model, :bandwidth, lower=0.001, upper=0.5)
    lse_model = LocationScaleDensity(location_model, scale_model, density_model, r, CV(nfolds=10))

    X = treatmentparents(dat)
    y = treatment(dat)
    Xy = responseparents(dat)

    lse_mach = machine(lse_model, X, y) |> fit!
    prediction = predict(lse_mach, responseparents(dat))

    @test prediction isa Array{Float64,1}
    @test all(@. prediction >= 0 && prediction < 1)

    # TODO: Need a better test to determine this actually works
    true_density = propensity(scm, dat, :A)
    @test mean(prediction .- true_density) .< 0.005

    cor(true_density, prediction)
    scatter(true_density, prediction)

    # Test within DensityRatioPlugIn
    Xy = responseparents(dat)
    Xy_shift = intervene(Xy, additive_mtp(-1.0))
    density_ratio_model = Condensity.DensityRatioPlugIn(lse_model)

    a = predict(lse_mach, Xy)
    histogram(a)
    b = predict(lse_mach, Xy_shift)

    dr_mach = machine(density_ratio_model, X, y) |> fit!
    prediction_ratio = predict(dr_mach, Xy_shift, Xy)

    true_ratio = propensity(scm, Xy_shift, :A) ./ propensity(scm, Xy, :A)

    scatter(prediction_ratio, true_ratio)
    cor(prediction_ratio, true_ratio)

    @test prediction_ratio isa Array{Float64,1}
    @test all(@. prediction_ratio > 0)
    @test abs(mean(prediction_ratio .- true_ratio)) .< 0.2

end

@testset "DensityRatioClassifier" begin

    Xy_de = responseparents(dat)
    Xy_nu = intervene(responseparents(dat), additive_mtp(-0.1))

    classifier_model = XGBoostClassifier(objective = "binary:logistic", tree_method = "exact",
                    num_round = 40, eta = 0.1, max_depth = 4, min_child_weight = 40)
    drc_model = DensityRatioClassifier(classifier_model)

    drc_mach = machine(drc_model, Xy_nu, Xy_de) |> fit!
    prediction_ratio = predict(drc_mach, Xy_nu, Xy_de)
    
    @test prediction_ratio isa Array{Float64,1}
    @test all(@. prediction_ratio > 0)

    # Test that this is close to the true density ratio
    true_model = DensityRatioPlugIn(OracleDensityEstimator(scm))
    X = treatmentparents(dat)
    y = treatment(dat)
    true_mach = machine(true_model, X, y) |> fit!
    true_prediction_ratio = predict(true_mach, Xy_nu, Xy_de)
    # Test if predictions are close to truth
    @test mean(@. (true_prediction_ratio - prediction_ratio)^2) < 0.05
end

#@testset "KLIEP" begin
    
    Xy_de = responseparents(dat)
    Xy_nu = intervene(responseparents(dat), additive_mtp(-0.1))

    truedr_model = DensityRatioPlugIn(Condensity.OracleDensityEstimator(scm))
    X = treatmentparents(dat)
    y = treatment(dat)
    truedr_mach = machine(truedr_model, X, y) |> fit!
    true_ratio = predict(truedr_mach, Xy_nu, Xy_de)

    kliep_model = DensityRatioKLIEP([10.0, 20.0, 50.0], [30])
    kliep_mach = machine(kliep_model, Xy_nu, Xy_de) |> fit!
    est_ratio = predict(kliep_mach, Xy_nu, Xy_de)
    # Test if predictions are close to truth    
    @test mean(@. (est_ratio - true_ratio)^2) < 0.05
end

@testset "Kernel" begin
    Xy_de = responseparents(dat)
    Xy_nu = intervene(responseparents(dat), additive_mtp(-0.1))

    truedr_model = DensityRatioPlugIn(Condensity.OracleDensityEstimator(scm))
    X = treatmentparents(dat)
    y = treatment(dat)
    truedr_mach = machine(truedr_model, X, y) |> fit!
    true_ratio = predict(truedr_mach, Xy_de, Xy_nu)

    kernel_model = DensityRatioKernel(uKMM())
    kernel_mach = machine(kernel_model, Xy_nu, Xy_de) |> fit!
    est_ratio = predict(kernel_mach, Xy_nu, Xy_de)
    # Test if predictions are close to truth
    @test mean(@. (est_ratio - true_ratio)^2) < 0.05
end


