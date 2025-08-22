using Test
using Condensity
using CausalTables
using Distributions
using MLJBase
using MLJLinearModels
using MLJModels
using Tables
using TableTransforms
using Random
using DensityRatioEstimation
using Optim

Random.seed!(1);

scm = StructuralCausalModel(
    @dgp(
        L1 ~ Beta(3,2),
        L2 ~ Binomial(2, 0.4),
        A ~ (@. Normal(L1 + 4*L2 + 1, 1.0)),
        Y ~ (@. Normal(10*L1 + 2*L2 + 2*A + 5, 1.0))
    ),
    treatment = :A,
    response = :Y
)
dat = rand(scm, 1000)

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
    @test all(prediction .== true_density)
end

@testset "LocationScaleDensity" begin
    location_model = LinearRegressor()
    scale_model = LinearRegressor()

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

    true_density = propensity(scm, dat, :A)
    @test mean((prediction .- true_density).^2) .< 0.005
    @test cor(prediction, true_density) .> 0.5

    # Test within DensityRatioPlugIn
    Xy = responseparents(dat)
    Xy_shift = intervene(Xy, additive_mtp(-1.0))
    density_ratio_model = Condensity.DensityRatioPlugIn(lse_model)
    dr_mach = machine(density_ratio_model, X, y) |> fit!
    prediction_ratio = predict(dr_mach, Xy_shift, Xy)

    true_ratio = propensity(scm, Xy_shift, :A) ./ propensity(scm, Xy, :A)

    @test prediction_ratio isa Array{Float64,1}
    @test all(@. prediction_ratio > 0)
    println("MSE: ", mean((prediction_ratio .- true_ratio).^2))
    println("Cor: ", cor(prediction_ratio, true_ratio))
    @test mean((prediction_ratio .- true_ratio).^2) .< 1.0
    @test cor(prediction_ratio, true_ratio) .> 0.5
end

@testset "DensityRatioClassifier" begin

    Xy_de = responseparents(dat)
    Xy_nu = intervene(responseparents(dat), additive_mtp(-0.1))

    classifier_model = LogisticClassifier()
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
    println("MSE: ", mean((prediction_ratio .- true_prediction_ratio).^2))
    println("Cor: ", cor(prediction_ratio, true_prediction_ratio))
    @test mean((true_prediction_ratio .- prediction_ratio).^2) < 0.1
    @test cor(true_prediction_ratio, prediction_ratio) > 0.5
    
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
    println("MSE: ", mean((est_ratio .- true_ratio).^2))
    println("Cor: ", cor(est_ratio, true_ratio))
    @test mean(@. (est_ratio - true_ratio)^2) < 0.1
    @test cor(est_ratio, true_ratio) > 0.5
end

# Other quality checkers
include("Aqua.jl")


