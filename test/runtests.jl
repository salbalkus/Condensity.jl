using Condensity
using Distributions
using MLJBase
using MLJLinearModels
using MLJModels
using CausalTables
using Tables
using TableOperations

using Test
using Random

Random.seed!(1);

@testset "KDE" begin
    n = 100
    kde = KDE(1.0, Epanechnikov)
    X = MLJBase.table((X = randn(n),))
    kde_mach = machine(kde, X) |> fit!
    prediction = predict(kde_mach, X)
    @test prediction isa Array{Float64,1}
    @test all(@. prediction > 0 && prediction < 1)
end

@testset "LocationScaleDensity" begin
    location_model = LinearRegressor()
    scale_model = ConstantRegressor()
    density_model = KDE(0.1, Epanechnikov)

    r = range(density_model, :bandwidth, lower=0.01, upper=0.5)
    lse_model = LocationScaleDensity(location_model, scale_model, density_model, r, CV(nfolds=10))

    dgp = DataGeneratingProcess([
        :X => (; O...) -> Normal(0, 1),
        :y => (; O...) -> @. Normal(3 * O[:X] + 3, 1)
    ])
    data = rand(dgp, 1000)

    X = reject(data, :y) |> Tables.columntable
    y = TableOperations.select(data, :y) |> Tables.columntable
    lse_mach = machine(lse_model, X, y) |> fit!
    
    prediction = predict(lse_mach, data)

    @test prediction isa Array{Float64,1}
    @test all(@. prediction > 0 && prediction < 1)

    # TODO: Need a better test to determine this actually works
    true_density = pdf.(condensity(dgp, data, :y), y.y)

    @test sum(@. true_density * log(true_density / prediction)) < 50

    # Test within DensityRatioPropensity
    data_shift = (X = Tables.getcolumn(data, :X), y = Tables.getcolumn(data, :y) .- 0.1)

    density_ratio_model = Condensity.DensityRatioPropensity(lse_model)
    dr_mach = machine(density_ratio_model, X, y) |> fit!
    prediction_ratio = predict(dr_mach, data, data_shift)

    @test prediction_ratio isa Array{Float64,1}
    @test all(@. prediction_ratio > 0)
end

@testset "OracleDensity" begin

    dgp = DataGeneratingProcess([
        :X1 => (; O...) -> Categorical(3),
        #:X2 => (; O...) -> Normal(0, 0.1),
        :y => (; O...) -> @. Normal(3 * O[:X1] + 3, 1)
    ])
    data = rand(dgp, 100)

    condensity_model = Condensity.OracleDensityEstimator(dgp)

    X = reject(data, :y) |> Tables.columntable
    y = TableOperations.select(data, :y) |> Tables.columntable
    
    condensity_mach = machine(condensity_model, X, y) |> fit!
    
    prediction = predict(condensity_mach, data)
    
    @test prediction isa Array{Float64,1}
    @test all(@. prediction > 0 && prediction < 1)

    true_density = pdf.(condensity(dgp, data, :y), y.y)
    @test all(@. prediction > 0 && prediction < 1)
    @test all(prediction .== true_density)
end

@testset "DensityRatioClassifier" begin

    dgp = DataGeneratingProcess([
            :X => (; O...) -> Normal(0, 1),
            :y => (; O...) -> @. Normal(3 * O[:X] + 3, 1)
        ])

    Xy_nu = rand(dgp, 500)
    Xy_de = (X = Tables.getcolumn(Xy_nu, :X), y = Tables.getcolumn(Xy_nu, :y) .- 0.1)

    classifier_model = LogisticClassifier()
    drc_model = DensityRatioClassifier(classifier_model, CV(nfolds = 10))

    drc_mach = machine(drc_model, nothing, nothing) |> fit!
    prediction_ratio = predict(drc_mach, Xy_nu, Xy_de)

    @test prediction_ratio isa Array{Float64,1}
    @test all(@. prediction_ratio > 0)

    # Test that this is close to the true density ratio
    true_model = DensityRatioPropensity(OracleDensityEstimator(dgp))
    true_mach = machine(true_model, reject(Xy_nu, :y), (y = Tables.getcolumn(Xy_nu, :y),)) |> fit!
    true_prediction_ratio = predict(true_mach, Xy_nu, Xy_de)

    @test mean(@. (true_prediction_ratio - prediction_ratio)^2) < 0.05
    
end