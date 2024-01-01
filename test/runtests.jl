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

@testset "KDE struct tests" begin
    kde = KDE(1.0, Normal)
    @test kde.bandwidth == 1.0
    @test kde.kernel == Normal
end

@testset "KDE predict tests" begin
    n = 100
    kde = KDE(1.0, Epanechnikov)
    X = MLJBase.table((X = randn(n),))
    kde_mach = machine(kde, X) |> fit!
    prediction = predict(kde_mach, X)
    @test prediction isa Array{Float64,1}
    @test all(@. prediction > 0 && prediction < 1)
end

@testset "LocationScaleDensity struct tests" begin
    location_model = LinearRegressor()
    scale_model = ConstantRegressor()
    density_model = KDE(0.1, Epanechnikov)

    r = range(density_model, :bandwidth, lower=0.01, upper=0.5)
    lse_model = LocationScaleDensity(location_model, scale_model, density_model, r, CV(nfolds=10))

    dgp = DataGeneratingProcess([
        :X1 => (; O...) -> Categorical(3),
        #:X2 => (; O...) -> Normal(0, 0.1),
        :y => (; O...) -> @. Normal(3 * O[:X1] + 3, 1)
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

    @test all(@. prediction > 0 && prediction < 1)
    @test sum(@. true_density * log(true_density / prediction)) < 50

end

@testset "OracleDensity tests" begin

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

