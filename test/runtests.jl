using Test
using CausalTables
using Condensity
using Distributions
using MLJBase
using MLJLinearModels
using MLJModels
using Tables
using TableOperations
using Graphs

using Random

Random.seed!(1);

# NOTE: If you have multiple interventions, i.e. a summarized A and A,
# you MUST put the summarized column first in the y input table. 
# Otherwise it will not work

distseq = Vector{Pair{Symbol, CausalTables.ValidDGPTypes}}([
        :L1 => (; O...) -> DiscreteUniform(1, 5),
        :L1_s => Sum(:L1, include_self = true),
        :A => (; O...) -> (@. Normal(O[:L1] + 0.2 * O[:L1_s], 0.5)),
        :A_s => Sum(:A, include_self = true),
        :Y => (; O...) -> (@. Normal(O[:A] + O[:A_s] + 0.2 * O[:L1], 1))
    ])

dgp = DataGeneratingProcess(n -> random_regular_graph(n, 2), distseq; treatment = :A, response = :Y, controls = [:L1]);
data = rand(dgp, 100)
sdata = summarize(data)

@testset "KDE" begin
    n = 100
    kde = KDE(1.0, Epanechnikov)
    X = MLJBase.table((X = randn(n),))
    kde_mach = machine(kde, X) |> fit!
    prediction = predict(kde_mach, X)
    @test prediction isa Array{Float64,1}
    @test all(@. prediction > 0 && prediction < 1)
end

# TODO: Several inaccuracy issues with this test... need to delve deeper
@testset "LocationScaleDensity" begin
    location_model = LinearRegressor()
    scale_model = ConstantRegressor()
    density_model = KDE(0.001, Epanechnikov)

    r = range(density_model, :bandwidth, lower=0.001, upper=0.5)
    lse_model = LocationScaleDensity(location_model, scale_model, density_model, r, CV(nfolds=10))

    X = reject(sdata, :A, :A_s, :Y) |> Tables.columntable
    y = TableOperations.select(sdata, :A) |> Tables.columntable
    
    lse_mach = machine(lse_model, X, y) |> fit!
    prediction = predict(lse_mach, reject(sdata, :Y) |> Tables.columntable)

    @test prediction isa Array{Float64,1}
    @test all(@. prediction >= 0 && prediction < 1)

    # TODO: Need a better test to determine this actually works
    true_density = pdf.(condensity(dgp, data, :A), y.A)
    @test sum(@. prediction * log(prediction / true_density)) < 50

    # Test within DensityRatioPlugIn
    Xy = reject(data, :Y, :A_s) |> Tables.columntable
    Xy_shift = (L1 = Tables.getcolumn(data, :L1), L1_s = Tables.getcolumn(data, :L1_s), A = Tables.getcolumn(data, :A) .- 0.5)
    density_ratio_model = Condensity.DensityRatioPlugIn(lse_model)

    dr_mach = machine(density_ratio_model, X, y) |> fit!
    prediction_ratio = predict(dr_mach, Xy, Xy_shift)

    @test prediction_ratio isa Array{Float64,1}
    @test all(@. prediction_ratio > 0)
end

@testset "OracleDensity" begin

    condensity_model = Condensity.OracleDensityEstimator(dgp)

    X = reject(data, :A, :A_s, :Y) |> Tables.columntable
    y = TableOperations.select(data, :A) |> Tables.columntable
    condensity_mach = machine(condensity_model, X, y) |> fit!
    prediction = predict(condensity_mach, sdata)
    @test prediction isa Array{Float64,1}
    @test all(@. prediction > 0 && prediction < 1)

    true_density = pdf.(condensity(dgp, sdata, :A), y.A)
    @test all(@. prediction > 0 && prediction < 1)
    true_density
    prediction
    @test all(prediction .== true_density)
end

@testset "DensityRatioClassifier" begin

    dgp = DataGeneratingProcess([
            :X => (; O...) -> Normal(0, 1),
            :y => (; O...) -> @. Normal(3 * O[:X] + 3, 1)
        ])

    Xy_nu = rand(dgp, 500)
    Xy_de = CausalTables.replacetable(Xy_nu, (X = Tables.getcolumn(Xy_nu, :X), y = Tables.getcolumn(Xy_nu, :y) .- 0.1))
    classifier_model = LogisticClassifier()
    drc_model = DensityRatioClassifier(classifier_model)

    drc_mach = machine(drc_model, Xy_nu, Xy_de) |> fit!
    prediction_ratio = predict(drc_mach, Xy_nu, Xy_de)
    
    @test prediction_ratio isa Array{Float64,1}
    @test all(@. prediction_ratio > 0)

    # Test that this is close to the true density ratio
    true_model = DensityRatioPlugIn(OracleDensityEstimator(dgp))
    true_mach = machine(true_model, reject(Xy_nu, :y), (y = Tables.getcolumn(Xy_nu, :y),)) |> fit!
    true_prediction_ratio = predict(true_mach, Xy_nu, Xy_de)
    @test mean(@. (true_prediction_ratio - prediction_ratio)^2) < 0.05
end