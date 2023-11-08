using Condense
using Test
using MLJ
using Distributions

@testset "Condense.jl" begin
    @test include("oracle.jl")
end
