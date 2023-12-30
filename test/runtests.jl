using .Condensity
using Test
using MLJ
using Distributions


@testset "Condensity.jl" begin
    @test include("oracle.jl")
end
