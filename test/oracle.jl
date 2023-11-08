
@testset "oracle_density.jl" begin
    g0(X) = Normal.(X, 0.1)

    Xv = rand(Normal(), 100)
    yv = rand.(g0(Xv))

    y = (y1 = yv,)
    treatment_names = propertynames(y)

    X = table((x1 = Xv,))
    y = table((y1 = yv,))
    Xnew = table((x1 = Xv, y1 = yv))

    oracle = OracleDensity(g0)

    mach = machine(oracle, X, y) |> fit!
    ypred = predict(mach, Xnew)

    @test all(ypred .== pdf.(g0(Xv), yv))
end

true