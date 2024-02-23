using Documenter, Condensity

makedocs(
    sitename="Condensity.jl",
    pages = [
        "Home" => "index.md",
        "Conditional Density Estimation" =>                 "man/density.md",
        "Conditional Density Ratio Estimation" =>           "man/density-ratio.md",
        "Oracle Density Estimation using `CausalTables.jl`" =>    "man/oracle.md",
        "API" =>                                                  "man/api.md",
    ]
)

deploydocs(
    repo = "github.com/salbalkus/Condensity.jl.git"
)