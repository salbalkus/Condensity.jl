"""
    negmeanloglik(y, ypred)

Compute the negative mean log-likelihood between the observed values `y` and the predicted values `ypred`.

# Arguments
- `y`: Observed values.
- `ypred`: Predicted values.

# Returns
The negative mean log-likelihood.

"""
negmeanloglik(y, ypred) = -mean(log.(ypred))

"""
    meanloglik(y, ypred)

Compute the mean log-likelihood between the observed values `y` and the predicted values `ypred`.

# Arguments
- `y`: Observed values.
- `ypred`: Predicted values.

# Returns
The mean log-likelihood.

"""
meanloglik(y, ypred) = mean(log.(ypred))

"""
    merge_tables(tables...)

Merge multiple tables into a single table by column-wise concatenation.

# Arguments
- `tables`: The tables to be merged.

# Returns
A merged table.

"""
merge_tables(tables...) = merge(Tables.columntable.(tables)...)

"""
    concat_tables(tables...)

Concatenates multiple tables into a single table by row.

# Arguments
- `tables`: The tables to be concatenated.

# Returns
A new table that is the row-wise concatenation of all input tables.
"""
concat_tables(tables...) = vcat(Tables.rowtable.(tables)...) |> Tables.columntable

"""
    bound(X::Vector; lower = -Inf, upper = Inf)

Bound the elements of a vector `X` within the specified lower and upper limits.

# Arguments
- `X::Vector`: The input vector.
- `lower::Real = -Inf`: The lower limit for the elements of `X`.
- `upper::Real = Inf`: The upper limit for the elements of `X`.

# Returns
- `X::Vector`: The vector `X` with elements bounded within the specified limits.

"""
function bound(X::Vector; lower = -Inf, upper = Inf)
    X = copy(X)
    X[X .> upper] .= upper
    X[X .< lower] .= lower
    return X
end

function bound!(X::Vector; lower = -Inf, upper = Inf)
    X[X .> upper] .= upper
    X[X .< lower] .= lower
end
