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
    reject(data, symb...)

Remove specified columns from a table.

# Arguments
- `data`: The input table.
- `symb...`: Symbols representing the columns to be removed.

# Returns
A new table with the specified columns removed.

"""
function reject(data, symb...)
    return TableOperations.select(data, filter(∉(symb), Tables.columnnames(data))...)
end