"""
Grad functions for GRAPE
"""

# include("./problems.jl")

"""
First order in dt gradient from Khaneja paper for unitary synthesis
"""
function grad_func!(prob::UnitarySynthesis, t, dt, k, B, U, L, props, gens, store)::Float64
    @views mul!(store, U[t, k]', L[t, k])
    @views -2.0 * real((1.0im * dt) * tr(L[t, k]' * B * U[t, k]) * tr(store))
end

# function grad_func(prob::Union{ClosedStateTransfer,OpenSystemCoherenceTransfer}, t, dt, k, B, U, L, props, gens, store)::Float64
#     @views real(tr(L[t, k]' * 1.0im * dt * commutator(B, U[t, k])))
# end

function grad_func!(prob::Union{ClosedStateTransfer,OpenSystemCoherenceTransfer}, t, dt, k, B, U, L, props, gens, store)::Float64
    @views mul!(store, L[t, k]', commutator(B, U[t, k]))

    real(tr((1.0im * dt) * store))
end

function grad_func(prob::UnitarySynthesis, t, dt, B, U, L, props, gens)
    -2.0 * real((-1.0im * dt)* 
        tr(L[t]' * B * U[t]) * tr(U[t]' * L[t]) 
    )
end

function grad_func(prob::Union{ClosedStateTransfer,OpenSystemCoherenceTransfer}, t, dt, B, U, L, props, gens)
    real(
        tr(
            (1.0im * dt) * (L[t]' * commutator(B, U[t]))
        )
    )
end


