"""
Grad functions for GRAPE
"""

# include("./problems.jl")

"""
First order in dt gradient from Khaneja paper for unitary synthesis
"""
function grad_func(prob::UnitarySynthesis, t, dt, k, H_ctrl, U, L, P_list, Gen, store)::Float64
    # this might be a bad idea
    # @views -2.0 * real(tr(L[t, k]' * 1.0im * dt * H_ctrl * U[t, k]) * tr(U[t, k]' * L[t, k]))
    # @views mul!(store, L[t, k]', (1.0im * dt) .* )
    @views mul!(store, U[t, k]', L[t, k])
    @views -2.0 * real(tr(L[t, k]' * 1.0im * dt * H_ctrl * U[t, k]) * tr(store))
end

# function grad_func(prob::Union{ClosedStateTransfer,OpenSystemCoherenceTransfer}, t, dt, k, H_ctrl, U, L, P_list, Gen, store)::Float64
#     @views real(tr(L[t, k]' * 1.0im * dt * commutator(H_ctrl, U[t, k])))
# end

function grad_func(prob::Union{ClosedStateTransfer,OpenSystemCoherenceTransfer}, t, dt, k, H_ctrl, U, L, P_list, Gen, store)::Float64
    @views mul!(store, L[t, k]', (1.0im * dt) * commutator(H_ctrl, U[t, k]))

    real(tr(store))
end

