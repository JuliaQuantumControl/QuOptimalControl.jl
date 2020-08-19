"""
Grad functions for GRAPE
"""

# include("./problems.jl")

"""
First order in dt gradient from Khaneja paper for unitary synthesis
"""
function grad_func(prob::UnitarySynthesis, t, dt, k, H_ctrl, U, L, P_list, Gen)::Float64
    # this might be a bad idea
    -2.0 * real(tr(L[t, k]' * 1.0im * dt * H_ctrl * U[t, k]) * tr(U[t, k]' * L[t, k]))
end

function grad_func(prob::Union{ClosedStateTransfer,OpenSystemCoherenceTransfer}, t, dt, k, H_ctrl, U, L, P_list, Gen)::Float64
    real(tr(L[t, k]' * 1.0im * dt * commutator(H_ctrl, U[t, k])))
    # real(tr((ρₜ_list[j])' * 1.0im * dt * commutator(H_ctrl_arr[k], ρ_list[j])))
end

