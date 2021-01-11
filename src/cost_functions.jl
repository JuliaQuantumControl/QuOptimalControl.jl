using LinearAlgebra

"""
This file contains a list of cost functions that are ready to be used in optimisations
I'm going to try and keep some standard inputs that make sense

Needs citation from the Schuster lab paper
"""

"""
Target-gate infidelity
"""
function C1(KT, KN)
    D = size(KT)[1]
    1 - abs2(tr(KT' * KN) / D)
end

"""
Target-state infidelity
"""
function C2(ψT, ψN)
    1 - abs2(ψT' * ψN)
end

"""
Control amplitudes
"""
function C3(u)
    sum(abs2.(u))
end

"""
Control variations
"""
function C4(u)
    (K, J) = size(u)
    sum(abs2.(diff(u, dims = 2)))
end

"""
Occupation of forbidden states
"""
function C5(ψF, ψj)
    # since ψj is the intermediate states we can assume its an array
    sum(abs2(ψF' * ψ) for ψ in ψj)
end

"""
Evolution time (target gate)
"""
function C6(KT, KJ, N, D)
    1 - 1 / N * sum(abs2(tr(KT' * Kj) / D) for Kj in KJ)
end

"""
Evolution time (target state)
"""
function C7(ψT, ψJ, N)
    1 - 1 / N * sum(abs2(tr(ψT * ψj)) for ψj in ψJ)
end

"""
Stores information, relative weights and functions to compute the penalty functions
"""
struct PenaltyFunctionals
    weights # relative weights of each
    functions # array of penalty functionals
end





"""
Cost functions for the GRAPE algorithm
"""

"""
Unitary synthesis figure of merit, from Khaneja et. al. paper (proper citation needed)
"""
function fom_func(prob::UnitaryProblem, t, k, state, costate, props, gens)::Float64
    @views tr(costate[t, k]' * state[t, k]) * tr(state[t, k]' * costate[t, k])
end

function fom_func(prob::Union{StateTransferProblem,OpenSystemCoherenceTransfer}, t, k, state, costate, props, gens)::Float64
    # recall that target is always the last entry of L
    # and that we have in U[end] the propagated forward target state
    @views C1(costate[t, k], state[t, k])
end

function fom_func(prob::UnitaryProblem, t, state, costate, props, gens)::Float64
    @views tr(state[t]' * costate[t]) * tr(state[t]' * costate[t])
end

function fom_func(prob::Union{StateTransferProblem,OpenSystemCoherenceTransfer}, t, state, costate, props, gens)::Float64
    @views C1(costate[t], state[t])
end