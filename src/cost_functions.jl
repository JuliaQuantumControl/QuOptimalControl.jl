using LinearAlgebra

"""
This file contains a list of cost functions that are ready to be used in optimisations
I'm going to try and keep some standard inputs that make sense

Needs citation from the Schuster lab paper
"""

"""
Target-gate fidelity
"""
function C1(KT, KN, D)
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
    1 - 1 / N * sum(abs2(tr(ψT, ψj)) for ψj in ψJ)
end
