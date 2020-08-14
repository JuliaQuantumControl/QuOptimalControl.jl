using LinearAlgebra
using StaticArrays
"""
Contains code to perform time evolution
"""

include("./problems.jl")

"""
Given a set of Hamiltonians (drift and control) compute the evolution
"""
function pw_evolve(H₀::T, Hₓ_array::Array{T,1}, x_arr::Array{Float64}, n_pulses, timestep, timeslices)::T where T
    D = size(H₀)[1] # get dimension of the system
    K = n_pulses
    U0 = SMatrix{D,D,ComplexF64}(I(D))
    for i = 1:timeslices
        # compute the propagator
        Htot = SMatrix{D,D,ComplexF64}(H₀ + sum(Hₓ_array .* x_arr[:, i]))
        U0 = exp(-1.0im * timestep * Htot) * U0
    end
    U0
end

"""
Given a set of Hamiltonians (drift and control) compute the evolution, Zygote compatible
lets dispatch to this properly sometime
"""
function pw_evolve_T(H₀::T, Hₓ_array::Array{T,1}, x_arr::Array{Float64}, n_pulses, timestep, timeslices)::T where T
    x_arr = real.(complex.(x_arr)) # needed for Zygote to use complex numbers internally
    D = size(H₀)[1] # get dimension of the system
    K = n_pulses
    U0 = I(D)
    for i = 1:timeslices
        # compute the propagator
        Htot = H₀ + sum(Hₓ_array .* x_arr[:, i])
        U0 = exp(-1.0im * timestep * Htot) * U0
    end
    U0
end

"""
Given a set of Hamiltonians compute the piecewise evolution saving the propagator for each time slice
"""
function pw_evolve_save(H₀::T, Hₓ_array::Array{T,1}, x_arr::Array{Float64}, n_pulses, timestep, timeslices) where T
    D = size(H₀)[1] # get dimension of the system
    K = n_pulses
    out = []
    U0 = SMatrix{D,D,ComplexF64}(I(2))
    for i = 1:timeslices
    # compute the propagator
        Htot = SMatrix{D,D,ComplexF64}(H₀ + sum(Hₓ_array .* x_arr[:, i]))
        U0 = exp(-1.0im * timestep * Htot)# * U0
        append!(out, [U0])
    end
    out
end



"""
Function to compute the Hamiltonians for a piecewise constant control and save them. This is useful in many stages
"""
function pw_ham_save(H₀::T, Hₓ_array::Array{T,1}, x_arr::Array{Float64}, n_pulses, timeslices) where T
    D = size(H₀)[1] # get dimension of the system
    K = n_pulses
    out = []
    U0 = SMatrix{D,D,ComplexF64}(I(2))
    for i = 1:timeslices
        Htot = SMatrix{D,D,ComplexF64}(H₀ + sum(Hₓ_array .* x_arr[:, i]))
        append!(out, [Htot])
    end
    out
end


"""
Evolution functions for various GRAPE tasks
"""
# does this really need to be so complicated? I think it'll just dispatch to the appropriate method given number of args rather than problem type
# also since we always know that in this case it'll just 

# """
# Let's use the same trick as before with multiple dispatch
# """
# function evolve_func(prob, t, k, U, L, P_list, Gen; forward = true)
#     return _evolve_func(prob, t, k, U, L, P_list, Gen, forward)
# end

# there really must be a better way to do this, given that we simply need to multiply two matrices and sometimes conjugate
"""
Function to compute the evolution for Unitary synthesis, since here we simply stack propagators
"""
function evolve_func(prob::UnitarySynthesis, t, k, U, L, P_list, Gen, forward)
    if forward
        P_list[t, k] * U[t, k]
    else
        P_list[t,k]' * L[t + 1, k]
    end
end
