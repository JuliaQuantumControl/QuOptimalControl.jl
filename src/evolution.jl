using LinearAlgebra
using StaticArrays
"""
Contains code to perform time evolution
"""

"""
Given a set of Hamiltonians (drift and control) compute the evolution
"""
function pw_evolve(H₀::T, Hₓ_array::Array{T,1}, x_arr::Array{Float64}, n_pulses, timestep, timeslices)::T where T
    D = size(H₀)[1] # get dimension of the system
    K = n_pulses
    U0::T = T(I(D))
    for i = 1:timeslices
        # compute the propagator
        Htot = H₀
        for j = 1:K
            @views Htot = Htot + Hₓ_array[j] * x_arr[j, i]
        end
        @views U0 = exp(-1.0im * timestep * Htot) * U0
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
    U0 = T(I(D))
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
    out = T[]
    U0 = T(I(D))
    for i = 1:timeslices
    # compute the propagator
        Htot = H₀
        for j = 1:K
            @views Htot = Htot + Hₓ_array[j] * x_arr[j, i]
        end
        @views U0 = exp(-1.0im * timestep * Htot)# * U0
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
    out = Vector{T}(undef, timeslices)
    U0 = T(I(D)) # I think this works like I want
    for i = 1:timeslices
        Htot = H₀
        for j = 1:K
            @views Htot = Htot + Hₓ_array[j] * x_arr[j, i]
        end
        out[i] = Htot
    end
    out
end

"""
An almost allocation free (1 alloc in my tests) version of the Hamiltonian saver, since the update is inplace we have to be careful with the definition of out!
"""
function pw_ham_save!(H₀::T, Hₓ_array::Array{T,1}, x_arr::Array{Float64,2}, n_pulses, timeslices, out) where T
    K = n_pulses
    Htot = similar(H₀)

    @views @inbounds for i = 1:timeslices
        @. Htot = 0.0 * Htot
        @inbounds for j = 1:K
            @. Htot = Htot + Hₓ_array[j] * x_arr[j, i]
        end
        @. out[i] = Htot + H₀
    end
end

"""
An almost allocation free (1 alloc in my tests) version of the Generator saving function, this saves the generators of propagators (implementation might differ for prob)
"""
function pw_gen_save!(prob::Union{ClosedStateTransfer,UnitarySynthesis}, H₀::T, Hₓ_array::Array{T,1}, x_arr::Array{Float64,2}, n_pulses, timeslices, duration, out) where T
    K = n_pulses
    Htot = similar(H₀)
    dt = duration/timeslices

    @views @inbounds for i = 1:timeslices
        @. Htot = 0.0 * Htot
        @inbounds for j = 1:K
            @. Htot = Htot + Hₓ_array[j] * x_arr[j, i]
        end
        @. out[i] = (Htot + H₀) .* (-1.0im * dt)
    end
end






"""
Evolution functions for various GRAPE tasks
"""

# there really must be a better way to do this, given that we simply need to multiply two matrices and sometimes conjugate
"""
Function to compute the evolution for Unitary synthesis, since here we simply stack propagators
"""

# ask someone if I've gone mad about using multiple dispatch here....
function evolve_func(prob::UnitarySynthesis, t, k, U, L, P_list, Gen, store; forward = true)
    if forward
        P_list[t, k] * U[t, k]
    else
        P_list[t,k]' * L[t + 1, k]
    end
end

function evolve_func(prob::Union{ClosedStateTransfer,OpenSystemCoherenceTransfer}, t, k, U, L, P_list, Gen, store;forward = true)
    if forward
        P_list[t, k] * U[t, k] * P_list[t, k]'
    else
        P_list[t, k]' * L[t + 1, k] * P_list[t, k]
    end
end



"""
In-place evolution functions 
"""
function evolve_func!(prob::UnitarySynthesis, t, k, U, L, P_list, Gen, store; forward = true)
    if forward
        @views mul!(U[t+1, k], P_list[t, k], U[t, k])
    else
        @views mul!(L[t, k], P_list[t, k]', L[t+1, k])
    end
end

"""
In-place evolution functions, I think these don't allocate at all
"""
function evolve_func!(prob::Union{ClosedStateTransfer,OpenSystemCoherenceTransfer}, t, k, U, L, P_list, Gen, store; forward = true)
    if forward
        @views mul!(store, U[t, k], P_list[t, k]')
        @views mul!(U[t+1, k], P_list[t, k], store)
    else
        @views mul!(store, L[t+1, k], P_list[t, k])
        @views mul!(L[t, k], P_list[t, k]',  store)
    end
end

