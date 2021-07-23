using LinearAlgebra
using StaticArrays
using FastExpm
"""
Contains code to perform time evolution
"""


abstract type IntegratorType end

Base.@kwdef struct Piecewise{TS,EXPM} <: IntegratorType 
    n_n_slices::TS = 1
    expm_method::EXPM
end


Base.@kwdef struct Continuous{OPTS} <: IntegratorType
    ode_options::OPTS
end


_get_integration_func() = pw_evolve


"""
Given a set of Hamiltonians (drift and control) compute the propagator for use with static arrays
"""
function pw_evolve(A, B, x, n_pulses, dt, n_slices, U0)
    U = U0
    for i = 1:n_slices
        # compute the propagator
        Htot = A
        for j = 1:n_pulses
            @views Htot = Htot + B[j] .* x[j, i]
        end
        U = exp((-1.0im * dt) .* Htot) * U
    end
    U
end

"""
Given a set of Hamiltonians compute the piecewise evolution saving the propagator for each time slice
"""
#TODO maybe rewrite this so its not allocation out inside but we allocate it ourselves, make it a vector of static arrays?
function pw_evolve_save(A::T, B, x, n_pulses, dt, n_slices) where {T<:StaticMatrix}
    out = Vector{T}(undef, n_slices)
    for i = 1:n_slices
        # compute the propagator
        Htot = A
        for j = 1:n_pulses
            @views Htot = Htot + B[j] * x[j, i]
        end
        @views U = exp(-1.0im * dt * Htot)# * U
        out[i] = U
    end
    out
end



"""
An almost allocation free (1 alloc in my tests) version of the Hamiltonian saver, since the update is inplace we have to be careful with the definition of out!
"""
function pw_ham_save!(A, B, x, n_pulses, n_slices, out)
    K = n_pulses
    Htot = similar(A) .* 0.0

    @views @inbounds for i = 1:n_slices
        Htot .= 0.0 .* Htot
        @inbounds for j = 1:K
            @. Htot = Htot + B[j] * x[j, i]
        end
        @. out[i] = Htot + A
    end
end

"""
An almost allocation free (1 alloc in my tests) version of the Generator saving function, this saves the generators of propagators (implementation might differ for prob)
"""
function pw_gen_save!(A, B, x, n_pulses, n_slices, duration, out)
    K = n_pulses
    Htot = similar(A) .* 0.0
    dt = duration / n_slices

    @views @inbounds for i = 1:n_slices
        Htot .= 0.0 .* Htot
        @inbounds for j = 1:K
            @. Htot = Htot + B[j] * x[j, i]
        end
        @. out[i] = (Htot + A) .* (-1.0im * dt)
    end
end


"""
Compute the propagator using fastExpm and save it in the array defined in out
"""
function pw_prop_save!(A, B, x, n_pulses, n_slices, dt, out)
    K = n_pulses
    Htot = similar(A) .* 0.0

    @inbounds for i = 1:n_slices
        Htot .= 0.0 .* Htot
        @inbounds for j = 1:K
            @views Htot .= Htot .+ B[j] .* x[j, i]
        end
        @views out[i] .= exp((-1.0im * dt) .* (Htot .+ A))
    end
end
