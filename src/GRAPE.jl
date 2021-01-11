# collection of GRAPE algorithms
using Zygote
using Optim
using ExponentialUtilities

import Base.@kwdef

"""
Simple. Use Zygote to solve all of our problems
Compute the gradient of a functional f(x) wrt. x
"""
#  f_tol = 1e-3, g_tol = 1e-6, iters=1000
function _ADGRAPE(functional, x; optim_options = Optim.options())

    function grad_functional!(G, x)
        G .= Zygote.gradient(functional, x)[1]
    end
    
    res = optimize(functional, grad_functional!, x, LBFGS(), optim_options)

end

"""
Flexible GRAPE algorithm that can use any array type to solve GRAPE problems. This is best when your system is so large that StaticArrays cannot be used! The arrays given will be updated in place!
"""
function _GRAPE!(A::T, B, u_c, n_timeslices, duration, n_controls, gradient, U_k, L_k, gens, props, X_init, X_target, evolve_store, prob) where T
    dt = duration / n_timeslices
    U_k[1] .= X_init
    L_k[end] .= X_target

    pw_ham_save!(A, B, u_c, n_controls, n_timeslices, @view gens[:])
    @views props[:] .= exp.(gens[:] .* (-1.0im * dt))

    for t = 1:n_timeslices
        evolve_func!(prob, t, U_k, L_k, props, gens, evolve_store, forward = true)
    end

    for t = reverse(1:n_timeslices)
        evolve_func!(prob, t, U_k, L_k, props, gens, evolve_store, forward = false)
    end

    t = n_timeslices
    
    for c = 1:n_controls
        for t = 1:n_timeslices
            @views gradient[c, t] = grad_func!(prob, t, dt, B[c], U_k, L_k, props, gens, evolve_store)
        end
    end

    return fom_func(prob, t, U_k, L_k, props, gens)

end

"""
Static GRAPE

Flexible GRAPE algorithm for use with StaticArrays where the size is always fixed. This works best if there are < 100 elements in the arrays. The result is that you can avoid allocations (this whole function allocates just 4 times in my tests). If your system is too large then try the GRAPE! algorithm above which should work for generic array types!
"""
function _sGRAPE(A::T, B, u_c, n_timeslices, duration, n_controls, gradient, U_k, L_k, X_init, X_target, prob) where T
    
    dt = duration / n_timeslices
    # arrays that hold static arrays?
    U_k[1] = X_init
    L_k[end] = X_target


    # now we want to compute the generators 
    gens = pw_ham_save(A, B, u_c, n_controls,n_timeslices)
    props = exp.(gens .* (-1.0im * dt))


    # forward evolution of states
    for t = 1:n_timeslices
        U_k[t+1] = evolve_func(prob, t, U_k, L_k, props, gens, forward = true)
    end
    # backward evolution of costates
    for t = reverse(1:n_timeslices)
        L_k[t] = evolve_func(prob, t, U_k, L_k, props, gens, forward = false)
    end
    # update the gradient array
    for c = 1:n_controls
        for t = 1:n_timeslices
            gradient[c, t] = grad_func(prob, t, dt, B[c], U_k, L_k, props, gens)

        end
    end

    t = n_timeslices
    return fom_func(prob, t, U_k, L_k, props, gens)
end




"""
Evolution functions for various GRAPE tasks
"""

"""
Function to compute the evolution for Unitary synthesis, since here we simply stack propagators
"""
function evolve_func(prob::UnitarySynthesis, t, U, L, props, gens; forward = true)
    if forward
        props[t] * U[t]
    else
        props[t]' * L[t + 1]
    end
end

"""
Evolution function for use with density matrices
"""
function evolve_func(prob::Union{ClosedStateTransfer,OpenSystemCoherenceTransfer}, t, U, L, props, gens ;forward = true)
    if forward
        props[t] * U[t] * props[t]'
    else
        props[t]' * L[t + 1] * props[t]
    end
end



"""
In-place evolution functions 
"""
function evolve_func!(prob::UnitarySynthesis, t, U, L, props, gens, store; forward = true)
    if forward
        @views mul!(U[t+1], props[t], U[t])
    else
        @views mul!(L[t], props[t]', L[t+1])
    end
end

"""
In-place evolution functions, I think these don't allocate at all
"""
function evolve_func!(prob::Union{ClosedStateTransfer,OpenSystemCoherenceTransfer}, t, U, L, props, gens, store; forward = true)
    if forward
        @views mul!(store, U[t], props[t]')
        @views mul!(U[t+1], props[t], store)
    else
        @views mul!(store, L[t+1], props[t])
        @views mul!(L[t], props[t]',  store)
    end
end


"""
Grad functions for GRAPE
"""

"""
First order in dt gradient from Khaneja paper for unitary synthesis
"""
function grad_func!(prob::UnitarySynthesis, t, dt, B, U, L, props, gens, store)::Float64
    @views mul!(store, U[t]', L[t])
    @views 2.0 * real((1.0im * dt) .* tr(L[t]' * B * U[t]) * tr(store))
end

function grad_func!(prob::Union{ClosedStateTransfer,OpenSystemCoherenceTransfer}, t, dt, B, U, L, props, gens, store)::Float64
    @views mul!(store, L[t]', commutator(B, U[t]))
    real(tr((1.0im * dt) .* store))
end

function grad_func(prob::UnitarySynthesis, t, dt, B, U, L, props, gens)
    2.0 * real((-1.0im * dt)* 
        tr(L[t]' * B * U[t]) * tr(U[t]' * L[t]) 
    )
end

function grad_func(prob::Union{ClosedStateTransfer,OpenSystemCoherenceTransfer}, t, dt, B, U, L, props, gens)
    real(tr(
        (1.0im * dt) .* (L[t]' * commutator(B, U[t]))
    ))
end


