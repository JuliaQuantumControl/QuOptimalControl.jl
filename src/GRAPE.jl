# collection of GRAPE algorithms
using Zygote
using Optim

import Base.@kwdef

"""
Simple. Use Zygote to solve all of our problems
Compute the gradient of a functional f(x) wrt. x
"""
#  f_tol = 1e-3, g_tol = 1e-6, iters=1000
function _ADGRAPE(functional, x, optim_options)

    function grad_functional!(G, x)
        G .= real(Zygote.gradient(functional, x)[1])
    end

    res = optimize(functional, grad_functional!, x, LBFGS(), optim_options)

end

"""
Evaluate the figure of merit and the gradient (updated in-place) for a given specification of the problem type problem.
"""
function _fom_and_gradient_GRAPE!(
    A::T,
    B,
    control_array,
    n_timeslices,
    duration,
    n_controls,
    gradient,
    fwd_state_store,
    bwd_costate_store,
    propagators,
    X_init,
    X_target,
    evolve_store,
    system_type,
) where {T}

    dt = duration / n_timeslices

    fwd_state_store[1] .= X_init
    bwd_costate_store[end] .= X_target
    # this seems really stupid since we dont ever use the generators again, we can just keep the propagators instead
    # pw_ham_save!(A, B, control_array, n_controls, n_timeslices, @view generators[:])
    # @views propagators[:] .= exp.(generators[:] .* (-1.0im * dt))
    pw_prop_save!(A, B, control_array, n_controls, n_timeslices, dt, propagators)
    # function pw_prop_save!(H₀::T, Hₓ_array::Array{T,1}, x_arr::Array{Float64,2}, n_pulses, timeslices, timestep, out::Array{T, 1}) where T


    for t = 1:n_timeslices
        evolve_func!(
            system_type,
            t,
            fwd_state_store,
            bwd_costate_store,
            propagators,
            evolve_store,
            forward = true,
        )
    end

    for t in reverse(1:n_timeslices)
        evolve_func!(
            system_type,
            t,
            fwd_state_store,
            bwd_costate_store,
            propagators,
            evolve_store,
            forward = false,
        )
    end

    t = n_timeslices

    for c = 1:n_controls
        for t = 1:n_timeslices
            @views gradient[c, t] = grad_func!(
                system_type,
                t,
                dt,
                B[c],
                fwd_state_store,
                bwd_costate_store,
                propagators,
                evolve_store,
            )
        end
    end

    return fom_func(system_type, t, fwd_state_store, bwd_costate_store, propagators)

end

"""
Static GRAPE

Flexible GRAPE algorithm for use with StaticArrays where the size is always fixed. This works best if there are < 100 elements in the arrays. The result is that you can avoid allocations (this whole function allocates just 4 times in my tests). If your system is too large then try the GRAPE! algorithm above which should work for generic array types!
"""
function _fom_and_gradient_sGRAPE(
    A::T,
    B,
    control_array,
    n_timeslices,
    duration,
    n_controls,
    gradient,
    fwd_state_store,
    bwd_costate_store,
    X_init,
    X_target,
    system_type,
) where {T}

    dt = duration / n_timeslices
    # arrays that hold static arrays?
    fwd_state_store[1] = X_init
    bwd_costate_store[end] = X_target

    # now we want to compute the generators
    propagators = pw_evolve_save(A, B, control_array, n_controls, dt, n_timeslices)

    # forward evolution of states
    for t = 1:n_timeslices
        fwd_state_store[t+1] = evolve_func(
            system_type,
            t,
            fwd_state_store,
            bwd_costate_store,
            propagators,
            forward = true,
        )
    end
    # backward evolution of costates
    for t in reverse(1:n_timeslices)
        bwd_costate_store[t] = evolve_func(
            system_type,
            t,
            fwd_state_store,
            bwd_costate_store,
            propagators,
            forward = false,
        )
    end
    # update the gradient array
    for c = 1:n_controls
        for t = 1:n_timeslices
            gradient[c, t] = grad_func(
                system_type,
                t,
                dt,
                B[c],
                fwd_state_store,
                bwd_costate_store,
                propagators,
            )

        end
    end

    t = n_timeslices
    return fom_func(system_type, t, fwd_state_store, bwd_costate_store, propagators)
end




"""
Evolution functions for various GRAPE tasks
"""

"""
Function to compute the evolution for Unitary synthesis, since here we simply stack propagators
"""
function evolve_func(
    system_type::UnitaryGate,
    t,
    state,
    costate,
    propagator;
    forward = true,
)
    if forward
        propagator[t] * state[t]
    else
        propagator[t]' * costate[t+1]
    end
end

"""
Evolution function for use with density matrices
"""
function evolve_func(
    system_type::Union{StateTransfer,CoherenceTransfer},
    t,
    state,
    costate,
    propagator;
    forward = true,
)
    if forward
        propagator[t] * state[t] * propagator[t]'
    else
        propagator[t]' * costate[t+1] * propagator[t]
    end
end



"""
In-place evolution functions
"""
function evolve_func!(
    system_type::UnitaryGate,
    t,
    state,
    costate,
    propagator,
    store;
    forward = true,
)
    if forward
        @views mul!(state[t+1], propagator[t], state[t])
    else
        @views mul!(costate[t], propagator[t]', costate[t+1])
    end
end

"""
In-place evolution functions, I think these don't allocate at all
"""
function evolve_func!(
    system_type::Union{StateTransfer,CoherenceTransfer},
    t,
    state,
    costate,
    propagator,
    store;
    forward = true,
)
    if forward
        @views mul!(store, state[t], propagator[t]')
        @views mul!(state[t+1], propagator[t], store)
    else
        @views mul!(store, costate[t+1], propagator[t])
        @views mul!(costate[t], propagator[t]', store)
    end
end


"""
Grad functions for GRAPE
"""

"""
First order in dt gradient from Khaneja paper for unitary synthesis
"""
function grad_func!(
    system_type::UnitaryGate,
    t,
    dt,
    B,
    state,
    costate,
    props,
    store,
)::Float64
    @views mul!(store, state[t]', costate[t])
    @views 2.0 * real((1.0im * dt) .* tr(costate[t]' * B * state[t]) * tr(store))
end

function grad_func!(
    system_type::Union{StateTransfer,CoherenceTransfer},
    t,
    dt,
    B,
    state,
    costate,
    props,
    store,
)::Float64
    @views mul!(store, costate[t]', commutator(B, state[t]))
    real(tr((1.0im * dt) .* store))
end

function grad_func(system_type::UnitaryGate, t, dt, B, state, costate, props)
    2.0 * real((-1.0im * dt) * tr(costate[t]' * B * state[t]) * tr(state[t]' * costate[t]))
end

function grad_func(
    system_type::Union{StateTransfer,CoherenceTransfer},
    t,
    dt,
    B,
    state,
    costate,
    props,
)
    real(tr((1.0im * dt) .* (costate[t]' * commutator(B, state[t]))))
end
