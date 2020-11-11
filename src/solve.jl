abstract type Solution end
using Optim
using FileWatching
using DelimitedFiles

"""
Contains the solve function interfaces for now, until I learn a better way to do it
"""

"""
The SolutionResult stores important optimisation information in a nice format
"""
struct SolutionResult <: Solution
    result # optimisation result (not saved)
    fidelity # lets just extract the figure of merit that was reached
    optimised_pulses # store an array of the optimised pulses
    prob_info # can we store the struct or some BSON of the struct that was originally used
end

# how about instead of that we give some definitions based on algorithm directly

function GRAPE(prob::Union{ClosedSystem,OpenSystem}; inplace = true, optim_options=())
    if inplace
        GRAPE!(prob, optim_options)
    else
        sGRAPE(prob, optim_options)
    end
end


"""
This solve function should handle ClosedStateTransfer, UnitarySynthesis and an open system governed in Liouville space, since the bulk of the code remains the same and we dispatch based on prob type to the correct evolution algorithms (Khaneja et. al.).
"""
function GRAPE!(prob, optim_options)
    # do stuff with _GRAPE!
    U, L, gens, props, fom, gradient = init_GRAPE(prob.X_init[1], prob.n_timeslices, prob.n_ensemble, prob.A[1], prob.n_controls)
    wts = ones(prob.n_ensemble)
    evolve_store = similar(U[:,1][1])

    function to_optim!(F, G, x)
        fom = 0.0
        for k = 1:prob.n_ensemble
            fom += @views _GRAPE!(prob.A[k], prob.B[k], x, prob.n_timeslices, prob.duration, prob.n_controls, gradient[k, :, :], U[:,k], L[:,k], gens[:,k], props[:,k], prob.X_init[k], prob.X_target[k], evolve_store, prob) * wts[k]
        end
        if G !== nothing
            @views G .= sum(gradient .* wts, dims = 1)[1, :,:]
        end
        if F !== nothing
            return fom
        end 
    end

    init = prob.initial_guess

    res = Optim.optimize(Optim.only_fg!(to_optim!), init, Optim.LBFGS(), optim_options)
    solres = SolutionResult([res], [res.minimum], [res.minimizer], prob)

    return solres

end

"""
Solve function for use with inplace=false, StaticArray problems!
"""
function sGRAPE(prob, optim_options)
    # prepare some storage arrays that we will make use of throughout the computation
    Ut = Vector{typeof(prob.A[1])}(undef, prob.n_timeslices + 1)
    Lt = Vector{typeof(prob.A[1])}(undef, prob.n_timeslices + 1)
    gradient = zeros(prob.n_ensemble, prob.n_controls, prob.n_timeslices)

    wts = ones(prob.n_ensemble)

    function to_optim!(F, G, x)
        fom = 0.0
        for k = 1:prob.n_ensemble
            grad = @views gradient[k, :, :]
            fom += _sGRAPE(prob.A[k], prob.B[k], x, prob.n_timeslices, prob.duration, prob.n_controls, grad, Ut, Lt, prob.X_init[k], prob.X_target[k], prob) * wts[k]
        end
        if G !== nothing
            @views G .= sum(gradient .* wts, dims = 1)[1, :,:]
        end
        if F !== nothing
            return fom
        end
    end

    init = prob.initial_guess

    res = Optim.optimize(Optim.only_fg!(to_optim!), init, Optim.LBFGS(), optim_options)
    solres = SolutionResult([res], [res.minimum], [res.minimizer], prob)

    return solres
end

"""
Solve closed state transfer prob using ADGRAPE, this means that we need to use a piecewise evolution function that is Zygote compatible!
"""
function ADGRAPE(prob::ClosedStateTransfer; optim_options = ())
    wts = ones(prob.n_ensemble)
    D = size(prob.A[1])[1]
    u0 = typeof(prob.A[1])(I(D))
    
    function user_functional(x)
        fom = 0.0
        for k = 1:prob.n_ensemble
            U = pw_evolve_T(prob.A[k], prob.B[k], x, prob.n_controls, prob.duration/prob.n_timeslices, prob.n_timeslices, u0)
            fom += C1(prob.X_target[k], (U * prob.X_init[k] * U')) * wts[k]
        end
        fom
    end

    init = prob.initial_guess
    res = _ADGRAPE(user_functional, init, optim_options=optim_options)

    solres = SolutionResult([res], [res.minimum], [res.minimizer], prob)
end

"""
Solve a unitary synthesis prob using ADGRAPE, this means that we need to use a piecewise evolution function that is Zygote compatible!
"""
function ADGRAPE(prob::UnitarySynthesis; optim_options = ())
    wts = ones(prob.n_ensemble)
    u0 = typeof(prob.A[1])(I(D))

    function user_functional(x)
        fom = 0.0
        for k = 1:prob.n_ensemble
            U = pw_evolve_T(prob.A[k], prob.B[k], x, prob.n_controls, prob.duration/prob.n_timeslices, prob.n_timeslices, u0)
            fom += C1(prob.X_target[k], U * prob.X_init[k]) * wts[k]
        end
        fom
    end

    init = prob.initial_guess
    res = _ADGRAPE(user_functional, init, optim_options=optim_options)

    solres = SolutionResult([res], [res.minimum], [res.minimizer], prob)
end



# """
# ClosedSystemStateTransfer using dCRAB to solve the prob
# Here we define a functional for the user, since we can assume that this is the type of prob that they want to solve
# """
# function _solve(prob::ClosedStateTransfer, alg::dCRAB_options)
#     # we define our own functional here for a closed system

#     wts = ones(prob.n_ensemble)
#     function user_functional(x)
#         fom = 0.0
#         for k = 1:prob.n_ensemble
#             # we get an a 2D array of pulses
#             U = pw_evolve(prob.A[k], prob.B[k], x, prob.n_controls, prob.duration/prob.n_timeslices, prob.n_timeslices)
#             # U = reduce(*, U)
#             fom += C2(prob.X_target, U * prob.X_init) * wts[k]
#         end
#         fom
#     end

#     coeffs, pulses, optim_results = dCRAB(prob.n_controls, prob.duration/prob.n_timeslices, prob.n_timeslices, prob.duration, alg.n_freq, alg.n_coeff, prob.initial_guess, user_functional)

#     solres = SolutionResult(optim_results, [optim_results[j].minimum for j = 1:length(optim_results)], pulses, prob)

# end


# """
# Unitary synthesis using dCRAB to solve the prob
# Here we define a functional for the user, since we can assume that this is the type of prob that they want to solve
# """
# function _solve(prob::UnitarySynthesis, alg::dCRAB_options)
#     # we define our own functional here for a closed system
#     wts = ones(prob.n_ensemble)

#     function user_functional(x)
#         fom = 0.0
#         for k = 1:prob.n_ensemble
#             # we get an a 2D array of pulses
#             U = pw_evolve(prob.A[k], prob.B[k], x, prob.n_controls, prob.duration/prob.n_timeslices, prob.n_timeslices)
#             # U = reduce(*, U)
#             fom += C1(prob.X_target[k], (U * prob.X_init[k])) * wts[k]
#         end
#         fom
#     end

#     coeffs, pulses, optim_results = dCRAB(prob.n_controls, prob.duration/prob.n_timeslices, prob.n_timeslices, prob.duration, alg.n_freq, alg.n_coeff, prob.initial_guess, user_functional)

#     solres = SolutionResult(optim_results, [optim_results[j].minimum for j = 1:length(optim_results)], pulses, prob)

# end

# """
# Closed loop experiment optimisation using dCRAB, the user functional isn't defined by the user at the moment, instead we define it. 
# """
# function _solve(prob::Experiment, alg::dCRAB_options)
#     """
#     user functional that will create a pulse file and start the experiment
#     """
#     function user_functional(x)
#         # we get a 2D array of pulses, with delimited files we can write this to a file
#         open(prob.pulse_path, "w") do io
#             writedlm(io, x')
#         end
#         # TODO you need to wait can use built in FileWatching function 
#         prob.start_exp()
#         o = watch_file(prob.infidelity_path, prob.timeout) # might need to monitor this differently

#         # then read in the result
#         infid = readdlm(prob.infidelity_path)[1]
#     end

#     coeffs, pulses, optim_results = dCRAB(prob.n_controls, prob.duration/prob.n_timeslices, prob.n_timeslices, prob.duration, alg.n_freq, alg.n_coeff, prob.initial_guess, user_functional)

#     solres = SolutionResult(optim_results, [optim_results[j].minimum for j = 1:length(optim_results)], pulses, prob)

# end

# """
# Can we combine algorithms to improve how we solve things? Comes from discussion with Phila
# """
# function hybrid_solve(prob, alg_list)
#     for alg in alg_list
#         sol = _solve(prob, alg)
#         prob.initial_guess[:] = sol.optimised_pulses
#     end
#     sol
# end
