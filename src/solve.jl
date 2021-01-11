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

function GRAPE(prob::Union{StateTransferProblem,UnitaryProblem}; inplace = true, optim_options=Optim.Options())
    if inplace
        GRAPE!(prob, optim_options)
    else
        sGRAPE(prob, optim_options)
    end
end


"""
GRAPE!

This function solves the single problem using Optim and the gradient defined for the problem type.
"""
function GRAPE!(problem, optim_options = Optim.Options())
    
    # initialise holding arrays for inplace operations, these will be modified
    state_store, costate_store, generators, propagators, fom, gradient = init_GRAPE(problem.X_init, problem.n_timeslices, problem.A, problem.n_controls)
    
    evolve_store = similar(state_store[1])

    function _to_optim!(F, G, x)        
        fom = @views _fom_and_gradient_GRAPE!(problem.A, problem.B, x, problem.n_timeslices, problem.duration, problem.n_controls, gradient, state_store, costate_store, generators, propagators, problem.X_init, problem.X_target, evolve_store, problem)

        if G !== nothing
            @views G .= gradient
        end
        if F !== nothing
            return fom
        end
    end


    init = problem.initial_guess

    res = Optim.optimize(Optim.only_fg!(to_optim!), init, Optim.LBFGS(), optim_options)
    solres = SolutionResult([res], [res.minimum], [res.minimizer], problem)
    return solres
end

"""
Solve function for use with inplace=false, StaticArray problemlems!
"""
function sGRAPE(problem, optim_options)
    # prepare some storage arrays that we will make use of throughout the computation
    Ut = Vector{typeof(problem.A[1])}(undef, problem.n_timeslices + 1)
    Lt = Vector{typeof(problem.A[1])}(undef, problem.n_timeslices + 1)
    gradient = zeros(problem.n_ensemble, problem.n_controls, problem.n_timeslices)

    wts = ones(problem.n_ensemble)

    function to_optim!(F, G, x)
        fom = 0.0
        for k = 1:problem.n_ensemble
            grad = @views gradient[k, :, :]
            fom += _sGRAPE(problem.A[k], problem.B[k], x, problem.n_timeslices, problem.duration, problem.n_controls, grad, Ut, Lt, problem.X_init[k], problem.X_target[k], problem) * wts[k]
        end
        if G !== nothing
            @views G .= sum(gradient .* wts, dims = 1)[1, :,:]
        end
        if F !== nothing
            return fom
        end
    end

    init = problem.initial_guess

    res = Optim.optimize(Optim.only_fg!(to_optim!), init, Optim.LBFGS(), optim_options)
    solres = SolutionResult([res], [res.minimum], [res.minimizer], problem)

    return solres
end

"""
Solve closed state transfer problem using ADGRAPE, this means that we need to use a piecewise evolution function that is Zygote compatible!
"""
function ADGRAPE(prob::ClosedStateTransfer; optim_options = Optim.Options())
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
function ADGRAPE(prob::UnitarySynthesis; optim_options = Optim.Options())
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
