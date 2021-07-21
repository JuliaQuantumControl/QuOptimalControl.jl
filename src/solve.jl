using Optim
using FileWatching
using DelimitedFiles
using Parameters

# what is this
abstract type Solution end

"""
Solution type - 

The SolutionResult stores important optimisation information in a nice format
"""
struct SolutionResult{R, FID, OPT, P <: Problem, A}
    result::R
    fidelity::FID
    opti_pulses::OPT
    problem::P
    alg::A
end



Base.@kwdef struct GRAPE{NS, iip, opts}
    n_slices::NS = 1
    isinplace::iip = true
    optim_options::opts = Optim.Options()
end

# Base.@kwdef struct dCRAB end
# Base.@kwdef struct ADGRAPE end
# Base.@kwdef struct ADGROUP end
default_algorithm(::Problem) = GRAPE()

This function solves the single problem using Optim and the gradient defined for the problem type.
"""
function GRAPE!(problem)
    
            if G !== nothing
                @views G .= grad
            end
            if F !== nothing
                return fom
            end
        end
    
    evolve_store = similar(state_store[1])
    function _to_optim!(F, G, x)   
        fom = 0.0     
        k = 1
        fom += @views QuOptimalControl._fom_and_gradient_GRAPE!(problem.A, problem.B, x, problem.n_timeslices, problem.duration, problem.n_controls, gradient[k, :, :], state_store[:, k], costate_store[:, k], generators[:,k], propagators[:,k], problem.X_init, problem.X_target, evolve_store, problem)
        if G !== nothing
            @views G .= gradient[1,:,:]
        end
        if F !== nothing
            return fom
        end

    end

    init = problem.initial_guess

    res = Optim.optimize(Optim.only_fg!(_to_optim!), init, Optim.LBFGS())
    solres = SolutionResult([res], [res.minimum], [res.minimizer], problem)
    return solres
end



println("stop")


# """
# Contains the solve function interfaces for now, until I learn a better way to do it
# """
# function GRAPE(prob::Union{StateTransferProblem,UnitaryProblem}; inplace = true, optim_options=Optim.Options())
#     if inplace
#         GRAPE!(prob, optim_options)
#     else
#         sGRAPE(prob, optim_options)
#     end
# end

# """
# Handles solving ensemble problems with GRAPE
# """
# function GRAPE(ens::ClosedEnsembleProblem; inplace = true, optim_options = Optim.Options())
#     if inplace
#         ensemble_GRAPE!(ens, optim_options)
#     else
#         ensemble_sGRAPE(ens, optim_options)
#     end
# end

# """
# GRAPE!

# This function solves the single problem using Optim and the gradient defined for the problem type.
# """
# 
# """
# sGRAPE

# This function solves the single problem defined in problem using Optim and the gradient for the problem type. Use this version with static arrays for 0 allocations!
# """
# function sGRAPE(problem, optim_options)
#     # prepare some storage arrays that we will make use of throughout the computation
#     Ut = Vector{typeof(problem.A)}(undef, problem.n_timeslices + 1)
#     Lt = Vector{typeof(problem.A)}(undef, problem.n_timeslices + 1)
#     gradient = zeros(problem.n_controls, problem.n_timeslices)

#     function _to_optim!(F, G, x)
#         grad = @views gradient[:, :]
#         fom = _fom_and_gradient_sGRAPE(problem.A, problem.B, x, problem.n_timeslices, problem.duration, problem.n_controls, grad, Ut, Lt, problem.X_init, problem.X_target, problem)

#         if G !== nothing
#             @views G .= grad
#         end
#         if F !== nothing
#             return fom
#         end
#     end

#     init = problem.initial_guess

#     res = Optim.optimize(Optim.only_fg!(_to_optim!), init, Optim.LBFGS(), optim_options)
#     solres = SolutionResult([res], [res.minimum], [res.minimizer], problem)

#     return solres
# end


# """
# ensemble_GRAPE!

# This function solves the ensemble problem defined in ensemble_problem using the mutating GRAPE algorithm for efficiency.
# """
# function ensemble_GRAPE!(ensemble_problem, optim_options)
#     # we expand the ensemble problem into an array of single problems that we can solve
#     ensemble_problem_array = init_ensemble(ensemble_problem)

#     # to initialise the holding arrays we define a problem
#     problem = ensemble_problem_array[1]
#     # initialise holding arrays for inplace operations, these will be modified
#     state_store, costate_store, propagators, fom, gradient = init_GRAPE(problem.X_init, problem.n_timeslices, ensemble_problem.n_ensemble, problem.A, problem.n_controls)

#     evolve_store = similar(state_store[1]) .* 0.0
#     weights = ensemble_problem.weights
#     n_ensemble = ensemble_problem.n_ensemble

#     function _to_optim!(F, G, x)
#         fom = 0.0
#         for k = 1:n_ensemble
#             problem = ensemble_problem_array[k]

#             fom += @views _fom_and_gradient_GRAPE!(problem.A, problem.B, x, problem.n_timeslices, problem.duration, problem.n_controls, gradient[k, :, :], state_store[:, k], costate_store[:, k], propagators[:,k], problem.X_init, problem.X_target, evolve_store, problem) * weights[k]
#         end

#         if G !== nothing
#             # TODO is this performant?
#             @views G .= sum(gradient .* weights, dims = 1)[1,:,:]
#         end
#         if F !== nothing
#             return fom
#         end
#     end


#     init = problem.initial_guess

#     res = Optim.optimize(Optim.only_fg!(_to_optim!), init, Optim.LBFGS(), optim_options)
#     solres = SolutionResult([res], [res.minimum], [res.minimizer], problem)
#     return solres
# end


# """
# ensemble_sGRAPE

# This function solves the ensemble problem defined in ensemble_problem using the non-mutating version of the GRAPE algorithm.
# """
# function ensemble_sGRAPE(ensemble_problem, optim_options)
#     ensemble_problem_array = init_ensemble(ensemble_problem)

#     problem = ensemble_problem_array[1]

#     Ut = Vector{typeof(problem.A)}(undef, problem.n_timeslices + 1)
#     Lt = Vector{typeof(problem.A)}(undef, problem.n_timeslices + 1)

#     gradient = zeros(ensemble_problem.n_ensemble, problem.n_controls, problem.n_timeslices)

#     n_ensemble = ensemble_problem.n_ensemble
#     weights = ensemble_problem.weights

#     function _to_optim!(F, G, x)
#         fom = 0.0
#             for k = 1:n_ensemble
#                 problem = ensemble_problem_array[k]
#                 grad = @views gradient[k, :, :]

#                 fom += _fom_and_gradient_sGRAPE(problem.A, problem.B, x, problem.n_timeslices, problem.duration, problem.n_controls, grad, Ut, Lt, problem.X_init, problem.X_target, problem) * weights[k]

#             if G !== nothing
#                 @views G .= sum(gradient .* weights, dims = 1)[1,:,:]
#             end
#             if F !== nothing
#                 return fom
#             end
#         end
#     end

#     init = problem.initial_guess

#     res = Optim.optimize(Optim.only_fg!(_to_optim!), init, Optim.LBFGS(), optim_options)
#     solres = SolutionResult([res], [res.minimum], [res.minimizer], problem)

#     return solres
# end




# # """
# # Solve closed state transfer problem using ADGRAPE, this means that we need to use a piecewise evolution function that is Zygote compatible!
# # """
# # function ADGRAPE(prob::StateTransferProblem; optim_options = Optim.Options())
# #     wts = ones(prob.n_ensemble)
# #     D = size(prob.A[1])[1]
# #     u0 = typeof(prob.A[1])(I(D))

# #     function user_functional(x)
# #         fom = 0.0
# #         for k = 1:prob.n_ensemble
# #             U = pw_evolve_T(prob.A[k], prob.B[k], x, prob.n_controls, prob.duration/prob.n_timeslices, prob.n_timeslices, u0)
# #             fom += C1(prob.X_target[k], (U * prob.X_init[k] * U')) * wts[k]
# #         end
# #         fom
# #     end

# #     init = prob.initial_guess
# #     res = _ADGRAPE(user_functional, init, optim_options=optim_options)

# #     solres = SolutionResult([res], [res.minimum], [res.minimizer], prob)
# # end

# # """
# # Solve a unitary synthesis prob using ADGRAPE, this means that we need to use a piecewise evolution function that is Zygote compatible!
# # """
# # function ADGRAPE(prob::UnitaryProblem; optim_options = Optim.Options())
# #     wts = ones(prob.n_ensemble)
# #     u0 = typeof(prob.A[1])(I(D))

# #     function user_functional(x)
# #         fom = 0.0
# #         for k = 1:prob.n_ensemble
# #             U = pw_evolve_T(prob.A[k], prob.B[k], x, prob.n_controls, prob.duration/prob.n_timeslices, prob.n_timeslices, u0)
# #             fom += C1(prob.X_target[k], U * prob.X_init[k]) * wts[k]
# #         end
# #         fom
# #     end

# #     init = prob.initial_guess
# #     res = _ADGRAPE(user_functional, init, optim_options=optim_options)

# #     solres = SolutionResult([res], [res.minimum], [res.minimizer], prob)
# # end

