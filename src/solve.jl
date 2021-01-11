using Optim
using FileWatching
using DelimitedFiles

"""
Contains the solve function interfaces for now, until I learn a better way to do it
"""
function GRAPE(prob::Union{StateTransferProblem,UnitaryProblem}; inplace = true, optim_options=Optim.Options())
    if inplace
        GRAPE!(prob, optim_options)
    else
        sGRAPE(prob, optim_options)
    end
end

"""
Handles solving ensemble problems with GRAPE
"""
function GRAPE(ens::ClosedEnsembleProblem, inplace = true, optim_options = Optim.Options())
end

"""
GRAPE!

This function solves the single problem using Optim and the gradient defined for the problem type.
"""
function GRAPE!(problem, optim_options = Optim.Options())
    
    # initialise holding arrays for inplace operations, these will be modified
    state_store, costate_store, generators, propagators, fom, gradient = init_GRAPE(problem.X_init, problem.n_timeslices, 1, problem.A, problem.n_controls)
    
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

    res = Optim.optimize(Optim.only_fg!(_to_optim!), init, Optim.LBFGS(), optim_options)
    solres = SolutionResult([res], [res.minimum], [res.minimizer], problem)
    return solres
end

"""
sGRAPE

This function solves the single problem defined in problem using Optim and the gradient for the problem type. Use this version with static arrays for 0 allocations!
"""
function sGRAPE(problem, optim_options = Optim.Options())
    # prepare some storage arrays that we will make use of throughout the computation
    Ut = Vector{typeof(problem.A)}(undef, problem.n_timeslices + 1)
    Lt = Vector{typeof(problem.A)}(undef, problem.n_timeslices + 1)
    gradient = zeros(problem.n_controls, problem.n_timeslices)

    function _to_optim!(F, G, x)
        grad = @views gradient[:, :]
        fom = _fom_and_gradient_sGRAPE(problem.A, problem.B, x, problem.n_timeslices, problem.duration, problem.n_controls, grad, Ut, Lt, problem.X_init, problem.X_target, problem)
        
        if G !== nothing
            @views G .= grad
        end
        if F !== nothing
            return fom
        end
    end

    init = problem.initial_guess

    res = Optim.optimize(Optim.only_fg!(_to_optim!), init, Optim.LBFGS(), optim_options)
    solres = SolutionResult([res], [res.minimum], [res.minimizer], problem)

    return solres
end

"""
Solve closed state transfer problem using ADGRAPE, this means that we need to use a piecewise evolution function that is Zygote compatible!
"""
function ADGRAPE(prob::StateTransferProblem; optim_options = Optim.Options())
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
function ADGRAPE(prob::UnitaryProblem; optim_options = Optim.Options())
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

