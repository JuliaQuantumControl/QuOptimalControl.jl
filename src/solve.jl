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
struct SolutionResult{R,FID,OPT,P<:Problem,A}
    result::R
    fidelity::FID
    opti_pulses::OPT
    problem::P
    alg::A
end


struct EnsembleSolutionResult{R,FID,OPT,P<:EnsembleProblem,A}
    result::R
    fidelity::FID
    opti_pulses::OPT
    problem::P
    alg::A
end



Base.@kwdef struct GRAPE{NS,IIP,OPTS}
    n_slices::NS = 1
    isinplace::IIP = true
    optim_options::OPTS = Optim.Options()
end

Base.@kwdef struct ADGRAPE{NS, opts}
    n_slices::NS = 1
    optim_options::opts = Optim.Options()
end

# Base.@kwdef struct dCRAB end

# Base.@kwdef struct ADGROUP end
default_algorithm(::Problem) = GRAPE()


solve(prob::Problem) = solve(prob, default_algorithm())


function solve(prob::Problem, alg::GRAPE)
    @unpack B, A, Xi, Xt, T, n_controls, guess, sys_type = prob
    @unpack n_slices, isinplace, optim_options = alg

    if isinplace
        state_store, costate_store, propagators, fom, gradient =
            init_GRAPE(Xi, n_slices, 1, A, n_controls)

        evolve_store = similar(state_store[1]) .* 0.0
        grad = @views gradient[1, :, :]

        topt =
            (F, G, x) -> begin
                fom = _fom_and_gradient_GRAPE!(
                    A,
                    B,
                    x,
                    n_slices,
                    T,
                    n_controls,
                    grad,
                    state_store,
                    costate_store,
                    propagators,
                    Xi,
                    Xt,
                    evolve_store,
                    sys_type,
                )

                if G !== nothing
                    @views G .= grad
                end
                if F !== nothing
                    return fom
                end
            end


    else
        # we could move this into another function
        Ut = Vector{typeof(A)}(undef, n_slices + 1)
        Lt = Vector{typeof(A)}(undef, n_slices + 1)
        gradient = zeros(n_controls, n_slices)

        topt =
            (F, G, x) -> begin
                grad = @views gradient[:, :]
                fom = _fom_and_gradient_sGRAPE(
                    A,
                    B,
                    x,
                    n_slices,
                    T,
                    n_controls,
                    grad,
                    Ut,
                    Lt,
                    Xi,
                    Xt,
                    sys_type,
                )

                if G !== nothing
                    @views G .= grad
                end
                if F !== nothing
                    return fom
                end
            end

    end

    init = guess
    res = Optim.optimize(Optim.only_fg!(topt), init, Optim.LBFGS(), optim_options)
    sol = SolutionResult(res, res.minimum, res.minimizer, prob, alg)

    return sol

end

function solve(ens_prob::EnsembleProblem, alg::GRAPE)
    @unpack prob, n_ens, A_g, B_g, XiG, XtG, wts = ens_prob
    @unpack n_slices, isinplace, optim_options = alg

    ensemble_problem_array = init_ensemble(ens_prob)

    # to initialise the holding arrays we define a problem
    problem = ensemble_problem_array[1]
    # @unpack B, A, Xi, Xt, T, n_controls, guess, sys_type = prob

    if isinplace

        # initialise holding arrays for inplace operations, these will be modified
        state_store, costate_store, propagators, fom, gradient = init_GRAPE(problem.Xi, n_slices, n_ens, problem.A, problem.n_controls)

        evolve_store = similar(state_store[1]) .* 0.0
        
        topt = function(F, G, x)
            fom = 0.0
            for k = 1:n_ens
                problem = ensemble_problem_array[k]

                # what is the performance hit of @unpack?
                @unpack B, A, Xi, Xt, T, n_controls, guess, sys_type = problem
                fom += @views _fom_and_gradient_GRAPE!(A, B, x, n_slices, T, n_controls, gradient[k, :, :], state_store[:, k], costate_store[:, k], propagators[:,k], Xi, Xt, evolve_store, sys_type) * wts[k]
            end

            if G !== nothing
                # TODO is this performant?
                @views G .= sum(gradient .* wts, dims = 1)[1,:,:]
            end
            if F !== nothing
                return fom
            end
        end
    else
            
        Ut = Vector{typeof(problem.A)}(undef, n_slices + 1)
        Lt = Vector{typeof(problem.A)}(undef, n_slices + 1)

        gradient = zeros(ens_prob.n_ens, problem.n_controls, n_slices)

        topt = function(F,G,x)
            fom = 0.0
                for k = 1:n_ens
                    problem = ensemble_problem_array[k]

                    @unpack B, A, Xi, Xt, T, n_controls, guess, sys_type = problem

                    grad = @views gradient[k, :, :]
    
                    fom += _fom_and_gradient_sGRAPE(A, B, x, n_slices, T, n_controls, grad, Ut, Lt, Xi, Xt, sys_type) * wts[k]
    
                if G !== nothing
                    @views G .= sum(gradient .* wts, dims = 1)[1,:,:]
                end
                if F !== nothing
                    return fom
                end
            end
        end
    

    end


    init = problem.guess

    res = Optim.optimize(Optim.only_fg!(topt), init, Optim.LBFGS(), optim_options)
    sol = EnsembleSolutionResult(res, res.minimum, res.minimizer, ens_prob, alg)

    return sol


end

# for ADGRAPE we need to dispatch on the system type too
function solve(prob::Problem, alg::ADGRAPE)
    # @unpack B, A, Xi, Xt, T, n_controls, guess, sys_type = prob
    @unpack n_slices, optim_options = alg

    func = _get_functional(prob, alg.n_slices, prob.sys_type)

    init = prob.guess
    res = _ADGRAPE(func, init, optim_options)
    sol = SolutionResult(res, res.minimum, res.minimizer, prob, alg)
    return sol
end

function _get_functional(prob, n_slices, sys_type::StateTransfer)
    @unpack B, A, Xi, Xt, T, n_controls, guess, sys_type = prob
    D = size(A, 1)
    u0 = typeof(A)(I(D))
    function functional(x)
        U = pw_evolve_T(A, B, x, n_controls, T/n_slices, n_slices, u0)
        ev = U * Xi * U'
        return C1(Xt, ev)
    end
    return functional
end

function solve(prob::EnsembleProblem, alg::ADGRAPE)
    @unpack n_slices, optim_options = alg
    ensemble_problem_array = init_ensemble(prob)

    func = _get_ensemble_functional(ensemble_problem_array, prob.n_ens, n_slices, prob.wts, ensemble_problem_array[1].sys_type)

    init = ensemble_problem_array[1].guess
    res = _ADGRAPE(func, init, optim_options)
    sol = EnsembleSolutionResult(res, res.minimum, res.minimizer, prob, alg)
    return sol
end

function _get_ensemble_functional(ens_prob_arr, n_ens, n_slices, wts, sys_type::StateTransfer)

    A = ens_prob_arr[1].A
    D = size(A, 1)
    u0 = typeof(A)(I(D))

    function functional(x)
        fom = 0.0
        for k = 1:n_ens
            # for a specific ensemble problem
            @unpack B, A, Xi, Xt, T, n_controls, guess, sys_type = ens_prob_arr[k]
            
            U = pw_evolve_T(A, B, x, n_controls, T/n_slices, n_slices, u0)
            ev = U * Xi * U'
            err = C1(Xt, ev) * wts[k]
            fom = fom + err
        end
        return fom
    end
    return functional
end

println("stop")





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
