# test idea of Liouville evolution


using LinearAlgebra
using QuantumInformation

H_ctrl = [pi * sx, pi * sy]
H0 = sz

function to_superoperator(H)
    D = size(H)[1]
    kron(I(D), H)  - kron(H', I(D))
end



H0 = to_superoperator(sz)
H_ctrls = to_superoperator.(H_ctrl)
duration = 1
n_slices = 10
drive = rand(2, n_slices)

function prop(x)
    Htotal = H0
    for k = 1:2
        Htotal = Htotal + x[k] * H_ctrls[k]
    end
    U = exp(-im * Htotal * dt)
end


p = []
for s = 1:n_slices
    p_n = prop(drive[:,s])

    append!(p, [p_n])
end


psi = [1;0.0 + 0.0im]

rho0 = psi * psi'
# stack it into vector
rho0 = reshape(rho0, 4)

o = [rho0]
for s = 1:n_slices
    o_n = p[s] * o[s]
    append!(o, [o_n])
end

obs = to_superoperator(sz)
z = []
for s = 1:n_slices
    # temp = obs * o[s]
    kk = reshape(o[s], (2,2))
    out = real(tr(sz * kk))
    append!(z, out)
end
using Plots
plot(z)

##

using FastClosures

prob = StateTransferProblem(
    B = [Sx, Sy],
    A = Sz,
    X_init = ρinit,
    X_target = ρfin,
    duration = 5.0,
    n_timeslices = 25,
    n_controls = 2,
    initial_guess = rand(2, 25)
)


ensemble_problem = ClosedEnsembleProblem(prob, 5, A_gens, B_gens, X_init_gens, X_target_gens, ones(5)/5)

# sol = GRAPE(ens, inplace=true)
# @test sol.result[1].minimum - C1(ρfin, ρfin) < tol * 10




ensemble_problem_array = QuOptimalControl.init_ensemble(ensemble_problem)
problem = ensemble_problem_array[1]

state_store, costate_store, generators, propagators, fom, gradient = init_GRAPE(problem.X_init, problem.n_timeslices, ensemble_problem.n_ensemble, problem.A, problem.n_controls)

evolve_store = similar(state_store[1])
weights = ensemble_problem.weights
n_ensemble = ensemble_problem.n_ensemble


topt = ensemble_GRAPE!(ensemble_problem, nothing)

using BenchmarkTools

topt(1.0, nothing, rand(25, 2))

@code_warntype ensemble_GRAPE!(ensemble_problem, nothing)


function _fom_and_gradient_GRAPE_unchanged(A::T, B, control_array, n_timeslices, duration, n_controls, gradient, fwd_state_store, bwd_costate_store, generators, propagators, X_init, X_target, evolve_store, problem) where T
    
    dt = duration / n_timeslices
    
    fwd_state_store[1] .= X_init
    bwd_costate_store[end] .= X_target
    # this seems really stupid since we dont ever use the generators again, we can just keep the propagators instead
    pw_ham_save!(A, B, control_array, n_controls, n_timeslices, @view generators[:])
    @views propagators[:] .= exp.(generators[:] .* (-1.0im * dt))

    for t = 1:n_timeslices
        QuOptimalControl.evolve_func!(problem, t, fwd_state_store, bwd_costate_store, propagators, generators, evolve_store, forward = true)
    end

    for t = reverse(1:n_timeslices)
        QuOptimalControl.evolve_func!(problem, t, fwd_state_store, bwd_costate_store, propagators, generators, evolve_store, forward = false)
    end

    t = n_timeslices
    
    for c = 1:n_controls
        for t = 1:n_timeslices
            @views gradient[c, t] = QuOptimalControl.grad_func!(problem, t, dt, B[c], fwd_state_store, bwd_costate_store, propagators, generators, evolve_store)
        end
    end

    return QuOptimalControl.fom_func(problem, t, fwd_state_store, bwd_costate_store, propagators, generators)

end

k = 1

x = rand(2,25)


state_store, costate_store, generators, propagators, ofom, gradient = init_GRAPE(prob.X_init, prob.n_timeslices, ensemble_problem.n_ensemble, prob.A, prob.n_controls)

_fom_and_gradient_GRAPE_unchanged(prob.A, prob.B, x, prob.n_timeslices, prob.duration, prob.n_controls, gradient[k, :, :], state_store[:, k], costate_store[:, k], generators[:,k], propagators[:,k], prob.X_init, prob.X_target, evolve_store, prob)

@benchmark _fom_and_gradient_GRAPE_unchanged($prob.A, $prob.B, $x, $prob.n_timeslices, $prob.duration, $prob.n_controls, $gradient[k, :, :], $state_store[:, k], $costate_store[:, k], $generators[:,k], $propagators[:,k], $prob.X_init, $prob.X_target, $evolve_store, $prob)


@benchmark _fom_and_gradient_GRAPE_updated($prob.A, $prob.B, $x, $prob.n_timeslices, $prob.duration, $prob.n_controls, $gradient[k, :, :], $state_store[:, k], $costate_store[:, k], $generators[:,k], $propagators[:,k], $prob.X_init, $prob.X_target, $evolve_store, $prob)


@code_warntype _fom_and_gradient_GRAPE_updated(prob.A, prob.B, x, prob.n_timeslices, prob.duration, prob.n_controls, gradient[k, :, :], state_store[:, k], costate_store[:, k], generators[:,k], propagators[:,k], prob.X_init, prob.X_target, evolve_store, prob)




function _fom_and_gradient_GRAPE_updated(A::T, B, control_array, n_timeslices, duration, n_controls, gradient, fwd_state_store, bwd_costate_store, generators, propagators, X_init, X_target, evolve_store, problem) where T
    
    dt = duration / n_timeslices
    
    fwd_state_store[1] .= X_init
    bwd_costate_store[end] .= X_target
    # this seems really stupid since we dont ever use the generators again, we can just keep the propagators instead
    # pw_ham_save!(A, B, control_array, n_controls, n_timeslices, @view generators[:])
    # @views propagators[:] .= exp.(generators[:] .* (-1.0im * dt))
    propagators[:] .= QuOptimalControl.pw_evolve_save_new(A, B, control_array, n_controls, dt, n_timeslices)

    for t = 1:n_timeslices
        QuOptimalControl.evolve_func!(problem, t, fwd_state_store, bwd_costate_store, propagators, generators, evolve_store, forward = true)
    end

    for t = reverse(1:n_timeslices)
        QuOptimalControl.evolve_func!(problem, t, fwd_state_store, bwd_costate_store, propagators, generators, evolve_store, forward = false)
    end

    t = n_timeslices
    
    for c = 1:n_controls
        for t = 1:n_timeslices
            @views gradient[c, t] = QuOptimalControl.grad_func!(problem, t, dt, B[c], fwd_state_store, bwd_costate_store, propagators, generators, evolve_store)
        end
    end

    return QuOptimalControl.fom_func(problem, t, fwd_state_store, bwd_costate_store, propagators, generators)

end





function ensemble_GRAPE!(ensemble_problem, optim_options)
    # we expand the ensemble problem into an array of single problems that we can solve
    ensemble_problem_array = QuOptimalControl.init_ensemble(ensemble_problem)
    
    # to initialise the holding arrays we define a problem
    problem = ensemble_problem_array[1]
    # initialise holding arrays for inplace operations, these will be modified
    state_store, costate_store, generators, propagators, ofom, gradient = init_GRAPE(problem.X_init, problem.n_timeslices, ensemble_problem.n_ensemble, problem.A, problem.n_controls)
    
    evolve_store = similar(state_store[1])
    weights = ensemble_problem.weights
    n_ensemble = ensemble_problem.n_ensemble

    function _to_optim!(F, G, x)
        fom = 0.0
        for k = 1:n_ensemble
            curr_prob = ensemble_problem_array[k]
            
            fom += @views QuOptimalControl._fom_and_gradient_GRAPE!(curr_prob.A, curr_prob.B, x, curr_prob.n_timeslices, curr_prob.duration, curr_prob.n_controls, gradient[k, :, :], state_store[:, k], costate_store[:, k], generators[:,k], propagators[:,k], curr_prob.X_init, curr_prob.X_target, evolve_store, curr_prob) * weights[k]
        end
        
        if G !== nothing
            @views G .= sum(gradient .* weights, dims = 1)[1,:,:]
        end
        if F !== nothing
            return fom
        end
    end
    return _to_optim!

    # init = problem.initial_guess

    # res = Optim.optimize(Optim.only_fg!(_to_optim!), init, Optim.LBFGS(), optim_options)
    # solres = SolutionResult([res], [res.minimum], [res.minimizer], problem)
    # return solres
end

prob = ensemble_problem.problem


