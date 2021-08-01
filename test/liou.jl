# test idea of Liouville evolution
using LinearAlgebra
using QuantumInformation

H_ctrl = [pi * sx, pi * sy]
H0 = sz

function to_superoperator(H)
    D = size(H)[1]
    kron(I(D), H) - kron(H', I(D))
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
    p_n = prop(drive[:, s])

    append!(p, [p_n])
end


psi = [1; 0.0 + 0.0im]

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
    kk = reshape(o[s], (2, 2))
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
    initial_guess = rand(2, 25),
)


ensemble_problem =
    ClosedEnsembleProblem(prob, 5, A_gens, B_gens, X_init_gens, X_target_gens, ones(5) / 5)

# sol = GRAPE(ens, inplace=true)
# @test sol.result[1].minimum - C1(ρfin, ρfin) < tol * 10




function ensemble_GRAPE!_unchanged(ensemble_problem, optim_options)
    # we expand the ensemble problem into an array of single problems that we can solve
    ensemble_problem_array = QuOptimalControl.init_ensemble(ensemble_problem)

    # to initialise the holding arrays we define a problem
    problem = ensemble_problem_array[1]
    # initialise holding arrays for inplace operations, these will be modified
    state_store, costate_store, generators, propagators, ofom, gradient = init_GRAPE(
        problem.X_init,
        problem.n_timeslices,
        ensemble_problem.n_ensemble,
        problem.A,
        problem.n_controls,
    )

    evolve_store = similar(state_store[1])
    weights = ensemble_problem.weights
    n_ensemble = ensemble_problem.n_ensemble

    function _to_optim!(F, G, x)
        fom = 0.0
        for k = 1:n_ensemble
            curr_prob = ensemble_problem_array[k]

            fom += @views QuOptimalControl._fom_and_gradient_GRAPE!(
                curr_prob.A,
                curr_prob.B,
                x,
                curr_prob.n_timeslices,
                curr_prob.duration,
                curr_prob.n_controls,
                gradient[k, :, :],
                state_store[:, k],
                costate_store[:, k],
                generators[:, k],
                propagators[:, k],
                curr_prob.X_init,
                curr_prob.X_target,
                evolve_store,
                curr_prob,
            ) * weights[k]
        end

        if G !== nothing
            @views G .= sum(gradient .* weights, dims = 1)[1, :, :]
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

prob = ensemble_problem







@code_warntype ensemble_GRAPE!_unchanged(ensemble_problem, 1.0)

topt = ensemble_GRAPE!_unchanged(ensemble_problem, 1.0)

@code_warntype topt(nothing, nothing, rand(2, 25))


@benchmark topt($nothing, $nothing, $rand(2, 25))



function ensemble_GRAPE!closure(ensemble_problem, optim_options)
    # we expand the ensemble problem into an array of single problems that we can solve
    ensemble_problem_array = QuOptimalControl.init_ensemble(ensemble_problem)

    # to initialise the holding arrays we define a problem
    problem = ensemble_problem_array[1]
    # initialise holding arrays for inplace operations, these will be modified
    state_store, costate_store, generators, propagators, ofom, gradient = init_GRAPE(
        problem.X_init,
        problem.n_timeslices,
        ensemble_problem.n_ensemble,
        problem.A,
        problem.n_controls,
    )

    evolve_store = similar(state_store[1])
    weights = ensemble_problem.weights
    n_ensemble = ensemble_problem.n_ensemble

    function _to_optim!(F, G, x)
        fom = 0.0
        for k = 1:n_ensemble
            curr_prob = ensemble_problem_array[k]

            fom += @views QuOptimalControl._fom_and_gradient_GRAPE!(
                curr_prob.A,
                curr_prob.B,
                x,
                curr_prob.n_timeslices,
                curr_prob.duration,
                curr_prob.n_controls,
                gradient[k, :, :],
                state_store[:, k],
                costate_store[:, k],
                generators[:, k],
                propagators[:, k],
                curr_prob.X_init,
                curr_prob.X_target,
                evolve_store,
                curr_prob,
            ) * weights[k]
        end

        if G !== nothing
            @views G .= sum(gradient .* weights, dims = 1)[1, :, :]
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


@code_warntype ensemble_GRAPE!closure(ensemble_problem, 1.0)

toptclosure = ensemble_GRAPE!closure(ensemble_problem, 1.0)


@code_warntype toptclosure(nothing, nothing, rand(2, 25))



function GRAPEfix!(problem, optim_options)

    # initialise holding arrays for inplace operations, these will be modified
    state_store, costate_store, generators, propagators, fom, gradient =
        init_GRAPE(problem.X_init, problem.n_timeslices, 1, problem.A, problem.n_controls)

    evolve_store = similar(state_store[1])
    k = 1
    function _to_optim!(F, G, x)
        fom = 0.0
        fom += @views _fom_and_gradient_GRAPE!(
            problem.A,
            problem.B,
            x,
            problem.n_timeslices,
            problem.duration,
            problem.n_controls,
            gradient[k, :, :],
            state_store[:, k],
            costate_store[:, k],
            generators[:, k],
            propagators[:, k],
            problem.X_init,
            problem.X_target,
            evolve_store,
            problem,
        )
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

using Optim

GRAPEfix!(prob, 1.0)


problem = prob
# initialise holding arrays for inplace operations, these will be modified
state_store, costate_store, generators, propagators, fom, gradient =
    init_GRAPE(problem.X_init, problem.n_timeslices, 1, problem.A, problem.n_controls)

evolve_store = similar(state_store[1])
k = 1
function _to_optim!(F, G, x)
    fom = 0.0
    fom += @views QuOptimalControl._fom_and_gradient_GRAPE!(
        problem.A,
        problem.B,
        x,
        problem.n_timeslices,
        problem.duration,
        problem.n_controls,
        gradient[k, :, :],
        state_store[:, k],
        costate_store[:, k],
        generators[:, k],
        propagators[:, k],
        problem.X_init,
        problem.X_target,
        evolve_store,
        problem,
    )
    if G !== nothing
        @views G .= gradient[1, :, :]
    end
    if F !== nothing
        return fom
    end
end


init = problem.initial_guess

res = Optim.optimize(Optim.only_fg!(_to_optim!), init, Optim.LBFGS())
solres = SolutionResult([res], [res.minimum], [res.minimizer], problem)
