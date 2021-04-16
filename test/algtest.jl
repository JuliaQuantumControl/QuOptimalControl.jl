
prob = StateTransferProblem(
    B = [SSx, SSy],
    A = SSz,
    X_init = ρinitS,
    X_target = ρfinS,
    duration = 5.0,
    n_timeslices = 25,
    n_controls = 2,
    initial_guess = rand(2, 25)
)


ens = ClosedEnsembleProblem(prob, 5, A_gens_static, B_gens_static, X_init_gens_static, X_target_gens_static, ones(5)/5)

sol = GRAPE(ens, inplace=false)
@test sol.result[1].minimum - C1(ρfin, ρfin) < tol * 10
