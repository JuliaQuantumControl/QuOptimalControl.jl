
# tests for problems and ensemble problem types

@testset "StateTransfer in place w/ GRAPE" begin

    prob = Problem(
        B = [Sx, Sy],
        A = Sz,
        Xi = ρinit,
        Xt = ρfin,
        T = 1.0,
        n_controls = 2,
        guess = rand(2, 10),
        sys_type = StateTransfer(),
    )

    sol = solve(prob, GRAPE(n_slices = 10, isinplace = true))

    @test sol.result.minimum - C1(ρfin, ρfin) < tol
end

@testset "StateTransfer GRAPE" begin

    prob = Problem(
        B = [SSx, SSy],
        A = SSz,
        Xi = ρinit,
        Xt = ρfin,
        T = 1.0,
        n_controls = 2,
        guess = rand(2, 10),
        sys_type = StateTransfer(),
    )

    sol = solve(prob, GRAPE(n_slices = 10, isinplace = false))
    @test sol.result.minimum - C1(ρfin, ρfin) < tol

end



@testset "StateTransfer Ensemble inplace" begin

    prob = Problem(
        B = [Sx, Sy],
        A = Sz,
        Xi = ρinit,
        Xt = ρfin,
        T = 5.0,
        n_controls = 2,
        guess = rand(2, 25),
        sys_type = StateTransfer()
    )


    ens = EnsembleProblem(
        problem = prob,
        n_ensemble = 5,
        A_generators = A_gens,
        B_generators = B_gens,
        X_init_generators = X_init_gens,
        X_target_generators = X_target_gens,
        weights = ones(5) / 5,
    )

    sol = GRAPE(ens, inplace = true)
    @test sol.result[1].minimum - C1(ρfin, ρfin) < tol * 10

end



@testset "StateTransfer Ensemble out of place" begin
    prob = StateTransferProblem(
        B = [SSx, SSy],
        A = SSz,
        X_init = ρinitS,
        X_target = ρfinS,
        duration = 5.0,
        n_timeslices = 25,
        n_controls = 2,
        initial_guess = rand(2, 25),
    )


    ens = ClosedEnsembleProblem(
        prob,
        5,
        A_gens_static,
        B_gens_static,
        X_init_gens_static,
        X_target_gens_static,
        ones(5) / 5,
    )

    sol = GRAPE(ens, inplace = false)
    @test sol.result[1].minimum - C1(ρfin, ρfin) < tol * 10

end
