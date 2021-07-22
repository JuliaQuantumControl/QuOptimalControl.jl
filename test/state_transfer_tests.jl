
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
        prob = prob,
        n_ens = 5,
        A_g = A_gens,
        B_g = B_gens,
        XiG = X_init_gens,
        XtG = X_target_gens,
        wts = ones(5) / 5,
    )

    sol = solve(ens, GRAPE(n_slices = 25, isinplace = true))
    @test sol.result.minimum - C1(ρfin, ρfin) < tol * 10

end



@testset "StateTransfer Ensemble out of place" begin
    
    prob = Problem(
        B = [SSx, SSy],
        A = SSz,
        Xi = ρinitS,
        Xt = ρfinS,
        T = 5.0,
        n_controls = 2,
        guess = rand(2, 25),
        sys_type = StateTransfer()
    )


    ens = EnsembleProblem(
        prob = prob,
        n_ens = 5,
        A_g = A_gens_static,
        B_g = B_gens_static,
        XiG = X_init_gens_static,
        XtG = X_target_gens_static,
        wts = ones(5) / 5,
    )

    sol = solve(ens, GRAPE(n_slices = 25, isinplace = false))
    @test sol.result.minimum - C1(ρfin, ρfin) < tol * 10

end


