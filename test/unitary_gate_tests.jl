
# tests for problems and ensemble problem types
@testset "Unitary X gate pulse in place w/ GRAPE" begin

    prob = Problem(
        B = [Sx, Sy],
        A = Sz,
        Xi = Uinit,
        Xt = Ufin,
        T = 1.0,
        n_controls = 2,
        guess = rand(2, 10),
        sys_type = UnitaryGate(),
    )

    sol = solve(prob, GRAPE(n_slices = 10, isinplace = true))

    @test sol.result.minimum - C1(Ufin, Ufin) < tol
end

@testset "Unitary X gate pulse static arrays" begin

    prob = Problem(
        B = [SSx, SSy],
        A = SSz,
        Xi = UinitS,
        Xt = UfinS,
        T = 1.0,
        n_controls = 2,
        guess = rand(2, 10),
        sys_type = UnitaryGate(),
    )

    sol = solve(prob, GRAPE(n_slices = 10, isinplace = false))
    @test sol.result.minimum - C1(UfinS, UfinS) < tol

end



@testset "Robust X gate pulse inplace" begin

    prob = Problem(
        B = [Sx, Sy],
        A = Sz,
        Xi = Uinit,
        Xt = Ufin,
        T = 5.0,
        n_controls = 2,
        guess = rand(2, 100),
        sys_type = UnitaryGate(),
    )


    ens = EnsembleProblem(
        prob = prob,
        n_ens = 5,
        A_g = A_gens,
        B_g = B_gens,
        XiG = U_init_gens,
        XtG = U_target_gens,
        wts = ones(5) / 5,
    )

    sol = solve(
        ens,
        GRAPE(
            n_slices = 100,
            isinplace = true,
            optim_options = Optim.Options(f_tol = 1e-3),
        ),
    )
    @test sol.result.minimum - C1(ρfin, ρfin) < tol

end


@testset "Robust X gate pulse inplace static arrays" begin

    prob = Problem(
        B = [SSx, SSy],
        A = SSz,
        Xi = UinitS,
        Xt = UfinS,
        T = 10.0,
        n_controls = 2,
        guess = rand(2, 100),
        sys_type = UnitaryGate(),
    )


    ens = EnsembleProblem(
        prob = prob,
        n_ens = 5,
        A_g = A_gens_static,
        B_g = B_gens_static,
        XiG = U_init_gens_static,
        XtG = U_target_gens_static,
        wts = ones(5) / 5,
    )

    sol = solve(
        ens,
        GRAPE(
            n_slices = 100,
            isinplace = false,
            optim_options = Optim.Options(f_tol = 1e-3),
        ),
    )
    @test sol.result.minimum - C1(UinitS, UfinS) < tol

end


@testset "Unitary X gate pulse ADGRAPE" begin

    prob = Problem(
        B = [Sx, Sy],
        A = 0.0 * Sz,
        Xi = Uinit,
        Xt = Ufin,
        T = 1.0,
        n_controls = 2,
        guess = rand(2, 25),
        sys_type = UnitaryGate(),
    )

    sol = solve(prob, ADGRAPE(n_slices = 25))

    pw_evolve(prob.A, prob.B, sol.opti_pulses, 2, 1.0 / 25, 25, Uinit)

    @test sol.result.minimum - C1(Ufin, Ufin) < tol
end



@testset "Robust X gate pulse ADGRAPE" begin

    prob = Problem(
        B = [Sx, Sy],
        A = Sz,
        Xi = Uinit,
        Xt = Ufin,
        T = 5.0,
        n_controls = 2,
        guess = rand(2, 100),
        sys_type = UnitaryGate(),
    )


    ens = EnsembleProblem(
        prob = prob,
        n_ens = 5,
        A_g = A_gens,
        B_g = B_gens,
        XiG = U_init_gens,
        XtG = U_target_gens,
        wts = ones(5) / 5,
    )

    sol = solve(ens, ADGRAPE(n_slices = 100))

    pw_evolve(prob.A, prob.B, sol.opti_pulses, 2, 5 / 100, 100, Uinit)

    @test sol.result.minimum - C1(ρfin, ρfin) < tol

end
