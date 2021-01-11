using QuOptimalControl
using Test
using LinearAlgebra

@testset "GRAPE" begin
    tol = 1e-6

    ρinit = [1.0  .0]' * [1.0 0.0+0im]
    ρfin = [0.0 1.0]' * [0.0 1.0+0im]
    
    Sx = [0.0 1.0; 1.0 0.0+0.0im]/2
    Sy = [0.0 -1.0im; 1.0im 0.0+0.0im]/2
    Sz = [1.0 0.0; 0.0 -1.0+0.0im]/2
    
    prob = StateTransferProblem(
        B = [Sx, Sy],
        A = Sz,
        X_init = [ρinit],
        X_target = [ρfin],
        duration = 1.0,
        n_timeslices = 10,
        n_controls = 2,
        initial_guess = rand(2, 10)
    )

    sol = GRAPE(prob)
    @test sol.result[1].minimum - C1(ρfin, ρfin) < tol

    Uinit = Array{ComplexF64,2}((I(2)))
    Ufin = 1/sqrt(2) * [0 1; 1 -0.0+0.0im]
    


    prob = UnitarySynthesis(
        B = [[Sx, Sy]],
        A = [Sz],
        X_init = [Uinit],
        X_target = [Ufin],
        duration = 10.0,
        n_timeslices = 100,
        n_controls = 2,
        n_ensemble = 1,
        initial_guess = rand(2, 100)
    )

    sol = GRAPE(prob)

    # check if the result is close to the target or not
    
end