using QuOptimalControl
using Test
using LinearAlgebra


const tol = 1e-6

const ρinit = [1.0  .0]' * [1.0 0.0+0im]
const ρfin = [0.0 1.0]' * [0.0 1.0+0im]

const Sx = [0.0 1.0; 1.0 0.0+0.0im]/2
const Sy = [0.0 -1.0im; 1.0im 0.0+0.0im]/2
const Sz = [1.0 0.0; 0.0 -1.0+0.0im]/2

const Uinit = Array{ComplexF64,2}((I(2)))
const Ufin = 1/sqrt(2) * [0 1; 1 -0.0+0.0im]


@testset "StateTransfer in place w/ GRAPE" begin
    
    prob = StateTransferProblem(
        B = [Sx, Sy],
        A = Sz,
        X_init = ρinit,
        X_target = ρfin,
        duration = 1.0,
        n_timeslices = 10,
        n_controls = 2,
        initial_guess = rand(2, 10)
    )

    sol = GRAPE(prob)
    @test sol.result[1].minimum - C1(ρfin, ρfin) < tol
    
end


@testset "StateTransfer GRAPE" begin
    
    prob = StateTransferProblem(
        B = [Sx, Sy],
        A = Sz,
        X_init = ρinit,
        X_target = ρfin,
        duration = 1.0,
        n_timeslices = 10,
        n_controls = 2,
        initial_guess = rand(2, 10)
    )

    sol = GRAPE(prob, inplace=false)
    @test sol.result[1].minimum - C1(ρfin, ρfin) < tol
    
end




# prob = UnitarySynthesis(
#     B = [[Sx, Sy]],
#     A = [Sz],
#     X_init = [Uinit],
#     X_target = [Ufin],
#     duration = 10.0,
#     n_timeslices = 100,
#     n_controls = 2,
#     n_ensemble = 1,
#     initial_guess = rand(2, 100)
# )

# sol = GRAPE(prob)

# # check if the result is close to the target or not