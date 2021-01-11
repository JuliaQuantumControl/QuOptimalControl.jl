using QuOptimalControl
using Test
using LinearAlgebra
using StaticArrays

const tol = 1e-6

const ρinit = [1.0  .0]' * [1.0 0.0+0im]
const ρfin = [0.0 1.0]' * [0.0 1.0+0im]

const Sx = [0.0 1.0; 1.0 0.0+0.0im]/2
const Sy = [0.0 -1.0im; 1.0im 0.0+0.0im]/2
const Sz = [1.0 0.0; 0.0 -1.0+0.0im]/2



const SSx = SArray{Tuple{2,2},ComplexF64}(Sx)
const SSy = SArray{Tuple{2,2},ComplexF64}(Sy)
const SSz = SArray{Tuple{2,2},ComplexF64}(Sz)


const Uinit = Array{ComplexF64,2}((I(2)))
const Ufin = 1/sqrt(2) * [0 1; 1 -0.0+0.0im]


A_gens = k -> (k - 2.5) / 2.5 * Sz * 5
B_gens = k -> [Sx, Sy]

function odd_switch(x)
    if Bool(x%2)
        return ρfin
    else
        return ρinit
    end
end

X_init_gens = k -> ρinit
X_target_gens = k -> odd_switch(k)


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
        B = [SSx, SSy],
        A = SSz,
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



@testset "StateTransfer Ensemble inplace" begin
    
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


    ens = ClosedEnsembleProblem(prob, 5, A_gens, B_gens, X_init_gens, X_target_gens, ones(5)/5)
    
    sol = GRAPE(ens, inplace=true)
    @test sol.result[1].minimum - C1(ρfin, ρfin) < tol * 10
    
end



@testset "StateTransfer Ensemble out of place" begin
    
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


    ens = ClosedEnsembleProblem(prob, 5, A_gens, B_gens, X_init_gens, X_target_gens, ones(5)/5)
    
    sol = GRAPE(ens, inplace=false)
    @test sol.result[1].minimum - C1(ρfin, ρfin) < tol * 10
    
end
