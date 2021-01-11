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



@testset "StateTransfer Ensemble inplace" begin
    
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

using Setfield

A_gens = k -> k * Sz
B_gens = k -> [Sx, Sy]
X_init_gens = k -> ρinit
X_target_gens = k -> ρfin
ens = ClosedEnsembleProblem(prob, 5, A_gens, B_gens, ones(5)/5)

prob_arr = [deepcopy(prob) for k = 1:ens.n_ensemble]
for k = 1:ens.n_ensemble
    prob_to_update = prob_arr[k]
    prob_to_update = @set prob_to_update.A = ens.A_generators(k) 
    prob_to_update = @set prob_to_update.B = ens.B_generators(k)
    prob_to_update = @set prob_to_update.X_init = X_init_gens(k)
    prob_to_update = @set prob_to_update.X_target = X_target_gens(k)
    prob_arr[k] = prob_to_update
    @show prob_arr[k].A
end

# now we initialise the storage arrays and then do everything as I would like. and it'll be done

function _to_optim!(F, G, x)
    
end

prob_k = deepcopy(prob)
prob_k.A[:] = A_gens(5)

# generate an array of problems and weights


# ok lets do the optimisation now!
function prep_ensemble_problem(ensemble_problem)
    
end

using StaticArrays

SSx = SArray{Tuple{2,2},ComplexF64}(Sx)
SSy = SArray{Tuple{2,2},ComplexF64}(Sy)
SSz = SArray{Tuple{2,2},ComplexF64}(Sz)

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