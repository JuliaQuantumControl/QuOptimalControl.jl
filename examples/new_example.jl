
using QuOptimalControl

using QuantumInformation

using LinearAlgebra

i2 = Matrix{ComplexF64}(I(2))

problem = UnitarySynthesis(
    [sx, sy],
    [sz],
    [i2], 
    [sz],
    1,
    0.1, 
    10,
    2,
    1,
    1.0,
    nothing
)

U, L, G_array, P_array, g, grad = init_GRAPE(problem.X_init[1], problem.timeslices, problem.n_ensemble, problem.H_drift[1], problem.number_pulses)


input = rand(2, 10)
save_grad = similar(input)
GRAPE_new(1.0, save_grad, input, U, L, G_array, P_array, g, grad, problem.H_drift, problem.H_ctrl, problem.timeslices, problem.n_ensemble, problem.duration, problem.number_pulses, problem)


ψ = Array{ComplexF64,1}([1, 0])

ρ = ψ * ψ'

ψ = Array{ComplexF64,1}([0, 1])
ρt = ψ * ψ'

problem = ClosedStateTransfer(
    [sx, sy],
    [-1.0 * sz, 1.0 * sz],
    [ρ], 
    [ρt],
    1,
    0.1, 
    10,
    2,
    1,
    1.0,
    nothing,
)

U, L, G_array, P_array, g, grad = init_GRAPE(problem.X_init[1], problem.timeslices, problem.n_ensemble, problem.H_drift[1], problem.number_pulses)

using BenchmarkTools
@code_warntype GRAPE_new(1.0, save_grad, input, U, L, G_array, P_array, g, grad, problem.H_drift, problem.H_ctrl, problem.timeslices, problem.n_ensemble, problem.duration, problem.number_pulses, problem)



using Plots

z1 = real.([tr(sz * x) for x in U])
z2 = real.([tr(sz * x) for x in L])

plot(z1)
plot!(z2)



test = (F, G, x) -> GRAPE_new(F, G, x, U, L, G_array, P_array, g, grad, problem.H_drift, problem.H_ctrl, problem.timeslices, problem.n_ensemble, problem.duration, problem.number_pulses, problem)


init = rand(problem.number_pulses, problem.timeslices) .* 0.0001

using Optim
res = Optim.optimize(Optim.only_fg!(test), init, Optim.LBFGS(), Optim.Options(show_trace = true, allow_f_increases = false))

