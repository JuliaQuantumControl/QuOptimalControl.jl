
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
    GRAPE_approx()
)


sol = solve(problem)

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




H_ctrl = [sx, sy]
H_drift = [sz]
n_pulses = length(H_ctrl)
duration = 5
n_timeslices = 10
timestep = duration / n_timeslices
ctrl_arr = rand(n_pulses, n_timeslices)
P_list = pw_evolve_save(H_drift[1], H_ctrl, ctrl_arr, n_pulses, timestep, n_timeslices)

X_init = ρ
states = similar(P_list)
states[1] = X_init
for i = 2:n_timeslices
    states[i] = P_list[i - 1] * states[i - 1]
end

z = [real(tr(sz * i)) for i in states]

