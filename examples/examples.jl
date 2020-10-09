using QuOptimalControl

using QuantumInformation
using DelimitedFiles

ψ1 = [1.0 + 0.0im 0.0]
ψt = [0.0 + 0.0im 1.0]

ρ1 = ψ1' * ψ1
ρt = ψt' * ψt

prob = ClosedStateTransfer(
    B = [[sx, sy]],
    A = [0.0 * sz],
    X_init = [ρ1],
    X_target = [ρt],
    duration = 1.0,
    n_timeslices = 10,
    n_controls = 2,
    n_ensemble = 1,
    norm2 = 1.0,
    alg = GRAPE_approx(inplace = true),
    initial_guess = rand(2, 10)
)

sol = solve(prob)

prob = ClosedStateTransfer(
    B = [[sx, sy]],
    A = [0.0 * sz],
    X_init = [ρ1],
    X_target = [ρt],
    duration = 1.0,
    n_timeslices = 10,
    n_controls = 2,
    n_ensemble = 1,
    norm2 = 1,
    alg = GRAPE_AD(),
    initial_guess = rand(2, 10)
)

sol = solve(prob)

prob_dCRAB = ClosedStateTransfer(
    B = [[sx, sy]],
    A = [0.0 * sz],
    X_init = ρ1,
    X_target = ρt,
    duration = 1.0,
    n_timeslices = 10,
    n_controls = 2,
    n_ensemble = 1,
    norm2 = 1.0,
    alg = dCRAB_options(),
    initial_guess = rand(2,10)
)


sol = solve(prob_dCRAB)

visualise_pulse(sol.optimised_pulses, duration = prob.duration)

"""
Dummy function that simply saves a random number in the result.txt file
"""
function start_exp()
    open("result.txt", "w") do io
        writedlm(io, rand())
    end
end

"""
Example functional for closed loop optimal control using dCRAB
"""
function user_functional_expt(x)
    # we get a 2D array of pulses, with delimited files we can write this to a file
    # writedlm("./")
    open("pulses.txt", "w") do io
        writedlm(io, x')
    end
    # then lets imagine we use PyCall or something to start exp
    start_exp()
    # then read in the result
    infid = readdlm("result.txt")[1]
end
