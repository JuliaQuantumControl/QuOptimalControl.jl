using QuOptimalControl

using QuantumInformation
using DelimitedFiles

ψ1 = [1.0 + 0.0im 0.0]
ψt = [0.0 + 0.0im 1.0]

ρ1 = ψ1' * ψ1
ρt = ψt' * ψt

prob_GRAPE = ClosedStateTransfer(
    [sx, sy],
    [0.0 * sz],
    ρ1,
    ρt,
    1.0,
    1 / 10,
    10,
    2,
    1,
    GRAPE_approx(GRAPE)
)

sol = solve(prob_GRAPE)

function test_x(x)
    # real value return
end
GRAPE_AD(test_x, rand(2, 10))


prob = ClosedStateTransfer(
    B = [sx, sy],
    A = [0.0 * sz],
    X_init = ρ1,
    X_target = ρt,
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
    [sx, sy],
    [0.0 * sz],
    ρ1,
    ρt,
    1.0,
    1 / 10,
    10,
    2,
    1,
    dCRAB_type()
)


sol = solve(prob_dCRAB)

using Plots
bar(sol.minimizer[1, :], ylabel = "Control amplitude", xlabel = "Index", label = "1")
bar!(sol.minimizer[2, :], label = "2")


sol = solve(prob_dCRAB)

# the users functional should take a drive as an input and return the infidelity
function user_functional(x)
    # we get an a 2D array of pulses
    U = pw_evolve(0 * sz, [π * sx, π * sy], x, 2, dt, timeslices)
    # U = reduce(*, U)
    C1(ρt, (U * ρ1 * U'))
end

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

dt = 1 / 10
duration  = 1
timeslices = 10
n_freq = 1
n_coeff = 2
coeffs, pulses = dCRAB(n_pulses, dt, timeslices, duration, n_freq, n_coeff, user_functional_expt)
# starting over from the a fresh repl

controls = vcat(pulses...)
using Plots
bar(controls[1, :], ylabel = "Control amplitude", xlabel = "Index", label = "1")
bar!(controls[2, :], label = "2")