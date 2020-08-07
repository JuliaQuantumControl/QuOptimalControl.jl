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
    GRAPE_approx(GRAPE)
)

sol = solve(prob_GRAPE)


prob_ADGRAPE = ClosedStateTransfer(
    [sx, sy],
    [0.0 * sz],
    ρ1,
    ρt,
    1.0,
    1 / 10,
    10,
    2,
    GRAPE_AD()
)

sol = solve(prob_ADGRAPE)

prob_dCRAB = ClosedStateTransfer(
    [sx, sy],
    [0.0 * sz],
    ρ1,
    ρt,
    1.0,
    1 / 10,
    10,
    2,
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


coeffs, pulses = dCRAB(n_pulses, dt, timeslices, duration, n_freq, n_coeff, user_functional)
# starting over from the a fresh repl

controls = vcat(pulses...)
using Plots
bar(controls[1, :], ylabel = "Control amplitude", xlabel = "Index", label = "1")
bar!(controls[2, :], label = "2")