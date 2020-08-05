include("./problems.jl")
include("./algorithms.jl")

using QuantumInformation


struct ClosedStateTransferTest2 <: ClosedSystem
    control_Hamiltonians
    drift_Hamiltonians
    state_init
    state_target
    duration
    timestep
    timeslices
    number_pulses
end

ψ1 = [1.0 + 0.0im 0.0]
ψt = [0.0 + 0.0im 1.0]

ρ1 = ψ1' * ψ1
ρt = ψt' * ψt

# need to figure out how keyword constructors work
prob = ClosedStateTransferTest2(
    [sx, sy],
    [0.0 * sz],
    ρ1,
    ρt,
    1.0,
    1 / 10,
    10,
    2 
)

function solve(problem)
    println("no algorithm, using default for problem type")
    # then use approx GRAPE for everything
end


function solve(problem::ClosedSystem, alg)
    # use the provided algorithm to solve the problem
    
    # in this case lets use GRAPE as an example
    test = (F, G, x) -> alg(F, G, problem.drift_Hamiltonians, problem.control_Hamiltonians, problem.state_init, problem.state_target, x, problem.number_pulses, problem.timestep, problem.timeslices)

    # generate a random initial guess for the algorithm if the user hasn't provided one
    init = rand(problem.number_pulses, problem.timeslices) .* 0.01

    res = Optim.optimize(Optim.only_fg!(test), init, Optim.LBFGS(), Optim.Options(show_trace = true, allow_f_increases = false))
    return res
end

sol = solve(prob, GRAPE)

using Plots
bar(sol.minimizer[1, :], ylabel = "Control amplitude", xlabel = "Index", label = "1")
bar!(sol.minimizer[2, :], label = "2")


# had this idea last night about providing alg_options to the algorithms that's different to the information that we pass to the step taking algorithm, this is about the actual algorithm

# testing params
n_pulses = 2
duration = 1
timeslices = 10
dt = 1 / 10


# alg specific params
n_freq = 10 # number of frequency components
n_coeff = 2 # number of coefficients


# the users functional should take a drive as an input and return the infidelity
function user_functional(x)
    # we get an a 2D array of pulses
    U = pw_evolve(0 * sz, [π * sx, π * sy], x, 1, dt, timeslices)
    # U = reduce(*, U)
    C1(ρt, (U * ρ1 * U'))
end

function user_functional_expt(x)
    # we get a 2D array of pulses, with delimited files we can write this to a file
end


coeffs, pulses = dCRAB(n_pulses, dt, timeslices, duration, n_freq, n_coeff, user_functional)
# starting over from the a fresh repl

controls = vcat(pulses...)
using Plots
bar(controls[1, :], ylabel = "Control amplitude", xlabel = "Index", label = "1")
bar!(controls[2, :], label = "2")