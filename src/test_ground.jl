include("./problems.jl")
include("./algorithms.jl")

using QuantumInformation


mutable struct ClosedStateTransferTest2 <: ClosedSystem
    control_Hamiltonians
    drift_Hamiltonians
    state_init # rho (?) 
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