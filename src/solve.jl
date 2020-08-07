"""
Contains the solve function interfaces for now, until I learn a better way to do it
"""



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
