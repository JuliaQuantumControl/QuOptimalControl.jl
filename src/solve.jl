abstract type Solution end
using Optim
using FileWatching

"""
Contains the solve function interfaces for now, until I learn a better way to do it
"""

"""
The SolutionResult stores important optimisation information in a nice format
"""
struct SolutionResult <: Solution
    result # optimisation result
    fidelity # lets just extract the figure of merit that was reached
    optimised_pulses # store an array of the optimised pulses
    problem_info # can we store the struct or some BSON of the struct that was originally used
end


"""
Generic interface, takes the alg choice out of the problem struct and then we can dispatch on it to the various _solve methods
"""
function solve(problem)
    # println("no algorithm, using default for problem type")
    # then use approx GRAPE for everything
    _solve(problem, problem.alg)
end

"""
ClosedSystem using approximation gradient of GRAPE uses this function to solve.
    Currently this calls optim, the other algorithms do not... maybe we should change it so that the optimisation is actually done in the algorithms.jl file.
"""
function _solve(problem::ClosedStateTransfer, alg::GRAPE_approx)
    @warn "out of date currently, needs fixed"
    test = (F, G, x) -> alg.func_to_call(F, G, problem.drift_Hamiltonians, problem.control_Hamiltonians, problem.state_init, problem.state_target, x, problem.number_pulses, problem.timestep, problem.timeslices)

    # generate a random initial guess for the algorithm if the user hasn't provided one
    init = rand(problem.number_pulses, problem.timeslices) .* 0.01

    res = Optim.optimize(Optim.only_fg!(test), init, Optim.LBFGS(), Optim.Options(show_trace = true, allow_f_increases = false))
    # TODO we need to decide on a common appearance for these SolutionResult structs
    solres = SolutionResult([res], [res.minimum], [res.minimizer], problem)

    return solres
end

"""
Solve closed state transfer problem using ADGRAPE, this means that we need to use a piecewise evolution function that is Zygote compatible!
"""
function _solve(problem::ClosedStateTransfer, alg::GRAPE_AD)
    
    function user_functional(x)
        U = pw_evolve_T(problem.drift_Hamiltonians[1], problem.control_Hamiltonians, x, problem.number_pulses, problem.timestep, problem.timeslices)
        C2(problem.state_target, (U * problem.state_init * U'))
    end

    init = rand(problem.number_pulses, problem.timeslices) .* 0.01
    res = ADGRAPE(user_functional, init)

    solres = SolutionResult([res], [res.minimum], [res.minimizer], problem)
end

"""
ClosedSystemStateTransfer using dCRAB to solve the problem
Here we define a functional for the user, since we can assume that this is the type of problem that they want to solve
"""
function _solve(problem::ClosedStateTransfer, alg::dCRAB_type)
    # we define our own functional here for a closed system

    function user_functional(x)
        # we get an a 2D array of pulses
        U = pw_evolve(problem.drift_Hamiltonians[1], problem.control_Hamiltonians, x, problem.number_pulses, problem.timestep, problem.timeslices)
        # U = reduce(*, U)
        C1(problem.state_target, (U * problem.state_init * U'))
    end

    coeffs, pulses, optim_results = dCRAB(problem.number_pulses, problem.timestep, problem.timeslices, problem.duration, alg.n_freq, alg.n_coeff, user_functional)

    solres = SolutionResult(optim_results, [optim_results[j].minimum for j = 1:length(optim_results)], pulses, problem)

end

"""
Closed loop experiment optimisation using dCRAB, the user functional isn't defined by the user at the moment, instead we define it. 
"""
function _solve(problem::Experiment, alg::dCRAB_type)
    """
    user functional that will create a pulse file and start the experiment
    """
    function user_functional(x)
        # we get a 2D array of pulses, with delimited files we can write this to a file
        open(problem.pulse_path, "w") do io
            writedlm(io, x')
        end
        # TODO you need to wait can use built in FileWatching function 
        problem.start_exp()
        o = watch_file(problem.infidelity_path, problem.timeout) # might need to monitor this differently

        # then read in the result
        infid = readdlm(problem.infidelity_path)[1]
    end

    coeffs, pulses, optim_results = dCRAB(problem.number_pulses, problem.timestep, problem.timeslices, problem.duration, alg.n_freq, alg.n_coeff, user_functional)

    solres = SolutionResult(optim_results, [optim_results[j].minimum for j = 1:length(optim_results)], pulses, problem)

end
