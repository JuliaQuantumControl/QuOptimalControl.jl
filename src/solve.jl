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
    result # optimisation result (not saved)
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
This solve function should handle ClosedStateTransfer, UnitarySynthesis and an open system governed in Liouville space, since the bulk of the code remains the same and we dispatch based on problem type to the correct evolution algorithms (Khaneja et. al.)

Concern below still valid!

ClosedSystem using approximation gradient of GRAPE uses this function to solve.
    Currently this calls optim, the other algorithms do not... maybe we should change it so that the optimisation is actually done in the algorithms.jl file.
"""
function _solve(problem::Union{ClosedStateTransfer,UnitarySynthesis}, alg::GRAPE_approx)

    # prepare some storage arrays that we will make use of throughout the computation
    U, L, G_array, P_array, g, grad = init_GRAPE(problem.X_init[1], problem.n_timeslices, problem.n_ensemble, problem.H_drift[1], problem.n_pulses)

    test = (F, G, x) -> GRAPE!(F, G, x, U, L, G_array, P_array, g, grad, problem.H_drift, problem.H_ctrl, problem.n_timeslices, problem.n_ensemble, problem.duration, problem.n_pulses, problem)


    # generate a random initial guess for the algorithm if the user hasn't provided one
    init = rand(problem.n_pulses, problem.n_timeslices) .* 0.001

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
        U = pw_evolve_T(problem.H_drift[1], problem.H_ctrl, x, problem.n_pulses, problem.timestep, problem.n_timeslices)
        C2(problem.X_target, (U * problem.X_init * U'))
    end

    init = rand(problem.n_pulses, problem.n_timeslices) .* 0.001
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
        U = pw_evolve(problem.H_drift[1], problem.H_ctrl, x, problem.n_pulses, problem.timestep, problem.n_timeslices)
        # U = reduce(*, U)
        C1(problem.X_target, U * problem.X_init)
    end

    coeffs, pulses, optim_results = dCRAB(problem.n_pulses, problem.timestep, problem.n_timeslices, problem.duration, alg.n_freq, alg.n_coeff, user_functional)

    solres = SolutionResult(optim_results, [optim_results[j].minimum for j = 1:length(optim_results)], pulses, problem)

end


"""
Unitary synthesis using dCRAB to solve the problem
Here we define a functional for the user, since we can assume that this is the type of problem that they want to solve
"""
function _solve(problem::UnitarySynthesis, alg::dCRAB_type)
    # we define our own functional here for a closed system

    function user_functional(x)
        # we get an a 2D array of pulses
        U = pw_evolve(problem.H_drift[1], problem.H_ctrl, x, problem.n_pulses, problem.timestep, problem.n_timeslices)
        # U = reduce(*, U)
        C1(problem.X_target, (U * problem.X_init * U'))
    end

    coeffs, pulses, optim_results = dCRAB(problem.n_pulses, problem.timestep, problem.n_timeslices, problem.duration, alg.n_freq, alg.n_coeff, user_functional)

    solres = SolutionResult(optim_results, [optim_results[j].minimum for j = 1:length(optim_results)], pulses, problem)

end



"""
Solve a unitary synthesis problem using ADGRAPE, this means that we need to use a piecewise evolution function that is Zygote compatible!
"""
function _solve(problem::UnitarySynthesis, alg::GRAPE_AD)
    
    function user_functional(x)
        U = pw_evolve_T(problem.H_drift[1], problem.H_ctrl, x, problem.n_pulses, problem.timestep, problem.n_timeslices)
        C1(problem.X_target, U * problem.X_init)
    end

    init = rand(problem.n_pulses, problem.n_timeslices) .* 0.001
    res = ADGRAPE(user_functional, init)

    solres = SolutionResult([res], [res.minimum], [res.minimizer], problem)
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

    coeffs, pulses, optim_results = dCRAB(problem.n_pulses, problem.timestep, problem.n_timeslices, problem.duration, alg.n_freq, alg.n_coeff, user_functional)

    solres = SolutionResult(optim_results, [optim_results[j].minimum for j = 1:length(optim_results)], pulses, problem)

end
