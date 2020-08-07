using Optim
"""
Contains the solve function interfaces for now, until I learn a better way to do it
"""



"""
Generic interface, takes the alg choice out of the problem struct and then we can dispatch on it to the various _solve methods
"""
function solve(problem)
    # println("no algorithm, using default for problem type")
    # then use approx GRAPE for everything
    _solve(problem, problem.alg)
end


"""
ClosedSystem using approximation gradient of GRAPE uses this function to solve
"""
function _solve(problem::ClosedStateTransfer, alg::GRAPE_approx)
    test = (F, G, x) -> alg.func_to_call(F, G, problem.drift_Hamiltonians, problem.control_Hamiltonians, problem.state_init, problem.state_target, x, problem.number_pulses, problem.timestep, problem.timeslices)

    # generate a random initial guess for the algorithm if the user hasn't provided one
    init = rand(problem.number_pulses, problem.timeslices) .* 0.01

    res = Optim.optimize(Optim.only_fg!(test), init, Optim.LBFGS(), Optim.Options(show_trace = true, allow_f_increases = false))
    return res
end

"""
ClosedSystem using dCRAB to solve
"""
function _solve(problem::ClosedStateTransfer, alg::dCRAB_type)
    # we define our own functional here for a closed system
    
    function user_functional(x)
        # we get an a 2D array of pulses
        U = pw_evolve(problem.drift_Hamiltonians[1], problem.control_Hamiltonians, x, problem.number_pulses, problem.timestep, problem.timeslices)
        # U = reduce(*, U)
        C1(problem.state_target, (U * problem.state_init * U'))
    end

    coeffs, pulses = dCRAB(problem.number_pulses, problem.timestep, problem.timeslices, problem.duration, alg.n_freq, alg.n_coeff, user_functional)
    
end

"""
Optimising an experiment here, using dCRAB should work closed loop I hope
"""
function _solve(problem::Experiment, alg::dCRAB_type)
    """
    user functional that will create a pulse file and start the experiment
    """
    function user_functional(x)
        # we get a 2D array of pulses, with delimited files we can write this to a file
        # writedlm("./")
        open(problem.pulse_path, "w") do io
            writedlm(io, x')
        end
        # TODO you need to wait can use built in FileWatching function 
        problem.start_exp()
    
        # then read in the result
        infid = readdlm(problem.infidelity_path)[1]
    end

    coeffs, pulses = dCRAB(problem.number_pulses, problem.timestep, problem.timeslices, problem.duration, alg.n_freq, alg.n_coeff, user_functional)
    
end
