
@kwdef struct dCRAB_options <: gradientFree
    n_coeff = 2
    n_freq = 2
end


"""
Using the dCRAB method to perform optimisation of a pulse. 

The user here must provide a functional that we will optimise, optimisation is carried out here
"""
function dCRAB(n_pulses, dt, timeslices, duration, n_freq, n_coeff, initial_guess, user_func)

    # lets set up an ansatz that will currently be for Fourier series
    # lets also refactor our ansatz
    ansatz(coeffs, ω, t) = coeffs[1] * cos(ω * t) + coeffs[2] * sin(ω * t)

    # initially randomly chosen frequencies (can refine this later)
    init_freq = rand(n_freq, n_pulses)

    # not sure the best way to handle multiple pulses at the moment sadly, but this method seems to work
    # we do this per pulse?
    init_coeffs = rand(n_freq, n_coeff, n_pulses)

    optimised_coeffs = []
    optim_results = [] # you do 1NM search per super iteration so you need to keep track of that

    # pulses = [zeros(1, timeslices) for i = 1:n_pulses]
    pulses = initial_guess # lets think about this a bit, it should be a list of pulses
    # pulses = [initial_guess[:, i] for i = 1:n_pulses] # something like this maybe?

    pulse_time = 0:dt:duration - dt

    # functions for computing indices because I find them hard
    first(j) = (j - 1) * n_coeff + 1
    second(j) = j * n_coeff

    # now we loop over everything

    for i = 1:n_freq
        freqs = init_freq[i, :] # so this contains the frequencies for all of the pulses

        # wrap the user defined function so that we can convert a list of coefficients into a pulse, user function should simply take an input 2D array and return the infidelity
        function to_minimize(x)
            # copy pulses 
            copy_pulses = copy(pulses)

            # I find getting indices hard, want to divide up the array x into n_coeff chunks
            [copy_pulses[j] += reshape(ansatz.((x[first(j):second(j)],), freqs[j], pulse_time), (1, timeslices)) for j = 1:n_pulses]
            user_func(vcat(copy_pulses...))
        end

        # now optimise with nelder mead
        result = Optim.optimize(to_minimize, reshape(init_coeffs[i, :, :], 4), Optim.NelderMead(), Optim.Options(show_trace = true, allow_f_increases = false))

        # update the pulses, save the coefficients
        [pulses[j] += reshape(ansatz.((result.minimizer[first(j):second(j)],), freqs[j], pulse_time), (1, timeslices)) for j = 1:n_pulses]

        # depending on the fidelity we should break here
        append!(optimised_coeffs, [result.minimizer])
        append!(optim_results, [result])
    end
    return optimised_coeffs, pulses, optim_results
end



# """
# ClosedSystemStateTransfer using dCRAB to solve the prob
# Here we define a functional for the user, since we can assume that this is the type of prob that they want to solve
# """
# function _solve(prob::StateTransferProblem, alg::dCRAB_options)
#     # we define our own functional here for a closed system

#     wts = ones(prob.n_ensemble)
#     function user_functional(x)
#         fom = 0.0
#         for k = 1:prob.n_ensemble
#             # we get an a 2D array of pulses
#             U = pw_evolve(prob.A[k], prob.B[k], x, prob.n_controls, prob.duration/prob.n_timeslices, prob.n_timeslices)
#             # U = reduce(*, U)
#             fom += C2(prob.X_target, U * prob.X_init) * wts[k]
#         end
#         fom
#     end

#     coeffs, pulses, optim_results = dCRAB(prob.n_controls, prob.duration/prob.n_timeslices, prob.n_timeslices, prob.duration, alg.n_freq, alg.n_coeff, prob.initial_guess, user_functional)

#     solres = SolutionResult(optim_results, [optim_results[j].minimum for j = 1:length(optim_results)], pulses, prob)

# end


# """
# Unitary synthesis using dCRAB to solve the prob
# Here we define a functional for the user, since we can assume that this is the type of prob that they want to solve
# """
# function _solve(prob::UnitaryProblem, alg::dCRAB_options)
#     # we define our own functional here for a closed system
#     wts = ones(prob.n_ensemble)

#     function user_functional(x)
#         fom = 0.0
#         for k = 1:prob.n_ensemble
#             # we get an a 2D array of pulses
#             U = pw_evolve(prob.A[k], prob.B[k], x, prob.n_controls, prob.duration/prob.n_timeslices, prob.n_timeslices)
#             # U = reduce(*, U)
#             fom += C1(prob.X_target[k], (U * prob.X_init[k])) * wts[k]
#         end
#         fom
#     end

#     coeffs, pulses, optim_results = dCRAB(prob.n_controls, prob.duration/prob.n_timeslices, prob.n_timeslices, prob.duration, alg.n_freq, alg.n_coeff, prob.initial_guess, user_functional)

#     solres = SolutionResult(optim_results, [optim_results[j].minimum for j = 1:length(optim_results)], pulses, prob)

# end

# """
# Closed loop experiment optimisation using dCRAB, the user functional isn't defined by the user at the moment, instead we define it. 
# """
# function _solve(prob::Experiment, alg::dCRAB_options)
#     """
#     user functional that will create a pulse file and start the experiment
#     """
#     function user_functional(x)
#         # we get a 2D array of pulses, with delimited files we can write this to a file
#         open(prob.pulse_path, "w") do io
#             writedlm(io, x')
#         end
#         # TODO you need to wait can use built in FileWatching function 
#         prob.start_exp()
#         o = watch_file(prob.infidelity_path, prob.timeout) # might need to monitor this differently

#         # then read in the result
#         infid = readdlm(prob.infidelity_path)[1]
#     end

#     coeffs, pulses, optim_results = dCRAB(prob.n_controls, prob.duration/prob.n_timeslices, prob.n_timeslices, prob.duration, alg.n_freq, alg.n_coeff, prob.initial_guess, user_functional)

#     solres = SolutionResult(optim_results, [optim_results[j].minimum for j = 1:length(optim_results)], pulses, prob)

# end

# """
# Can we combine algorithms to improve how we solve things? Comes from discussion with Phila
# """
# function hybrid_solve(prob, alg_list)
#     for alg in alg_list
#         sol = _solve(prob, alg)
#         prob.initial_guess[:] = sol.optimised_pulses
#     end
#     sol
# end


