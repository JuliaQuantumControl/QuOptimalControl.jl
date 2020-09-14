# Phlia and I always talked about building up a list of algorithms under their types
abstract type algorithm end # capitalisation?
abstract type gradientBased <: algorithm end 
abstract type gradientFree <: algorithm end

using Zygote
using Optim
using ExponentialUtilities

import Base.@kwdef

"""
Structs here simply used for dispatch to the correct method
"""
struct GRAPE_approx <: gradientBased end
struct GRAPE_AD <: gradientBased end

@kwdef struct dCRAB_options <: gradientFree
    n_coeff = 2
    n_freq = 2
end


"""
Simple. Use Zygote to solve all of our problems
Compute the gradient of a functional f(x) wrt. x
"""
function ADGRAPE(functional, x; g_tol = 1e-6)

    function grad_functional!(G, x)
        G .= Zygote.gradient(functional, x)[1]
    end
    
    res = optimize(functional, grad_functional!, x, LBFGS(), Optim.Options(g_tol = g_tol, show_trace = true, store_trace = true))

end

"""
Learn Yota and use it to do AD because then things are fast and we can compile the tape and all that good stuff 
"""
function ADGRAPE()

end

"""
Initialise all the storage arrays that will be used in a GRAPE optimisation
"""
function init_GRAPE(X, n_timeslices, n_ensemble, A, n_controls)
    # states
    U = [similar(X) for i = 1:n_timeslices + 1, j = 1:n_ensemble]
    # costates
    L = [similar(X) for i = 1:n_timeslices + 1, j = 1:n_ensemble]
    # list of generators
    Generators = [similar(A) for i = 1:n_timeslices, j = 1:n_ensemble]
    # exp(generators)
    Propagators = [similar(A) for i = 1:n_timeslices, j = 1:n_ensemble]

    fom = zeros(n_ensemble)
    gradient = zeros(n_ensemble, n_controls, n_timeslices)
    return (U, L, Generators, Propagators, fom, gradient)
end


"""
Flexible GRAPE algorithm, should be able to handle all cases from the original Khaneja et al. paper (need citation)
Note: this code updates the arrays passed to it in place, this means it doesn't allocate memory but things can get messy
"""
function GRAPE!(F, G, x, U, L, Gen, P_list, g, grad, A, B, n_timeslices, n_ensemble, duration, n_controls, prob, evolve_store, weights)

    dt = duration / n_timeslices
    for k = 1:n_ensemble
        U[1, k] = prob.X_init[k]
        L[end, k] = prob.X_target[k]


        # do we need to actually save the generators of the transforms or do we simply need the propagators?
        pw_ham_save!(A[k], B[k], x, n_controls, n_timeslices, @view Gen[:,k])
        # @views P_list[:, k] .= ExponentialUtilities._exp!.(Gen[:,k] .* (-1.0im * dt))
        @views P_list[:, k] .= exp.(Gen[:, k] .* (-1.0im * dt))

        # forward propagate
        for t = 1:n_timeslices
            evolve_func!(prob, t, k, U, L, P_list, Gen, evolve_store, forward = true)
        end
        
        # prob backwards in time
        for t = reverse(1:n_timeslices)
            evolve_func!(prob, t, k, U, L, P_list, Gen, evolve_store, forward = false)
        end
        
        t = n_timeslices # can be chosen arbitrarily
        g[k] = fom_func(prob, t, k, U, L, P_list, Gen)

        # we can optionally compute this actually
        for c = 1:n_controls
            for t = 1:n_timeslices
                # might want to alter this to just pass the matrices that matter rather than everything
                @views grad[k, c, t] = grad_func(prob, t, dt, k, B[k][c], U, L, P_list, Gen, evolve_store)
            end
        end
            
    end

    # then we average over everything
    if G !== nothing
        @views G .= sum(weights .* grad, dims = 1)[1,:, :]
    end

    if F !== nothing
        return sum(g .* weights)
    end

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

    # functions for computing indices becaues I find them hard
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



