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
@kwdef struct GRAPE_approx <: gradientBased 
    g_tol = 1e-6
    inplace = true # if you want to use StaticArrays set this to false!
end

struct GRAPE_AD <: gradientBased end

@kwdef struct dCRAB_options <: gradientFree
    n_coeff = 2
    n_freq = 2
end


"""
Simple. Use Zygote to solve all of our problems
Compute the gradient of a functional f(x) wrt. x
"""
function ADGRAPE(functional, x; g_tol = 1e-6, iters=1000)

    function grad_functional!(G, x)
        G .= Zygote.gradient(functional, x)[1]
    end
    
    res = optimize(functional, grad_functional!, x, LBFGS(), Optim.Options(g_tol = g_tol, show_trace = true, store_trace = true, iterations=iters))

end

"""
Learn Yota and use it to do AD because then things are fast and we can compile the tape and all that good stuff 
"""
function ADGRAPE()

end


"""
Flexible GRAPE algorithm that can use any array type to solve GRAPE problems. This is best when your system is so large that StaticArrays cannot be used! The arrays given will be updated in place!
"""
function GRAPE!(A::T, B, u_c, n_timeslices, duration, n_controls, gradient, U_k, L_k, gens, props, X_init, X_target, evolve_store, prob) where T
    dt = duration / n_timeslices
    U_k[1] .= X_init
    L_k[end] .= X_target

    pw_ham_save!(A, B, u_c, n_controls, n_timeslices, @view gens[:])
    @views props[:] .= exp.(gens[:] .* (-1.0im * dt))

    for t = 1:n_timeslices
        evolve_func!(prob, t, U_k, L_k, props, gens, evolve_store, forward = true)
    end

    for t = reverse(1:n_timeslices)
        evolve_func!(prob, t, U_k, L_k, props, gens, evolve_store, forward = false)
    end

    t = n_timeslices
    
    for c = 1:n_controls
        for t = 1:n_timeslices
            @views gradient[c, t] = grad_func!(prob, t, dt, B[c], U_k, L_k, props, gens, evolve_store)
        end
    end

    return fom_func(prob, t, U_k, L_k, props, gens)

end

"""
Static GRAPE

Flexible GRAPE algorithm for use with StaticArrays where the size is always fixed. This works best if there are < 100 elements in the arrays. The result is that you can avoid allocations (this whole function allocates just 4 times in my tests). If your system is too large then try the GRAPE! algorithm above which should work for generic array types!
"""
function sGRAPE(A::T, B, u_c, n_timeslices, duration, n_controls, gradient, U_k, L_k, X_init, X_target, prob) where T
    
    dt = duration / n_timeslices
    # arrays that hold static arrays?
    U_k[1] = X_init
    L_k[end] = X_target


    # now we want to compute the generators 
    gens = pw_ham_save(A, B, u_c, n_controls,n_timeslices)
    props = exp.(gens .* (-1.0im * dt))


    # forward evolution of states
    for t = 1:n_timeslices
        U_k[t+1] = evolve_func(prob, t, U_k, L_k, props, gens, forward = true)
    end
    # backward evolution of costates
    for t = reverse(1:n_timeslices)
        L_k[t] = evolve_func(prob, t, U_k, L_k, props, gens, forward = false)
    end
    # update the gradient array
    for c = 1:n_controls
        for t = 1:n_timeslices
            gradient[c, t] = grad_func(prob, t, dt, B[c], U_k, L_k, props, gens)

        end
    end

    t = n_timeslices
    return fom_func(prob, t, U_k, L_k, props, gens)
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



