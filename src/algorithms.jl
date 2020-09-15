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
    gens = [similar(A) for i = 1:n_timeslices, j = 1:n_ensemble]
    # exp(gens)
    props = [similar(A) for i = 1:n_timeslices, j = 1:n_ensemble]

    fom = zeros(n_ensemble)
    gradient = zeros(n_ensemble, n_controls, n_timeslices)
    return (U, L, gens, props, fom, gradient)
end


"""
Flexible GRAPE algorithm that can use any array type to solve GRAPE problems. This is best when your system is so large that StaticArrays cannot be used! The arrays given will be updated in place!
"""
function GRAPE!(F, G, x, U, L, gens, props, fom, gradient, A, B, n_timeslices, n_ensemble, duration, n_controls, prob, evolve_store, weights)

    dt = duration / n_timeslices
    for k = 1:n_ensemble
        U[1, k] = prob.X_init[k]
        L[end, k] = prob.X_target[k]


        # do we need to actually save the generators of the transforms or do we simply need the propagators?
        pw_ham_save!(A[k], B[k], x, n_controls, n_timeslices, @view gens[:,k])
        # @views props[:, k] .= ExponentialUtilities._exp!.(Gen[:,k] .* (-1.0im * dt))
        @views props[:, k] .= exp.(gens[:, k] .* (-1.0im * dt))

        # forward propagate
        for t = 1:n_timeslices
            evolve_func!(prob, t, k, U, L, props, gens, evolve_store, forward = true)
        end
        
        # prob backwards in time
        for t = reverse(1:n_timeslices)
            evolve_func!(prob, t, k, U, L, props, gens, evolve_store, forward = false)
        end
        
        t = n_timeslices # can be chosen arbitrarily
        fom[k] = fom_func(prob, t, k, U, L, props, gens)

        # we can optionally compute this actually
        for c = 1:n_controls
            for t = 1:n_timeslices
                # might want to alter this to just pass the matrices that matter rather than everything
                @views gradient[k, c, t] = grad_func!(prob, t, dt, k, B[k][c], U, L, props, gens, evolve_store)
            end
        end
            
    end

    # then we average over everything
    if G !== nothing
        @views G .= sum(weights .* gradient, dims = 1)[1,:, :]
    end

    if F !== nothing
        return sum(fom .* weights)
    end

end


"""
Flexible GRAPE algorithm for use with StaticArrays where the size is always fixed. This works best if there are < 100 elements in the arrays. The result is that you can avoid allocations (this whole function allocates just 4 times in my tests). If your system is too large then try the GRAPE! algorithm above which should work for generic array types!
"""
function GRAPE(A::T, B, u_c, n_timeslices, duration, n_controls, gradient_store, U_k, L_k, xinit, xtarget) where T
    
    dt = duration / n_timeslices
    # arrays that hold static arrays?
    U_k[1] = xinit
    L_k[end] = xtarget


    # now we want to compute the generators 
    gens = pw_ham_save(A, B, u_c, n_controls,n_timeslices)
    props = exp.(gens .* (-1.0im * dt))


    # forward evolution of states
    for t = 1:n_timeslices
        U_k[t+1] = evolve_func(prob, t, U_k, L_k, props, 1, 1, forward = true)
    end
    # backward evolution of costates
    for t = reverse(1:n_timeslices)
        L_k[t] = evolve_func(prob, t, U_k, L_k, props, 1, 1, forward = false)
    end

    t = n_timeslices

    # update the gradient array
    for c = 1:n_controls
        for t = 1:n_timeslices
            # @views gradient_store[c, t] = real((1.0im * dt) .* tr(L_k[t]' * commutator(B[c], U_k[t]) ))
            # real(tr( L_k[t]' * (1.0im * dt) * commutator(B[c], U_k[t]) ))
            @views gradient_store[c, t] = grad_func(prob, t, dt, B, U, L, props, gens)

        end
    end


    return C1(L_k[t], U_k[t])
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



