# Phlia and I always talked about building up a list of algorithms under their types
abstract type algorithm end # capitalisation?
abstract type gradientBased <: algorithm end 
abstract type gradientFree <: algorithm end

using Zygote
using Optim

# include("./cost_functions.jl")
# include("./grad_functions.jl")
# include("./problems.jl")
# include("./evolution.jl")
# include("./tools.jl")



import Base.@kwdef

"""
Structs here will replace the functions below I think, so that we can dispatch on them, need to figure this one out before continuing.
"""
# TODO I think we can implement the algorithms as structs that way we can dispatch on them properly
@kwdef struct GRAPE_approx <: gradientBased
    func_to_call = GRAPE! # currently we'll just store the function associated with the algorithm
end

@kwdef struct GRAPE_AD <: gradientBased
    func_to_call = ADGRAPE
end

@kwdef struct dCRAB_type <: gradientFree
    n_coeff = 2
    n_freq = 2
    func_to_call = dCRAB
end


"""
Simple. Use Zygote to solve all of our problems
Compute the gradient of a functional f(x) wrt. x
"""
function ADGRAPE(functional, x)

    function grad_functional!(G, x)
        G .= Zygote.gradient(functional, x)[1]
    end
    
    res = optimize(functional, grad_functional!, x, LBFGS(), Optim.Options(show_trace = true, store_trace = true))

end

"""
Learn Yota and use it to do AD because then things are fast and we can compile the tape and all that good stuff 
"""
function ADGRAPE()

end


"""
Function that is compatible with Optim.jl, takes Hamiltonians, states and some other information and returns either the value of the functional (in this case its an overlap) or a first order approximation of the gradient. Here we don't explicitly solve the problem using Optim, that's handled elsewhere for now (which might need changed).
"""
# TODO - Shai has this Dynamo paper where he gives an exact gradient using the eigen function of the linear algebra library
# TODO - How do we use this with a simple ensemble?
# TODO - can we tidy it up and make it generically usable?
function OLD_GRAPE(F, G, H_drift, H_ctrl_arr, ρ, ρₜ, x_drive, n_ctrls, dt, n_steps)
    @warn "this was my first try at GRAPE, its inflexible and unused now"
    # compute the propgators
    U_list = pw_evolve_save(H_drift[1], H_ctrl_arr, x_drive, n_ctrls, dt, n_steps)

    # now we propagate the initial state forward in time
    ρ_list = [ρ] # need to give it a type it ρ
    temp_state = ρ
    for U in U_list
        temp_state = U * temp_state * U'
        append!(ρ_list, [temp_state])
    end

    ρₜ_list = [ρₜ] # can also type this or do something else
    temp_state = ρₜ
    for U in reverse(U_list)
        temp_state = U' * temp_state * U
        append!(ρₜ_list, [temp_state])
    end
    ρₜ_list = reverse(ρₜ_list)

    # approximate gradient from Glaser paper is used here

    
    # compute total propagator
    U = reduce(*, U_list)
    # now lets compute the infidelity to minimize

    fid = C1(ρₜ, (U * ρ * U'))
    
    if G !== nothing
        grad = similar(x_drive)
        for k = 1:n_ctrls
            for j = 1:n_steps
                grad[k, j] = real(tr((ρₜ_list[j])' * 1.0im * dt * commutator(H_ctrl_arr[k], ρ_list[j])))
                # grad[k, j] = -real(tr(ρₜ_list[j + 1]' * commutator(H_ctrl_arr[k], U_list[j]) * ρ_list[j])) * pc
            end
        end
        G .= grad
    end

    if F !== nothing
        return fid
    end

end

# depending on the problem type there are a lot of similarities between GRAPE methods, lets think about how one might combine them

# lets assume that setting things up for the ensemble is done externally
# hence we get an array of X_init and X_target, norm2 and also a list of drift Hamiltonians, control Hamiltonians are assumed to be constant across the ensemble (for now)

# using a problem definition we can initialise all of the internal storage arrays
"""
Initialise all the storage arrays for a GRAPE optimisation
"""
function init_GRAPE(X, timeslices, n_ensemble, H_drift, n_controls)
    U = repeat([similar(X)], timeslices + 1, n_ensemble)
    L = repeat([similar(X)], timeslices + 1, n_ensemble)
    # list of generators
    G_array = repeat([similar(H_drift)], timeslices, n_ensemble)
    # exp(generators)
    P_array = repeat([similar(H_drift)], timeslices, n_ensemble)

    g = zeros(n_ensemble)
    grad = zeros(n_controls, timeslices, n_ensemble)
    return (U, L, G_array, P_array, g, grad)
end

"""
More flexible GRAPE algorithm, should be able to handle all cases from the original Khaneja et al. paper (need citation)
Note: this code updates the arrays passed to it in place, this means it doesn't allocate memory but things can get messy
"""
# TODO decide if we should pass functions as arguments or not
function GRAPE!(F, G, x, U, L, Gen, P_list, g, grad, H_drift, H_ctrls, timeslices, n_ensemble, duration, n_controls, prob)# , fom_func, gradient_func, evolve_func)
    dt = duration / timeslices
    for k = 1:n_ensemble
        U[1, k] = prob.X_init[k]
        L[end, k] = prob.X_target[k]
        # do we treat all generators like this really?
        Gen[:,k] .= pw_ham_save(H_drift[k], H_ctrls, x, n_controls, timeslices) .* -1.0im * dt
        P_list[:, k] = exp.(Gen[:, k])

        # forward propagate
        # in order for this to be general it will look for an evolution rule
        for t = 1:timeslices
            U[t + 1, k] = evolve_func(prob, t, k, U, L, P_list, Gen)
        end
        
        # prob backwards in time
        for t = reverse(1:timeslices)
            L[t, k] = evolve_func(prob, t, k, U, L, P_list, Gen, forward = false)
        end
        
        t = timeslices # can be chosen arbitrarily
        g[k] = fom_func(prob, t, k, U, L, P_list, Gen)

        # we can optionally compute this actually
        # in order for this to be general it uses looks for a gradient definition function
        for c = 1:n_controls
            for t = 1:timeslices
                # might want to alter this to just pass the matrices that matter rather than everything
                grad[c, t, k] = grad_func(prob, t, dt, k, H_ctrls[c], U, L, P_list, Gen)
            end
        end
            
    end

    # then we average over everything, currently everything is equally weighted
    wts = ones(n_ensemble)

    if G !== nothing
        G .= sum(wts .* grad, dims = 3)[:,:,1]
    end

    if F !== nothing
        return sum(g .* wts)
    end

end


"""
Using the dCRAB method to perform optimisation of a pulse. 

The user here must provide a functional that we will optimise, optimisation is carried out here
"""
function dCRAB(n_pulses, dt, timeslices, duration, n_freq, n_coeff, user_func)

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

    pulses = [zeros(1, timeslices) for i = 1:n_pulses]

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



