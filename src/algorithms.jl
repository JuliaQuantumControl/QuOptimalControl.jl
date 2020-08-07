abstract type algorithm end # capitalisation?
abstract type gradientBased <: algorithm end # Phlia and I always talked about building up a list of algorithms under their types
abstract type gradientFree <: algorithm end

"""
Testing a little idea, things are becoming a little confused. Need to write down what the ultimate goal is
# TODO I think we can implement the algorithms as structs that way we can dispatch on them properly
"""
struct GRAPE_approx <: gradientBased
    func_to_call # currently we'll just store the function associated with the algorithm
end

struct dCRAB_type <: gradientFree
    n_coeff
    n_freq
    func_to_call
end


include("./cost_functions.jl")
include("./evolution.jl")
include("./tools.jl")

"""
Simple. Use Zygote to solve all of our problems
Ideally we can use Zygote for open system problems because one day it'll work with master equations and in the meantime its good at handling control flow
"""
function ADGRAPE()
end

"""
Learn Yota and use it to do AD because then things are fast and we can compile the tape and all that good stuff 
"""
function ADGRAPE()

end


"""
Function that is compatible with Optim.jl, takes Hamiltonians, states and some other information and returns either the value of the functional (in this case its an overlap) or a first order approximate of the gradient. 
"""
# TODO - Shai has this Dynamo paper where he gives an exact gradient using the eigen function of the linear algebra library
# TODO - How do we use this with a simple ensemble?
# TODO - can we tidy it up and make it generically usable?
function GRAPE(F, G, H_drift, H_ctrl_arr, ρ, ρₜ, x_drive, n_ctrls, dt, n_steps)
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
    grad = similar(x_drive)
    for k = 1:n_ctrls
        for j = 1:n_steps
            grad[k, j] = real(tr((ρₜ_list[j])' * 1.0im * dt * commutator(H_ctrl_arr[k], ρ_list[j])))
            # grad[k, j] = -real(tr(ρₜ_list[j + 1]' * commutator(H_ctrl_arr[k], U_list[j]) * ρ_list[j])) * pc
        end
    end
    
    # compute total propagator
    U = reduce(*, U_list)
    # now lets compute the infidelity to minimize

    fid = C1(ρₜ, (U * ρ * U'))
    # fid = 1.0 - abs2(tr(ρₜ * (U * ρ * U')))
    
    if G !== nothing
        G .= grad
    end

    if F !== nothing
        return fid
    end

end

# using Optim
# test = (F, G, x) -> GRAPE(F, G, 0 * sz, [sx, sy], ρ, ρₜ, x, n_ctrls, dt, n_steps)

# res = Optim.optimize(Optim.only_fg!(test), init, Optim.LBFGS(), Optim.Options(show_trace = true, allow_f_increases = false))

"""
Using the dCRAB method to perform optimisation of a pulse. 

The user here must provide a functional that we will optimise
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
    end
    return optimised_coeffs, pulses
end



