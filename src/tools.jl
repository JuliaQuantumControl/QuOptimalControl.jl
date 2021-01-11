"""
Just some useful functions
"""
abstract type Solution end

using BSON
using DelimitedFiles
using Setfield
using StaticArrays

#const σₓ = SMatrix{} # etc



"""
Compute the commutator between two matrices A and B
"""
function commutator(A, B)
    A * B - B * A
end

# mul!ing over a in-place commutator :D (I need to get out more)
function commutator!(A, B, store)
    mul!(store, A, B)

    A * B - B * A
end



"""
Initialise all the storage arrays that will be used in a GRAPE optimisation
"""
function init_GRAPE(X, n_timeslices, n_ensemble, A, n_controls)
    # states
    states = [similar(X) for i = 1:n_timeslices + 1, j = 1:n_ensemble]
    # costates
    costates = [similar(X) for i = 1:n_timeslices + 1, j = 1:n_ensemble]
    # list of generators
    generators = [similar(A) for i = 1:n_timeslices, j = 1:n_ensemble]
    # exp(gens)
    propagators = [similar(A) for i = 1:n_timeslices, j = 1:n_ensemble]

    fom = zeros(n_ensemble)
    gradient = zeros(n_ensemble, n_controls, n_timeslices)
    return (states, costates, generators, propagators, fom, gradient)
end

"""
Initialise an ensemble from an ensemblr problem definition. Right now this creates an array of problems just now, not sure if this is the best idea though. 
"""
function init_ensemble(ens)
    ensemble_problem_array = [deepcopy(ens.problem) for k = 1:ens.n_ensemble]
    for k = 1:ens.n_ensemble
        prob_to_update = ensemble_problem_array[k]
        prob_to_update = @set prob_to_update.A = ens.A_generators(k) 
        prob_to_update = @set prob_to_update.B = ens.B_generators(k)
        prob_to_update = @set prob_to_update.X_init = ens.X_init_gens(k)
        prob_to_update = @set prob_to_update.X_target = ens.X_target_gens(k)
        ensemble_problem_array[k] = prob_to_update
    end
    ensemble_problem_array
end

"""
The SolutionResult stores important optimisation information in a nice format
"""
struct SolutionResult <: Solution
    result # optimisation result (not saved)
    fidelity # lets just extract the figure of merit that was reached
    optimised_pulses # store an array of the optimised pulses
    prob_info # can we store the struct or some BSON of the struct that was originally used
end


"""
Save a SolutionResult to file
"""
function save(solres, file_path)
    # convert the solres file into a dict for saving

    solres_dict = Dict()
    solres_keys = fieldnames(typeof(solres))[2:end]

    for k in solres_keys
        push!(solres_dict, k => getfield(solres, k))
    end


    bson(file_path, solres_dict)
end


"""
Load a SolutionResult from file
"""
function load(file_path)
    solres_dict = BSON.load(file_path)
    sol = SolutionResult(nothing, solres_dict[:fidelity], solres_dict[:problem_info], solres_dict[:optimised_pulses])
end

"""
Write an array called pulse to a file, could also write a time array if that's necessary.
"""
function pulse_to_file(pulse, file_path)
    # can maybe check the shape and reshape so that its always time going down the file
    open(file_path, "w") do io
        writedlm(io, pulse')
    end 
end

function pulse_to_file(pulse, file_path, duration)
    # can maybe check the shape and reshape so that its always time going down the file
    N = length(pulse)
    time_array = collect(range(0, duration, length = N))
    open(file_path, "w") do io
        writedlm(io, [time_array pulse'])
    end 
end


"""
Compute the eigen decomposition of a generator matrix G. Based on work by both Shai Machnes and Ville Bergholm.

Types are an issue here.
"""
function eig_factors(G; antihermitian = false, tol = 1e-10)
    if antihermitian
        d, v = eigen(1im * G)
        d = -1im .* d
    else
        d, v = eigen(G)
    end

    ooo = ones(size(d)[1])
    temp = d * ooo'
    diff = temp - temp'
    degenerate_mask = abs.(diff) .< tol
    diff[degenerate_mask] .= 1
    exp_d = exp.(d)

    temp = exp_d * ooo'
    exp_diff = temp - temp'
    zeta = exp_diff / diff
    zeta[degenerate_mask] .= temp[degenerate_mask]
    return (v, zeta, exp_d)
end


"""
Function to compute the matrix exponential using the eigenvalue decomposition of the matrix, this is slower than exp if your matrix can be a StaticArray
"""
function expm_exact_gradient(H::T, dt)::T where T
    dt_H = dt * H
    v, zeta, exp_d = eig_factors(dt_H, antihermitian = true)

    v * diagm(exp_d) * v'
end

"""
Copied from Ville's code, seems to be slower in Julia than simply taking trace but is faster in Matlab/Python
Function to compute trace(A @ B) efficiently.

Utilizes the identity trace(A @ B) == sum(transpose(A) * B).
Left side is O(n^3) to compute, right side is O(n^2).
"""
function trace_matmul(A, B)
    sum(transpose(A) .* B)
end