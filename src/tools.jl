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
    # if n_ensemble == 1
    #     # states
    #     states = [similar(X) for i = 1:n_timeslices + 1]
    #     # costates
    #     costates = [similar(X) for i = 1:n_timeslices + 1]
    #     # list of generators
    #     generators = [similar(A) for i = 1:n_timeslices]
    #     # exp(gens)
    #     propagators = [similar(A) for i = 1:n_timeslices]

    #     fom = 0.0
    #     gradient = zeros(n_controls, n_timeslices)
    # else
        # states
        states = [similar(X) for i = 1:n_timeslices + 1, j = 1:n_ensemble]
        # costates
        costates = [similar(X) for i = 1:n_timeslices + 1, j = 1:n_ensemble]
        # list of generators
        generators = [similar(A) for i = 1:n_timeslices, j = 1:n_ensemble]
        # exp(gens)
        propagators = [similar(A) for i = 1:n_timeslices, j = 1:n_ensemble]

        fom = 0.0
        gradient = zeros(n_ensemble, n_controls, n_timeslices)
    # end
    return (states, costates, generators, propagators, fom, gradient)
end

"""
Initialise an ensemble from an ensemble problem definition. Right now this creates an array of problems just now, not sure if this is the best idea though. 
"""
function init_ensemble(ens)
    ensemble_problem_array = [deepcopy(ens.problem) for k = 1:ens.n_ensemble]
    for k = 1:ens.n_ensemble
        prob_to_update = ensemble_problem_array[k]
        prob_to_update = @set prob_to_update.A = ens.A_generators(k) 
        prob_to_update = @set prob_to_update.B = ens.B_generators(k)
        prob_to_update = @set prob_to_update.X_init = ens.X_init_generators(k)
        prob_to_update = @set prob_to_update.X_target = ens.X_target_generators(k)
        ensemble_problem_array[k] = prob_to_update
    end
    ensemble_problem_array # vector{ens.problem}
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





using LinearAlgebra
using SparseArrays

"""
    fastExpm(A)
    fastExpm(A; threshold=1e-6)
    fastExpm(A; nonzero_tol=1e-14)
    fastExpm(A; threshold=1e-6, nonzero_tol=1e-14)

 This function efficiently implements matrix exponential for sparse and full matrices.
 This code is based on scaling, taylor series and squaring.
 Currently works only on the CPU

 Two optional keyword arguments are used to speed up the computation and preserve sparsity.
 [1] threshold determines the threshold for the Taylor series (default 1e-6): e.g. fastExpm(A, threshold=1e-10)
 [2] nonzero_tol strips elements smaller than nonzero_tol at each computation step to preserve sparsity (default 1e-14) ,e.g. fastExpm(A, nonzero_tol=1e-10)
 The code automatically switches from sparse to full if sparsity is below 25% to maintain speed.

 This code was originally developed by Ilya Kuprov (http://spindynamics.org/) and has been adapted by F. Mentink-Vigier (fmentink@magnet.fsu.edu)
 and Murari Soundararajan (murari@magnet.fsu.edu)
 If you use this code, please cite
  - H. J. Hogben, M. Krzystyniak, G. T. P. Charnock, P. J. Hore and I. Kuprov, Spinach – A software library for simulation of spin dynamics in large spin systems, J. Magn. Reson., 2011, 208, 179–194.
  - I. Kuprov, Diagonalization-free implementation of spin relaxation theory for large spin systems., J. Magn. Reson., 2011, 209, 31–38.
"""
function fastExpm(A::AbstractMatrix;threshold=1e-6,nonzero_tol=1e-14)
    mat_norm=norm(A,Inf);
    scaling_factor = nextpow(2,mat_norm); # Native routine, faster
    A = A./scaling_factor;
    delta=1;
    rows = LinearAlgebra.checksquare(A); # Throws exception if not square

    # Run Taylor series procedure on the CPU
    if nnz_ext(A)/(rows^2)>0.25 || rows<64
        A=Matrix(A);
        P=Matrix((1.0+0*im)*I,(rows,rows)); next_term=P; n=1;
    else
        A=sparse(A);
        P=sparse((1.0+0*im)*I,(rows,rows)); next_term=P; n=1;
    end

    while delta>threshold
        # Compute the next term
        if issparse(next_term)
            next_term=(1/n)*A*next_term;
            #Eliminate small elements
            next_term=droptolerance!(next_term, nonzero_tol);
            if nnz_ext(next_term)/length(next_term)>0.25
                next_term=Matrix(next_term);
            end
        else
            next_term=(1/n)*next_term*A;
        end
        delta=norm(next_term,Inf);
        #Add to the total and increment the counter
        P .+= next_term; n=n+1;
    end
    #Squaring of P to generate correct P
    for n=1:log2(scaling_factor)
        P=P*P;
        if issparse(P)
            if nnz_ext(P)/length(P)<0.25
                P = droptolerance!(P, nonzero_tol);
            else
                P=Matrix(P);
            end
        end
    end
    return P
end

function droptolerance!(A::Matrix, tolerance)
    A .= tolerance*round.((1/tolerance).*A)
end
function droptolerance!(A::SparseMatrixCSC, tolerance)
    droptol!(A, tolerance) # Native routine, faster
end

function nnz_ext(A::Matrix)
    count(x->x>0, abs.(A))
end
function nnz_ext(A::SparseMatrixCSC)
    nnz(A) # Native routine, faster
end