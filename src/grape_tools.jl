"""
Initialise all the storage arrays that will be used in a GRAPE optimisation
"""
function init_GRAPE(X, n_timeslices, n_ensemble, A, n_controls)
    # states
    states = [similar(X) .* 0.0 for i = 1:n_timeslices+1, j = 1:n_ensemble]
    # costates
    costates = [similar(X) .* 0.0 for i = 1:n_timeslices+1, j = 1:n_ensemble]
    # exp(gens)
    propagators = [similar(A) .* 0.0 for i = 1:n_timeslices, j = 1:n_ensemble]

    fom = 0.0
    gradient = zeros(n_ensemble, n_controls, n_timeslices)
    # end
    return (states, costates, propagators, fom, gradient)
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
function expm_exact_gradient(H::T, dt)::T where {T}
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
