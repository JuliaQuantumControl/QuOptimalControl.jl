"""
Just some useful functions
"""

"""
Compute the commutator between two matrices A and B
"""
function commutator(A, B)
    A * B - B * A
end

"""
Save a SolutionResult to file
"""
function save(solres)

end

"""
Based directly on Ville's code. Types are an issue here
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