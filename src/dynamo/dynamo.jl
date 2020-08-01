# lets try our hand at recreating dynamo in Julia


function PSU_norm(v::Number)
    abs(v) / OC.config / normNorm
end

function PSU_norm(v)
    v = phi0_norm(v)
    v = abs(v) / OC.config.normNorm
end

function phi0_norm(A)
    trace(A)
end

function phi0_norm(A, B)
    trace_matmul(A, B)
end

function SU_norm(v::Number)
    real(v) / OC.config.normNorm
end

function SU_norm(v)
    v = phi0_norm(v)
    real(v) / OC.config.normNorm
end

function trace_matmul(A, B)
    sum(sum(A' * B))
end

function calcPfromH_exact_gradient(t)
    minus_i_dt_H = -1.0im * OC.timeSlots.tau[t] * OC.timeSlots.currPoint.H[t]

    N = length(minus_i_dt_H)
    eigVec, eigVal = eigen(minus_i_dt_H)

    eigVal = reshape(diag(eigVal), (N, 1))
    eigValExp = exp(eigVal)

    eigVal_row_mat = eigVal * ones(1, N)
    eigVal_diff_mat = eigVal_row_mat - transpose(eigVal_row_mat)

    eigValExp_row_mat = eigValExp * ones(1, N)
    eigValExp_diff_mat = eigValExp_row_mat - transpose(eigValExp_row_mat)

    degenerate_mask = abs(eigVal)
    
    
end