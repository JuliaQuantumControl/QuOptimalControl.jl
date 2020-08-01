"""
store all of the algorithms
"""


"""
Simple. Use Zygote to solve all of our problems
"""
function ADGRAPE()
end


"""
Function that evaluates the figure of merit and computes the gradient, returned as a tuple I guess
"""
function GRAPE(F, G, H_drift, H_ctrl_arr, ρ, ρₜ, x_drive, n_ctrls, dt, n_steps)
    # x_init = reshape(x_init, (n_ctrls, n_steps))
    # compute the propgators
    U_list = pw_evolve_save(H_drift, H_ctrl_arr, x_drive, n_ctrls, dt, n_steps)

    # now we propagate the initial state forward in time
    ρ_list = [ρ] # need to give it a type itρ
    temp_state = ρ
    for U in U_list
        temp_state = U * temp_state * U'
        append!(ρ_list, [temp_state])
    end

    ρₜ_list = [ρₜ]
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
    
    # compute all of the 
    U = reduce(*, U_list)
    # now lets compute the infidelity to minimize
    fid = 1.0 - abs2(tr(ρₜ * (U * ρ * U')))
    
    if G !== nothing
        G .= grad
    end

    if F !== nothing
        return fid
    end

end


test = (F, G, x) -> GRAPE(F, G, 0 * sz, [sx, sy], ρ, ρₜ, x, n_ctrls, dt, n_steps)

res = Optim.optimize(Optim.only_fg!(test), init, Optim.LBFGS(), Optim.Options(show_trace = true, allow_f_increases = false))

