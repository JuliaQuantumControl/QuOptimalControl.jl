"""
a place where we can store all of the algorithms
"""

include("./cost_functions.jl")
include("./evolution.jl")
include("./tools.jl")



"""
Simple. Use Zygote to solve all of our problems
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
    
    # compute all of the 
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

