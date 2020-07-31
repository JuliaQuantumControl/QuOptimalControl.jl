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
function GRAPE(H_drift, H_ctrl_arr, ρ, ρₜ, x_drive, n_ctrls, dt, n_steps)
    # compute the propgators
    U_list = pw_evolve_save(H_drift, H_ctrl_arr, x_init, n_ctrls, dt, n_steps)

    # now we propagate the initial state forward in time
    ρ_list = [] # need to give it a type it
    
    temp_state = ρ
    for U in U_list
        temp_state = U * temp_state * U'
        append!(ρ_list, [temp_state])
    end

    ρₜ_list = []
    temp_state = ρₜ
    for U in reverse(U_list)
        temp_state = U * temp_state * U'
        append!(ρₜ_list, [temp_state])
    end

    grad = similar(x_init)
    for k = 1:n_ctrls
        for j = 1:n_steps # either you do this or you remove the ρ entry in the list above
            grad[k, j] = -real(tr((ρₜ_list[j])' * 1.0im * dt * commutator(H_ctrl_arr[k], ρ_list[j])))
        end
    end
    
    # compute all of the 
    U = reduce(*, U_list)
    # now lets compute the overlap
    fid = 1.0 - real(tr(ρₜ' * (U * ρ * U')))
    (fid, grad)
end

ρ = [1.0 + 0.0im ; 0.0 + 0.0im]
ρₜ = [0.0 + 0.0im ; 1.0 + 0.0im]

ρ = ρ * ρ'
ρₜ = ρₜ * ρₜ'


# define controls
H_ctrl_arr = [sx, sy]

# define drift
H_drift = 0 * sz

# initial drive guess
n_ctrls = length(H_ctrl_arr)
n_steps = 50
dt = 1 / n_steps

x_init = rand(n_ctrls, n_steps) .* 0.0001


k, f = GRAPE(0 * sz, [sx, sy], ρ, ρₜ, x_init, n_ctrls, dt, n_steps)


@benchmark GRAPE(0 * sz, [sx, sy], ρ, ρₜ, x_init, n_ctrls, dt, n_steps)



using QuantumInformation # right now for convenience
# this might take a bit more work...
# initial state
ρ = [1.0 + 0.0im ; 0.0 + 0.0im]
ρₜ = [0.0 + 0.0im ; 1.0 + 0.0im]

ρ = ρ * ρ'
ρₜ = ρₜ * ρₜ'

# define controls
H_ctrl_arr = [sx, sy]

# define drift
H_drift = 0 * sz

# initial drive guess
n_ctrls = length(H_ctrl_arr)
n_steps = 10
dt = 1 / n_steps

x_init = rand(n_ctrls, n_steps)

# compute the propgators
U_list = pw_evolve_save(H_drift, H_ctrl_arr, x_init, n_ctrls, dt, n_steps)

# now we propagate the initial state forward in time
ρ_list = []
temp_state = ρ
for U in U_list
    global temp_state = U * temp_state * U'
    append!(ρ_list, [temp_state])
end

ρₜ_list = []
temp_state = ρₜ
for U in reverse(U_list)
    global temp_state = U * temp_state * U'
    append!(ρₜ_list, [temp_state])
end


grad = similar(x_init)
for k = 1:n_ctrls
    for j = 1:n_steps # either you do this or you remove the ρ entry in the list above
        grad[k, j] = -real(tr((ρₜ_list[j])' * 1.0im * dt * commutator(H_ctrl_arr[k], ρ_list[j])))
    end
end


@show grad