"""
store all of the algorithms
"""


"""
Simple. Use Zygote to solve all of our problems
"""
function ADGRAPE()
end


"""
Actual GRAPE
"""


using Plots


ρₜ_list = []
temp_state = ρₜ
for U in reverse(U_list)
    global temp_state = U * temp_state * U'
    append!(ρₜ_list, [temp_state])
end



# density matrix simulation of rabi oscillation
H_ctrl = [2 * pi * sx]
n_steps = 50 # 10 // 2
dt = 1 / 50
x_init = ones(1, 50)

U_list = pw_evolve_save(H_drift, H_ctrl, x_init, 1, dt, n_steps)

temp_state = ρₜ
test_list = []
append!(test_list, [temp_state])
for U in reverse(U_list)
    global temp_state = U * temp_state * U'
    append!(test_list, [temp_state])
end

z = map(x -> real(tr(sz * x)), test_list)

plot!(z)


println("Hi")


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