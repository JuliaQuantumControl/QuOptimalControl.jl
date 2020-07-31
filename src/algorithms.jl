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


using QuantumInformation # right now for convenience
using StaticArras
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
for U in U_list
    append!(ρ_list, [U * ρ * U'])
end

ρₜ_list = []
for U in reverse(U_list)
    append!(ρₜ_list, [U * ρₜ * U'])
end


grad = similar(x_init)
for k = 1:n_ctrls
    for j = 1:n_steps
        grad[k, j] = -real(tr((ρₜ_list[j])' * 1.0im * dt * commutator(H_ctrl_arr[k], ρ_list[j])))
    end
end


