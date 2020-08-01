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


n_steps = 50
dt = 1 / n_steps
n_ctrls = 2

test = (F, G, x) -> GRAPE(F, G, 0 * sz, [sx, sy], ρ, ρₜ, x, n_ctrls, dt, n_steps)

using Optim
init =  rand(n_ctrls, n_steps) .* 0.1 # .+ 1 / sqrt(2) / 2

grad = similar(init)
test(0, grad, init)
grad

res = Optim.optimize(Optim.only_fg!(test), init, Optim.LBFGS(), Optim.Options(show_trace = true, allow_f_increases = false))

# init = rand(n_ctrls, n_steps) .* 0 .+ 1 / sqrt(2) / 2 .+ 0.00001

init = res.minimizer
U_list = pw_evolve_save(0 * sz, [sx,sy], init, 2, dt, n_steps)
# now we propagate the initial state forward in time
ρ_list = [ρ] # need to give it a type it
temp_state = ρ
for U in U_list
    global temp_state = U * temp_state * U'
    append!(ρ_list, [temp_state])
end

ρₜ_list = [ρₜ]
temp_state = ρₜ
for U in reverse(U_list)
    global temp_state = U' * temp_state * U
    append!(ρₜ_list, [temp_state])
end
ρₜ_list = reverse(ρₜ_list)

z = map(x -> real(tr(sz * x)), ρ_list)
z2 = map(x -> real(tr(sz * x)), ρₜ_list)

using Plots
plot(z)
plot!(z2)



##


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
pc = 1 * 2 * pi

# compute the propgators
U_list = pw_evolve_save(H_drift, H_ctrl_arr, x_init * pc, n_ctrls, dt, n_steps)

# now we propagate the initial state forward in time
ρ_list = []
temp_state = ρ
# append!(ρ_list, [temp_state])
for U in U_list
    global temp_state = U * temp_state * U'
    append!(ρ_list, [temp_state])
end

ρₜ_list = []
temp_state = ρₜ
# append!(ρₜ_list, [temp_state])
for U in reverse(U_list)
    global temp_state = U' * temp_state * U
    append!(ρₜ_list, [temp_state])
end
ρₜ_list = reverse(ρₜ_list)

# controls run from k = 1:2
# timeslices run from j = 1: n steps

grad = similar(x_init)
for k = 1:n_ctrls
    for j = 1:n_steps # either you do this or you remove the ρ entry in the list above
        # grad[k, j] = -real(tr((ρₜ_list[j])' * 1.0im * dt * commutator(H_ctrl_arr[k], ρ_list[j])))
        # grad[k, j - 1] = real(tr(ρₜ_list[j + 1]' * commutator(H_ctrl_arr[k], U_list[j]) * ρ_list[j - 1]))
    end
end


k = 1 
j = 2

ρₜ_list[j + 1]'

commutator(H_ctrl_arr[k], U_list[j])





# here's the idea
# I want to be able to set up a problem
# I then call solve on the problem
# solve then uses the algorithm I said, alg, to solve the problem

# looking at something like Dynamo you have a lot of optimisations, sure, but you have an odd? framework
# especially odd since we have the beauty of multiple dispatch on our hands
# so lets start with a naive approach, using expm
# and then implement Shai's stuff
# and finite difference (and AD)