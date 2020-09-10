using QuOptimalControl

using BenchmarkTools
using ExponentialUtilities
using QuantumInformation
using LinearAlgebra

const sz3 = [1.0+0.0im 0.0; 0.0 -1.0]

ρinit = [1.0 0.0]' * [1.0 0.0+0im]
ρfin = [0.0 1.0]' * [0.0 1.0+0im]


drift = [sz3, sz3, sz3]
ctrl = [sx, sy]
n_pulses = 2
timestep = 1/1000
timeslices = 1000
n_ensemble = 3
u0input = Array{ComplexF64,2}(I(2))
input = rand(n_pulses, timeslices)

prob = ClosedStateTransfer(ctrl, drift, [ρinit,ρinit,ρinit], [ρfin,ρfin,ρfin], 1.0, timeslices, n_pulses, n_ensemble, 1, GRAPE_approx(), input)
const dt = prob.duration/prob.n_timeslices


(U, L, G_array, P_array, g, grad) = init_GRAPE(prob.X_init[1], prob.n_timeslices, prob.n_ensemble, prob.A[1], prob.n_controls)
wts = ones(prob.n_ensemble)
evolve_store = similar(U[:,1][1])

G = similar(input)
@benchmark GRAPE!(1, $G, $prob.initial_guess, $U, $L, $G_array, $P_array, $g, $grad, $prob.A, $prob.B, $prob.n_timeslices, $prob.n_ensemble, $prob.duration, $prob.n_controls, $prob, $evolve_store, $wts)

@benchmark benchGRAPE!(1, $G, $prob.initial_guess, $U, $L, $G_array, $P_array, $g, $grad, $prob.A, $prob.B, $prob.n_timeslices, $prob.n_ensemble, $prob.duration, $prob.n_controls, $prob, $prob.X_init, $prob.X_target, $evolve_store, $wts)

function benchGRAPE!(F, G, x, U, L, Gen, P_list, g, grad, A, B, n_timeslices, n_ensemble, duration, n_controls, prob, X_init, X_target, evolve_store, weights)
    dt = duration / n_timeslices
    for k = 1:n_ensemble
        U[1, k] = X_init[k]
        L[end, k] = X_target[k]


        # do we need to actually save the generators of the transforms or do we simply need the propagators?
        # why does this sometimes fail?
        pw_ham_save!(A[k], B, x, n_controls, n_timeslices, @view Gen[:,k])
        # we can probably do this better I think
        # not sure about whether to use exp or this exp! here
        # could do multiple dispatch... to a something
        @views P_list[:, k] .= ExponentialUtilities._exp!.(Gen[:,k] .* (-1.0im * dt))

        # forward propagate
        for t = 1:n_timeslices
            evolve_func!(prob, t, k, U, L, P_list, Gen, evolve_store, forward = true)
        end
        
        # prob backwards in time
        for t = reverse(1:n_timeslices)
            evolve_func!(prob, t, k, U, L, P_list, Gen, evolve_store, forward = false)
        end
        
        t = n_timeslices # can be chosen arbitrarily
        g[k] = fom_func(prob, t, k, U, L, P_list, Gen)

        # we can optionally compute this actually
        # in order for this to be general it uses looks for a gradient definition function
        for c = 1:n_controls
            for t = 1:n_timeslices
                # might want to alter this to just pass the matrices that matter rather than everything
                @views grad[k, c, t] = grad_func(prob, t, dt, k, B[c], U, L, P_list, Gen, evolve_store)
            end
        end
            
    end

    # then we average over everything
    if G !== nothing
        @views G .= sum(weights .* grad, dims = 1)[1,:, :]
    end

    if F !== nothing
        return sum(g .* weights)
    end

end



@benchmark pw_ham_save!($prob.A[1], $prob.B, $input, $prob.n_controls, $prob.n_timeslices, @view $G_array[:,1])

@benchmark pw_gen_save!($prob, $prob.A[1], $prob.B, $input, $prob.n_controls, $prob.n_timeslices, $prob.duration, @view $P_array[:,1])


@benchmark pw_gen_save!($prob, $prob.A[1], $prob.B, $input, $prob.n_controls, $prob.n_timeslices, $prob.duration, @view $P_array[:,1])

@benchmark pw_prop_save!($prob.A[1], $prob.B, $input, $prob.n_controls, $prob.n_timeslices, $prob.duration, $@view P_array[:,1])


function exp_test(x, A, B, n_controls, n_timeslices, G, P, k)
    pw_ham_save!(A[k], B, x, n_controls, n_timeslices, @view G[:,k])
    # we can probably do this better I think
    # not sure about whether to use exp or this exp! here
    # could do multiple dispatch... to a something
    @views P[:, k] .= ExponentialUtilities._exp!.(G[:,k] .* (-1.0im * dt))
end

@benchmark exp_test($input, $prob.A, $prob.B, $prob.n_controls, $prob.n_timeslices, $G_array, $P_array, $1)

function exp_gen_test(x, A, B, n_controls, n_timeslices, duration, G, P, k, prob)
    pw_gen_save!(prob, A[k], B, x, n_controls, n_timeslices, duration, @view G[:,k])
    # we can probably do this better I think
    # not sure about whether to use exp or this exp! here
    # could do multiple dispatch... to a something
    @views P[:, k] .= ExponentialUtilities._exp!.(G[:,k])
end

function exp_prop_test(x, A, B, n_controls, n_timeslices, duration, G, P, k, prob)
    # @benchmark pw_prop_save!($prob.A[1], $prob.B, $input, $prob.n_controls, $prob.n_timeslices, $prob.duration, $@view P_array[:,1])
    pw_prop_save!(A[1], B, x, n_controls, n_timeslices, duration, @view P[:,k])
        # pw_gen_save!(prob, A[k], B, x, n_controls, n_timeslices, duration, @view G[:,k])
    # we can probably do this better I think
    # not sure about whether to use exp or this exp! here
    # could do multiple dispatch... to a something
    # @views P[:, k] .= ExponentialUtilities._exp!.(G[:,k])
end

@benchmark exp_gen_test($input, $prob.A, $prob.B, $prob.n_controls, $prob.n_timeslices, $prob.duration, $G_array, $P_array, $1, $prob)


@code_warntype exp_gen_test(input, prob.A, prob.B, prob.n_controls, prob.n_timeslices, prob.duration, G_array, P_array, 1, prob)

@benchmark exp_prop_test($input, $prob.A, $prob.B, $prob.n_controls, $prob.n_timeslices, $prob.duration, $G_array, $P_array, $1, $prob)



function pw_ham_save!(H₀::T, Hₓ_array::Array{T,1}, x_arr::Array{Float64,2}, n_pulses, timeslices, out) where T
    K = n_pulses
    Htot = similar(H₀)

    @views @inbounds for i = 1:timeslices
        @. Htot = 0.0 * Htot
        @inbounds for j = 1:K
            @. Htot = Htot + Hₓ_array[j] * x_arr[j, i]
        end
        @. out[i] = Htot + H₀
    end
end

"""
An almost allocation free (1 alloc in my tests) version of the Generator saving function, this saves the generators of propagators (implementation might differ for prob)
"""
function pw_gen_save!(prob::Union{ClosedStateTransfer,UnitarySynthesis}, H₀::T, Hₓ_array::Array{T,1}, x_arr::Array{Float64,2}, n_pulses, timeslices, duration, out) where T
    K = n_pulses
    Htot = similar(H₀)
    dt = timeslices / duration

    @views @inbounds for i = 1:timeslices
        @. Htot = 0.0 * Htot
        @inbounds for j = 1:K
            @. Htot = Htot + Hₓ_array[j] * x_arr[j, i]
        end
        @. out[i] = (Htot + H₀) .* (-1.0im * dt)
    end
end

function pw_prop_save!(H₀::T, Hₓ_array::Array{T,1}, x_arr::Array{Float64,2}, n_pulses, timeslices, duration, out) where T
    K = n_pulses
    Htot = similar(H₀)
    dt = duration/timeslices

    @views @inbounds for i = 1:timeslices
        @. Htot = 0.0 * Htot
        @inbounds for j = 1:K
            @. Htot = Htot + Hₓ_array[j] * x_arr[j, i]
        end
        @. out[i] = exp((Htot + H₀) .* (-1.0im * dt))
    end
end

pw_ham_save!(prob.A[1], prob.B, input, prob.n_controls, prob.n_timeslices, @view G_array[:,1])



pw_prop_save!(prob.A[1], prob.B, input, prob.n_controls, prob.n_timeslices, prob.duration, @view P_array[:,1])






#


# current test



#


# closed systems use common propagators! so thats good!

# okay so lets step through as though we were running a dynamo problem


using QuantumInformation

include("./evolution.jl")
include("./tools.jl")
# just some general stuff, usually stored in a problem type
B = [sx, sy]
A = [sz]
n_ensemble = 1
n_controls = 2
timeslices = 10
duration = 1
X_initial = [1.0 + 0im, 0.0]# pure states to start with
X_target = [0.0, 1.0 + 0.0im]
norm2 = 1.0

X_initial = X_initial * X_initial'
X_target = X_target * X_target'

# since we can dispatch based on the type of the system then we can assume that we know 1. which error func to use and also 2. which method for gradient

# pure state transfer
# that means we use eig and abs error
# if we have some ensembles we need to essentially set up a list of X_target
X_target = repeat(X_target, n_ensemble)
X_initial = repeat(X_initial, n_ensemble)
norm2 = repeat([norm2], n_ensemble)


# lets create a set of controls
init = rand(n_controls, timeslices) # should allow some control over this
# tau can vary but lets keep it constant, if it varies it must be defined for each time slice


ctrl = copy(init)

# might define elsewhere, will also define the length [ timeslices + 1, ensemble]
U = repeat([similar(X_initial)], timeslices + 1, n_ensemble)
L = repeat([similar(X_target)], timeslices + 1, n_ensemble)
H_list # same except only timeslices long
P_list # as above
g = zeros(n_ensemble)

# loop over ensemble of systems
k = 1
U[1,k] = X_initial
L[end, k] = X_target


# compute the generators
H_list = pw_ham_save(A[k], B, ctrl, n_controls, duration / timeslices, timeslices) .* -1.0im
# now we compute the matrix exponential, here we can cheat a bit, or we don't... depending on whats faster
P_list = exp.(H_list)
# P_list = [expm_exact_gradient(i, duration / timeslices) for i in H_list]
all_vals = [eig_factors(i * duration / timeslices, antihermitian = true) for i in H_list]
# eig_factors(G; antihermitian = false, tol = 1e-10)
# v, zeta, exp_d = eig_factors(dt_H, antihermitian = true)

# ULMIXED
# and now we propagate stuff
for t = 1:timeslices
    # P_list here needs a k
    U[t + 1, k] = P_list[t] * U[t] * P_list[t]'
end

for t = reverse(1:timeslices)
    L[t, k] = P_list[t]' * L[t + 1, k] * P_list[t]
end

# here it doesnt matter where we compute the error in time
g_ind = 1
# now we want to compute the error
g[k] = real(trace_matmul(L[g_ind, k], U[g_ind, k]))

# computing the gradient too
grad = zeros(n_controls, timeslices)
for j = 1:n_controls
    for t = 1:timeslices
        dPdU_tail = dPdU(all_vals[t][1], all_vals[t][2], B[j]) * P_list[t]'
        grad[j, t] = real(-1.0 * trace_matmul(L[t + 1, k], dPdU_tail))
    end
end

function dPdU(H_v, H_eig_factor, B)
    temp = (H_v' * B * H_v) .* H_eig_factor
    dPdU = H_v * temp * H_v'
end



test = expm_exact_gradient(H_list[1], duration / timeslices)

test


##
H_ctrl = [sx, sy]
H_drift = [sz]
n_ensemble = 1
n_controls = 2
timeslices = 10
duration = 1
X_initial = Matrix{ComplexF64}(I(2))# pure states to start with
X_target = sz
norm2 = 1.0


X_target = repeat([X_target], n_ensemble)
X_initial = repeat([X_initial], n_ensemble)
norm2 = repeat([norm2], n_ensemble)


init = rand(n_controls, timeslices) # should allow some control over this

ctrl = copy(init)

# might define elsewhere, will also define the length [ timeslices + 1, ensemble]
U = repeat([similar(X_initial[1])], timeslices + 1, n_ensemble)
L = repeat([similar(X_target[1])], timeslices + 1, n_ensemble)
G_list = repeat([similar(H_drift[1])], timeslices, n_ensemble) # same except only timeslices long
P_list = repeat([similar(H_drift[1])], timeslices, n_ensemble)# as above

# g is the error or figure of merit
g = zeros(n_ensemble)
grad = zeros(n_controls, timeslices, n_ensemble)


# loop over ensemble of systems
for k = 1:n_ensemble
    U[1,k] = X_initial[k]
    L[end, k] = X_target[k]

    # compute and store the generators
    G_list[:, k] = pw_ham_save(A[k], B, ctrl, n_controls, duration / timeslices, timeslices) .* -1.0im * duration / timeslices
    # now we compute the matrix exponential
    P_list[:,k] = exp.(G_list)

    # prop forwards in time
    for t = 1:timeslices
        U[t + 1, k] = P_list[t, k] * U[t, k]
    end
    
    # prob backwards in time
    for t = reverse(1:timeslices)
        L[t, k] = P_list[t, k]' * L[t + 1, k]
    end

    t = timeslices # can be chosen arbitrarily
    g[k] = tr(L[t, k]' * U[t, k]) * tr(U[t, k]' * L[t, k])

    # compute the gradient too
    grad[:,:,1]

    for i = 1:n_controls
        for t = 1:timeslices
            grad[i, t, k] = -2.0 * real(tr(L[t, k]' * 1.0im * duration / timeslices * H_ctrl[i] * U[t, k]) * tr(U[t, k]' * L[t, k]))
        end
    end
end


grad = similar(x_drive)
for k = 1:n_ctrls
    for j = 1:n_steps
        grad[k, j] = real(tr((ρₜ_list[j])' * 1.0im * dt * commutator(H_ctrl_arr[k], ρ_list[j])))
            # grad[k, j] = -real(tr(ρₜ_list[j + 1]' * commutator(H_ctrl_arr[k], U_list[j]) * ρ_list[j])) * pc
    end
end