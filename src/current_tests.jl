# closed systems use common propagators! so thats good!

# okay so lets step through as though we were running a dynamo problem


using QuantumInformation

include("./evolution.jl")
include("./tools.jl")
# just some general stuff, usually stored in a problem type
A = [sz]
B = [sx, sy]
n_ensemble = 1
n_controls = 2
timeslices = 10
duration = 1
X_initial = [1.0 + 0im, 0.0]# pure states to start with
X_target = [0.0, 1.0 + 0.0im]
norm2 = 1.0


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
# loop ove ensembel of systems
# compute the generators
H_list = pw_ham_save(A[1], B, ctrl, n_controls, duration / timeslices, timeslices) .* -1.0im

# now we compute the matrix exponential, here we can cheat a bit, or we don't... depending on whats faster
U_list = exp.(H_list)
# U_list = [expm_exact_gradient(i, duration / timeslices) for i in H_list]



test = expm_exact_gradient(H_list[1], duration / timeslices)

test

# U_list = exp.(H_list)

"""
error function and gradient for closed systems
"""
function error_abs()
    # we need to compute the value of g, defined as the trace matmul over U and K
end

# should be ready to run the optimisation now

println("stop you violated the law, pay the court a fine or serve your sentence")

# # task 1 closed system unitary gate synthesis up to a global phase
# using QuantumInformation
# using StaticArrays
# using LinearAlgebra

# n_pulses = 2
# timeslices = 10
# timestep = 0.1

# Xtarget = 1 / sqrt(2) * (sz * sz)
# X0 = Matrix{ComplexF64}(I(2))

# H_drift = sz # A 
# H_ctrl = [sx, sy] # B

# U_list = pw_evolve_save(H_drift, H_ctrl, rand(n_pulses, timeslices), n_pulses, timestep, timeslices)

# # X_k is the propagator from t_{k-1} to t_k
# # X_{k:0} is the forward propagator of the initial state up to the timeslice k X3*X2*X1*X0
# # X_{M+1:k+1} backward propagated target state up to time t_k


# U = reduce(*, U_list)
# Λ = reduce(*, reverse(U_list))
# N = size(Xtarget)[1]

# k = 1



# v, zeta, exp_d = eig_factors(H_list[1])


# v
# zeta
# exp_d

# temp = v' * H_ctrl[1] * v
# dPdu_eigenbasis = temp .* zeta
# dPdu = v * dPdu_eigenbasis * v'

# # temp = H_v.T.conj() @ B @ H_v
# #     dPdu_eigenbasis = temp * H_eig_factor  # note the elementwise multiplication here
# #     dPdu = H_v @ dPdu_eigenbasis @ H_v.T.conj()  # to computational basis
# #     return dPdu



# # j = 1:m # control Hamiltonian index
# # k = 1:M # timeslice index

# # figure of merit stuff
# g = 1 / N * tr(Utarget' * U)
# gp = g'
# expg = gp / abs(g)

# # exact gradient
# function pw_ham_save(H₀::T, Hₓ_array::Array{T,1}, x_arr::Array{Float64}, n_pulses, timestep, timeslices) where T
#     D = size(H₀)[1] # get dimension of the system
#     K = n_pulses
#     out = []
#     U0 = SMatrix{D,D,ComplexF64}(I(2))
#     for i = 1:timeslices
#     # compute the propagator
#         Htot = SMatrix{D,D,ComplexF64}(H₀ + sum(Hₓ_array .* x_arr[:, i]))
#         # U0 = exp(-1.0im * timestep * Htot)# * U0
#         append!(out, [Htot])
#     end
#     out
# end

# H_list = pw_ham_save(H_drift, H_ctrl, rand(n_pulses, timeslices), n_pulses, timestep, timeslices)

# k = 1
# vals, vecs = eigen(H_list[k])
# # vecs are the columns of the matrix
# vecs[1:2], vals[1]
# vecs[3:4], vals[2]

# if vals[1] == vals[2]P
#     grad = -i * timestep * vecs[1:2]' * H_ctrl[1] * vecs[3:4]
# else
#     grad = -i * timestep * vecs[1:2]' * H_ctrl[1] * vecs[3:4] * (() / ())

#     vals
# end
# # 1 / N * real(
# #     tr(
# #         expg * Λ[k] * 
# #     )
# # )


# vals, vecs = eigen(I(4))