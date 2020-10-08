# # tests of some new designs of GRAPE algorithms for the closedstatetransfer right now

# # lets handle the static array case first
# # if your problem is small enough you should use static arrays, these will have in-place = false


# # thinking about how the penalty functional should work

# myC1 = x -> C1(x, #whatever else)
# myC2 = x -> C2(x, ...)


# penalty = PenaltyFunctionals([1.0, 2.0, 3.0], [myC1, myC2, myC3])
# prob = ClosedStateTransfer(# control stuff etc, penalty)

# # then in the code we can evaluate penalty functional 
# # when we return the FoM we can evaluate sum(weights .* functions.(x))
# # and we can also evaluate the gradient using AD (probably)









# ################################################################





# ################################################################

# using StaticArrays
# using QuantumOpticsBase
# b = SpinBasis(1//2)
# sx = SMatrix{2,2}(DenseOperator(sigmax(b)).data)
# sy = SMatrix{2,2}(DenseOperator(sigmay(b)).data)
# sz = SMatrix{2,2}(DenseOperator(sigmaz(b)).data)
# su = [1.0, 0.0+0im]
# sd = [0, 1.0 + 0im]
# ρ0 = SMatrix{2,2}(kron(su', su))
# ρT = SMatrix{2,2}(kron(sd', sd))

# A = [-10*sz, 0*sz, 10*sz]
# B = [[sx, sy], [sx, sy], [sx, sy]]
# n_controls = 2
# n_timeslices = 1000
# duration = 1.0
# n_ensemble = 3
# u_c = rand(n_controls, n_timeslices)
# const dt = duration/n_timeslices

# prob = ClosedStateTransfer(B[1], A[1], ρ0, ρT, duration, n_timeslices, n_controls, 1, 1.0, 1, 1)

# using BenchmarkTools

# # grape function should still be an inplace update
# # we should assume constant ensemble number
# # return fom and we update gradient vector inplace
# #k = 1
# # drift, control term, controls, 
# #function static_grape(prob::ClosedStateTransfer, A, B, u, n_timeslices, duration)

# Ut, Lt, gens, props, dom, gradient = QuOptimalControl.init_GRAPE(ρ0, n_timeslices, n_ensemble, A[1], n_controls)

# # we allocate these to not be mutable but rather undefined, we will copy into them later (but thats free if things are static)
# Ut = Vector{typeof(A[1])}(undef, n_timeslices + 1)
# Lt = Vector{typeof(A[1])}(undef, n_timeslices + 1)


# function sGRAPE(A::T, B, u_c, n_timeslices, duration, n_controls, gradient_a, U_k, L_k, X_init, X_target, prob) where T
    
#     dt = duration / n_timeslices
#     # arrays that hold static arrays?
#     U_k[1] = X_init
#     L_k[end] = X_target


#     # now we want to compute the generators 
#     gens = pw_ham_save(A, B, u_c, n_controls,n_timeslices)
#     props = exp.(gens .* (-1.0im * dt))


#     # forward evolution of states
#     for t = 1:n_timeslices
#         U_k[t+1] = evolve_func(prob, t, U_k, L_k, props, gens, forward = true)
#     end
#     # backward evolution of costates
#     for t = reverse(1:n_timeslices)
#         L_k[t] = evolve_func(prob, t, U_k, L_k, props, gens, forward = false)
#     end

#     t = n_timeslices

#     # update the gradient array
#     for c = 1:n_controls
#         for t = 1:n_timeslices
#             @views gradient_a[c, t] = grad_func(prob, t, dt, B[c], U_k, L_k, props, gens)

#         end
#     end

#     return fom_func(prob, t, U_k, L_k, props, gens)
# end


# # now we want to do an optimisation
# function to_optim(F, G, x)
#     fom = 0.0
#     for k=1:n_ensemble
#         fom += @views sGRAPE(A[k], B[k], x, n_timeslices, duration, n_controls, gradient[k, :, :], Ut,Lt, ρ0, ρT, prob) # * weights[k]
#     end

#     if G !== nothing
#         @views G .= sum(gradient, dims = 1)[1, :, :]#o
#     end
#     if F !== nothing
#         return fom
#     end
# end

# o = similar(gradient[1, :, :])

# to_optim(1.0, o, opt.minimizer)

# using Optim
# opt = Optim.optimize(Optim.only_fg!(to_optim), u_c .* pi, Optim.LBFGS(), Optim.Options(g_abstol = 1e-5, show_trace = true))

# sGRAPE(A[2], B[1], o.minimizer, n_timeslices, duration, n_controls, gradient[1,:,:], Ut,Lt, ρ0, ρT, prob)

# using QuOptimalControl
# visualise_pulse(o.minimizer, 2)

# struct dootdoot2
# inplace
# end

# struct test2
#     a
# end


# function first(x)
#     _first(x, x.a)
# end

# function _first(x, a::dootdoot2)
#     _first(x, a, Val(a.inplace))
# end

# function _first(x, a, inp::Val{true})
#     @show "inplace true"
# end


# function _first(x, a, inp::Val{false})
#     @show "inplace false"
# end


# a = dootdoot2(true)
# b = dootdoot2(false)

# inp1 = test2(a)
# inp2 = test2(b)

# first(inp2)

# f = test(10, false)
# t = test(0, true)

# function kss(p)
#     _kss(p, Val(p.inplace))
# end

# function _kss(p, inp::Val{true})
#     @show "hi im new hered"
#     @show p.a
# end

# function _kss(p, inp::Val{false})
#     @show "how do I work, doot"
#     @show p.a
#     @show inp
# end


# kss(f)


# using StaticArrays


# s = @SVector zeros(3)

# typeof(s)

# function doot(x::SArray)
#     @show typeof(x)
# end

# function doot(x::Array)
#     @show typeof(x)
# end

# doot(Array(s))






using ForwardDiff, QuantumInformation

using LinearAlgebra

H_ctrl(alpha, t) = alpha[1] * t * sx .+ alpha[2]
H0 = sz
H(alpha, t) = H_ctrl(alpha, t) + H0

ForwardDiff.jacobian(a -> H(a, 1.), [0.5, 0.6])


u0 = Array{ComplexF64}(I(2))

alpha = [0.5, 0.6]
t = 0.1

Ham = H(alpha, t)
dalpha_H = ForwardDiff.jacobian(a -> H(a, t), alpha)
zero = 0 .* zeros(size(dalpha_H'))


firstrow = Ham * u0
#secondrow:
N = length(alpha)
o = []
for i = 1:N
    append!(o, [dalpha_H[2*i-1:2*i, :] * u0 + Ham * similar(u0)])
end



function test(du, u, p, t)
    alpha = p
    Ham = H(alpha, t)
    zero = 0 .* Ham
    dalpha = ForwardDiff.jacobian(a -> H(a, t), alpha)

    du = []
    append!(du, )
    du[1] .= Ham * u[1]
    @show typeof(du[2])
    N = length(alpha)
    for i = 1:N
        x =  dalpha[2*i-1:2*i, :] * u[1] + Ham * u[i+1]
        @show typeof(x)
        du[i+1] .= dalpha[2*i-1:2*i, :] * u[1] + Ham * u[i+1]
    end
end

function test2(u, p, t)
    alpha = p
    Ham = H(alpha, t)
    zero = 0 .* Ham
    dalpha = ForwardDiff.jacobian(a -> H(a, t), alpha)
    out = []
    append!(out, [Ham * u[1]])

    N = length(alpha)
    for i = 1:N
        x =  dalpha[2*i-1:2*i, :] * u[1] + Ham * u[i+1]
        append!(out, [x])
    end
    out
end

using DifferentialEquations

test2(init, [0.5, 0.6], 0.1)

type = eltype(dalpha_H[1:2, :])
type = eltype(dalpha[1:2, :] * u[1] + Ham * u[2])
init = [u0, zeros(type, size(dalpha_H[1:2,:])),  zeros(type, size(dalpha_H[1:2,:]))]

test(similar.(init), init, [0.5, 0.6], 0.1)



p = [0.5, 0.6]
du = similar.(init)
u = init

alpha = p
Ham = H(alpha, t)
zero = 0 .* Ham
dalpha = ForwardDiff.jacobian(a -> H(a, t), alpha)

du[1] .= Ham * u[1]
@show typeof(du[2])
N = length(alpha)
for i = 1:N
    x =  dalpha[2*i-1:2*i, :] * u[1] + Ham * u[i+1]
    @show typeof(x)
    du[i+1] .= dalpha[2*i-1:2*i, :] * u[1] + Ham * u[i+1]
end

# come back to this later



function test(x)
    for i = 1:length(x)
        x[i] = i
    end
end

z = zeros(2, 6)
test(@views z[2, :])