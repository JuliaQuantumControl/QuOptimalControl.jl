# tests of some new designs of GRAPE algorithms for the closedstatetransfer right now

# lets handle the static array case first
# if your problem is small enough you should use static arrays, these will have in-place = false


using StaticArrays
using QuantumOpticsBase
b = SpinBasis(1//2)
sx = SMatrix{2,2}(DenseOperator(sigmax(b)).data)
sy = SMatrix{2,2}(DenseOperator(sigmay(b)).data)
sz = SMatrix{2,2}(DenseOperator(sigmaz(b)).data)
su = [1.0, 0.0+0im]
sd = [0, 1.0 + 0im]
ρ0 = SMatrix{2,2}(kron(su', su))
ρT = SMatrix{2,2}(kron(sd', sd))

A = [-10*sz, 0*sz, 10*sz]
B = [[sx, sy], [sx, sy], [sx, sy]]
n_controls = 2
n_timeslices = 1000
duration = 1.0
n_ensemble = 3
u_c = rand(n_controls, n_timeslices)
const dt = duration/n_timeslices

prob = ClosedStateTransfer(B[1], A[1], ρ0, ρT, duration, n_timeslices, n_controls, 1, 1.0, 1, 1)

using BenchmarkTools

# grape function should still be an inplace update
# we should assume constant ensemble number
# return fom and we update gradient vector inplace
#k = 1
# drift, control term, controls, 
#function static_grape(prob::ClosedStateTransfer, A, B, u, n_timeslices, duration)

Ut, Lt, gens, props, dom, gradient = QuOptimalControl.init_GRAPE(ρ0, n_timeslices, n_ensemble, A[1], n_controls)

# we allocate these to not be mutable but rather undefined, we will copy into them later (but thats free if things are static)
Ut = Vector{typeof(A[1])}(undef, n_timeslices + 1)
Lt = Vector{typeof(A[1])}(undef, n_timeslices + 1)


function sGRAPE(A::T, B, u_c, n_timeslices, duration, n_controls, gradient_a, U_k, L_k, X_init, X_target, prob) where T
    
    dt = duration / n_timeslices
    # arrays that hold static arrays?
    U_k[1] = X_init
    L_k[end] = X_target


    # now we want to compute the generators 
    gens = pw_ham_save(A, B, u_c, n_controls,n_timeslices)
    props = exp.(gens .* (-1.0im * dt))


    # forward evolution of states
    for t = 1:n_timeslices
        U_k[t+1] = evolve_func(prob, t, U_k, L_k, props, gens, forward = true)
    end
    # backward evolution of costates
    for t = reverse(1:n_timeslices)
        L_k[t] = evolve_func(prob, t, U_k, L_k, props, gens, forward = false)
    end

    t = n_timeslices

    # update the gradient array
    for c = 1:n_controls
        for t = 1:n_timeslices
            @views gradient_a[c, t] = grad_func(prob, t, dt, B[c], U_k, L_k, props, gens)

        end
    end

    return fom_func(prob, t, U_k, L_k, props, gens)
end


# now we want to do an optimisation
function to_optim(F, G, x)
    fom = 0.0
    for k=1:n_ensemble
        fom += @views sGRAPE(A[k], B[k], x, n_timeslices, duration, n_controls, gradient[k, :, :], Ut,Lt, ρ0, ρT, prob) # * weights[k]
    end

    if G !== nothing
        @views G .= sum(gradient, dims = 1)[1, :, :]#o
    end
    if F !== nothing
        return fom
    end
end

o = similar(gradient[1, :, :])

to_optim(1.0, o, opt.minimizer)

using Optim
opt = Optim.optimize(Optim.only_fg!(to_optim), u_c .* pi, Optim.LBFGS(), Optim.Options(g_abstol = 1e-5, show_trace = true))

sGRAPE(A[2], B[1], o.minimizer, n_timeslices, duration, n_controls, gradient[1,:,:], Ut,Lt, ρ0, ρT, prob)

using QuOptimalControl
visualise_pulse(o.minimizer, 2)

struct dootdoot2
inplace
end

struct test2
    a
end


function first(x)
    _first(x, x.a)
end

function _first(x, a::dootdoot2)
    _first(x, a, Val(a.inplace))
end

function _first(x, a, inp::Val{true})
    @show "inplace true"
end


function _first(x, a, inp::Val{false})
    @show "inplace false"
end


a = dootdoot2(true)
b = dootdoot2(false)

inp1 = test2(a)
inp2 = test2(b)

first(inp2)

f = test(10, false)
t = test(0, true)

function kss(p)
    _kss(p, Val(p.inplace))
end

function _kss(p, inp::Val{true})
    @show "hi im new hered"
    @show p.a
end

function _kss(p, inp::Val{false})
    @show "how do I work, doot"
    @show p.a
    @show inp
end


kss(f)


using StaticArrays


s = @SVector zeros(3)

typeof(s)

function doot(x::SArray)
    @show typeof(x)
end

function doot(x::Array)
    @show typeof(x)
end

doot(Array(s))

