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

A = sz
B = [sx, sy]
n_controls = 2
n_timeslices = 1000
duration = 1.0
u_c = rand(n_controls, n_timeslices)

using BenchmarkTools

# grape function should still be an inplace update
# we should assume constant ensemble number
# return fom and we update gradient vector inplace
#k = 1
# drift, control term, controls, 
#function static_grape(prob::ClosedStateTransfer, A, B, u, n_timeslices, duration)

Ut, Lt, gens, props, dom, gradient = QuOptimalControl.init_GRAPE(ρ0, n_timeslices, 1, A, n_controls)

# we allocate these to not be mutable but rather undefined, we will copy into them later (but thats free if things are static)
Ut = Vector{typeof(A)}(undef, n_timeslices + 1)
Lt = Vector{typeof(A)}(undef, n_timeslices + 1)

# in the case of static arrays pw_ham_save is the best option
function static_grape(A::T, B, u_c, n_timeslices, duration, n_controls, gradient_store, U_k, L_k, xinit, xtarget) where T

    dt = duration / n_timeslices
    # array that holds static arrays?
    
    # U_k = Vector{T}(undef, n_timeslices)
    # L_k = Vector{T}(undef, n_timeslices)
    U_k[1] = xinit
    L_k[end] = xtarget


    # now we want to compute the generators 
    gens = pw_ham_save(A, B, u_c, n_controls,n_timeslices)
    props = exp.(gens .* (-1.0im * dt))


    # forward evolution of states
    for t = 1:n_timeslices
        U_k[t+1] = my_evolvefn(t, U_k, L_k, props, 1, 1, forward = true)
    end
    # backward evolution of costates
    for t = reverse(1:n_timeslices)
        L_k[t] = my_evolvefn(t, U_k, L_k, props, 1, 1, forward = false)
    end

    t = n_timeslices

    # update the gradient array
    for c = 1:n_controls
        for t = 1:n_timeslices
            @views gradient_store[c, t] = real((1.0im * dt) .* tr(L_k[t]' * commutator(B[c], U_k[t]) ))
            # real(tr( L_k[t]' * (1.0im * dt) * commutator(B[c], U_k[t]) ))

        end
    end
    grad_func(prob, t, dt, B, U, L, props, gens)

    return C1(L_k[t], U_k[t])

end

const dt = duration/n_timeslices
@benchmark real(tr(Lt[1000]' * (1.0im * dt) .* commutator(B[1], Ut[1000])))


@benchmark real((1.0im*dt) .* tr($Lt[1000]' * commutator($B[1], $Ut[1000])))

real((1.0im*dt) .* tr(Lt[1000]' * commutator(B[1], Ut[1000])))
real(tr(Lt[1000]' * (1.0im * dt) .* commutator(B[1], Ut[1000])))

@benchmark tr($B[1])


@benchmark commutator($B[1], $Ut[2] )

@benchmark static_grape($A, $B, $u_c, $n_timeslices, $duration,  $n_controls, $res, $Ut, $Lt, $ρ0, $ρT)

@benchmark static_grape($A, $B, $u_c, $n_timeslices, $duration,  $n_controls, $res, $Ut, $Lt, $ρ0, $ρT)

res = gradient[1, :, :]

static_grape(A, B, u_c, n_timeslices, duration, n_controls,res , Ut, Lt, ρ0, ρT)



function my_evolvefn(t, U, L, P_list, Gen, store;forward = true)
    if forward
        P_list[t] * U[t] * P_list[t]'
    else
        P_list[t]' * L[t + 1] * P_list[t]
    end
end


function bench(t, dt, B, U, L, props, gens)
    -2.0 * real((-1.0im * dt)* 
        tr(L[t]' * B * U[t]) * tr(U[t]' * L[t]) 
    )
end


function bench2(t, dt, B, U, L, props, gens)
    real(
        tr(
            (1.0im * dt) * (L[t]' * commutator(B, U[t]))
        )
    )
end


@benchmark bench2($1, $dt, $B[1], $Ut, $Lt, $1, $1)