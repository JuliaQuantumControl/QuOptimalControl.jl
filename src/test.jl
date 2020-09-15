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


function GRAPE_testing!(A::T, B, u_c, n_timeslices, duration, n_controls, gradient, U_k, L_k, gens, props, X_init, X_target, evolve_store, prob) where T
    dt = duration / n_timeslices
    U_k[1] .= X_init
    L_k[end] .= X_target

    pw_ham_save!(A, B, u_c, n_controls, n_timeslices, @view gens[:])
    @views props[:] .= exp.(gens[:] * (-1.0im * dt))

    for t = 1:n_timeslices
        evolve_func!(prob, t, 1, U_k, L_k, props, gens, evolve_store, forward = true)
    end

    for t = reverse(1:n_timeslices)
        evolve_func!(prob, t, 1, U_k, L_k, props, gens, evolve_store, forward = false)
    end

    t = n_timeslices
    
    for c = 1:n_controls
        for t = 1:n_timeslices
            @views gradient[c, t] = grad_func!(prob, t, dt, B[c], U_k, L_k, props, gens, evolve_store)
        end
    end

    return fom_func(prob, t, U_k, L_k, props, gens)

end

function grad_func2!(t, dt, B, U, L, props, gens, store)::Float64
    @views mul!(store, L[t]', commutator(B, U[t]))

    real(tr((1.0im * dt) .* store))
end

function fom_func2(t, U, L, props, gens)::Float64
    # recall that target is always the last entry of L
    # and that we have in U[end] the propagated forward target state
    @views C1(L[t], U[t])
end

Aarray = Array(A)
Barray = Array.(B)
ρ0A = Array(ρ0)
ρTA = Array(ρT)

UtA, LtA, gensA, propsA, domA, gradientA = QuOptimalControl.init_GRAPE(ρ0A, n_timeslices, 1, Aarray, n_controls)

evs = similar(gensA[1])
@benchmark GRAPE_testing!($Aarray, $Barray, $u_c, $n_timeslices, $duration, $n_controls, $gradientA[1,:,:], $UtA, $LtA, $gensA, $propsA, $ρ0A, $ρTA, $evs, $Val(ClosedStateTransfer))


@benchmark GRAPE_testing!($Aarray, $Barray, $u_c, $n_timeslices, $duration, $n_controls, $1.0, $UtA, $LtA, $gensA, $propsA, $ρ0A, $ρTA)

@benchmark GRAPE_testing!($Aarray, $Barray, $u_c, $n_timeslices, $duration, $n_controls, $1.0, $UtA, $LtA, $gensA, $propsA, $ρ0A, $ρTA)


function GRAPE!(F, G, x, U, L, gens, props, fom, gradient, A, B, n_timeslices, n_ensemble, duration, n_controls, prob, evolve_store, weights)

    dt = duration / n_timeslices
    for k = 1:n_ensemble
        U[1, k] = prob.X_init[k]
        L[end, k] = prob.X_target[k]


        # do we need to actually save the generators of the transforms or do we simply need the propagators?
        pw_ham_save!(A[k], B[k], x, n_controls, n_timeslices, @view gens[:,k])
        # @views props[:, k] .= ExponentialUtilities._exp!.(Gen[:,k] .* (-1.0im * dt))
        @views props[:, k] .= exp.(gens[:, k] .* (-1.0im * dt))

        # forward propagate
        for t = 1:n_timeslices
            evolve_func!(prob, t, k, U, L, props, gens, evolve_store, forward = true)
        end
        
        # prob backwards in time
        for t = reverse(1:n_timeslices)
            evolve_func!(prob, t, k, U, L, props, gens, evolve_store, forward = false)
        end
        
        t = n_timeslices # can be chosen arbitrarily
        fom[k] = fom_func(prob, t, k, U, L, props, gens)

        # we can optionally compute this actually
        for c = 1:n_controls
            for t = 1:n_timeslices
                # might want to alter this to just pass the matrices that matter rather than everything
                @views gradient[k, c, t] = grad_func!(prob, t, dt, k, B[k][c], U, L, props, gens, evolve_store)
            end
        end
            
    end

    # then we average over everything
    if G !== nothing
        @views G .= sum(weights .* gradient, dims = 1)[1,:, :]
    end

    if F !== nothing
        return sum(fom .* weights)
    end

end