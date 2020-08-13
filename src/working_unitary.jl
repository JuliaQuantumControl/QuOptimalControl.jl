
# working synthesis of unitary transformations using first order gradient
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
