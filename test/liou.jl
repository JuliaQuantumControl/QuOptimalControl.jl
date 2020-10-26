# test idea of Liouville evolution


using LinearAlgebra
using QuantumInformation

H_ctrl = [pi * sx, pi * sy]
H0 = sz

function to_superoperator(H)
    D = size(H)[1]
    kron(I(D), H)  - kron(H', I(D))
end



H0 = to_superoperator(sz)
H_ctrls = to_superoperator.(H_ctrl)
duration = 1
n_slices = 10
drive = rand(2, n_slices)

function prop(x)
    Htotal = H0
    for k = 1:2
        Htotal = Htotal + x[k] * H_ctrls[k]
    end
    U = exp(-im * Htotal * dt)
end


p = []
for s = 1:n_slices
    p_n = prop(drive[:,s])

    append!(p, [p_n])
end


psi = [1;0.0 + 0.0im]

rho0 = psi * psi'
# stack it into vector
rho0 = reshape(rho0, 4)

o = [rho0]
for s = 1:n_slices
    o_n = p[s] * o[s]
    append!(o, [o_n])
end

obs = to_superoperator(sz)
z = []
for s = 1:n_slices
    # temp = obs * o[s]
    kk = reshape(o[s], (2,2))
    out = real(tr(sz * kk))
    append!(z, out)
end
using Plots
plot(z)