using Yota
using QuantumInformation
using StaticArrays
using LinearAlgebra
# lets write a super basic idea of what we might want


function pw_evolve(x)
    U0 = SMatrix{2,2,ComplexF64}(I(2))
    N = length(x)
    dt = 1.0 / N

    for i = 1:N
        U0 = exp(-1.0im * dt * sx * x[i]) * U0
    end
    U0
end

pw_evolve([1,2,3,4,5])