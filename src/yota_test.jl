using Yota
using LinearAlgebra

# lets write a super basic idea of what we might want
sx = [0.0 + 0.0im 1.0 + 0.0im
      1.0 + 0.0im 0.0 + 0.0im]

function evolve(x)
    N = length(x)
    U0 = Array{ComplexF64,2}(I(2))
    for i = 1:N
        ham = -1.0im * 0.1 * sx * x[i]
        # U0 = exp(-1.0im * 0.1 * sx * x[i]) * U0
        U0 = (((u02 + ham) + (ham^2 / 2 + ham^3 / 6)) + ham^4 / 24) * U0
    end
    U0
end

ρ = [1.0 + 0.0im, 0.0 + 0.0im]
ρt = [0.0 + 0.0im, 1.0 + 0.0im]

function functional(x)
    U = pw_evolve2(x)

    1 - abs2(ρt' * (U * ρ))
end

x_input = rand(10)
functional(x_input)
val, tape = Yota.itrace(functional, x_input)

Yota.back!(tape)


import Base./

# this seems to work for abs2
@diffrule abs2(x::Number) x  real(dy) * (x + x)
@diffrule /(Array{ComplexF64,2}, x::Number) (x, y) x / y




function (F̄)
    n = size(A, 1)
    E = eigen(A)
    w = E.values
    ew = exp.(w)
    X = 
    V = E.vectors
    VF = factorize(V)
    Āc = (V * ((VF \ F̄' * V) .* X) / VF)'
    Ā = isreal(A) && isreal(F̄) ? real(Āc) : Āc
    return (Ā,)
end


@adjoint exp(A::AbstractMatrix) = exp(A), function (F̄)
    n = size(A, 1)
    E = eigen(A)
    w = E.values
    ew = exp.(w)
    X = _pairdiffquotmat(exp, n, w, ew, ew, ew)
    V = E.vectors
    VF = factorize(V)
    Āc = (V * ((VF \ F̄' * V) .* X) / VF)'
    Ā = isreal(A) && isreal(F̄) ? real(Āc) : Āc
    return (Ā,)
end



@adjoint abs2(x::Number) = abs2(x), Δ -> (real(Δ) * (x + x),)