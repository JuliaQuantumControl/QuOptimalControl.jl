using Yota
using LinearAlgebra

using Zygote

# lets write a super basic idea of what we might want
sx = [0.0 + 0.0im 1.0 + 0.0im
      1.0 + 0.0im 0.0 + 0.0im]

function series_exp(A)
    u02 = Array{ComplexF64,2}(I(2))
    u02 + (A + (A * A) / 2) # + ((A^3) / 6 + (A^4) / 24))
end

function evolve(x)
    N = length(x)
    U0 = Array{ComplexF64,2}(I(2))
    for i = 1:N
        ham = ((-1.0im * 0.1) * sx) * x[i] # there is an issue with this line
        U0 = series_exp(ham) * U0
        # U0 = exp(-1.0im * 0.1 * sx * x[i]) * U0
    end
    U0
end

# define some states
ρ = [1.0 + 0.0im, 0.0 + 0.0im]
ρt = [0.0 + 0.0im, 1.0 + 0.0im]

function functional(x)
    U = evolve(x)
    1 - abs2(ρt' * (U * ρ))
end

x_input = rand(10)
functional(x_input)
val, tape = Yota.itrace(functional, x_input)

Zygote.gradient(functional, x_input)

val, tape = Yota.itrace(functional, x_input)
Yota.back!(tape)
Yota.compile!(tape)

function test(G, x)
    val, tape = Yota.itrace(functional, x)
    Yota.back!(tape)
    G .= Yota.GradResult(tape)[1]
end

using Optim

gtest = similar(x_input)
test(gtest, x_input)

Yota.simplegrad(functional, x_input)

res = Optim.optimize(functional, test, x_input, Optim.LBFGS(), Optim.Options(show_trace = true))

grad(functional, x_input)


Yota.simplegrad(functional, x_input)

o = Yota.GradResult(tape)[1]

Yota.back!(tape)
import Base./

# first diffrule needed
@diffrule abs2(x::Number) x real(dy) * (x + x)
@diffrule (/)(u::Array{ComplexF64,2}, v::Number) u dy / v
# expand list of rules for (*) to match complex numbers and arrays
@diffrule *(u::Number, v::Number)            u     v * dy
@diffrule *(u::Number, v::AbstractArray)    u     sum(v .* dy)
@diffrule *(u::AbstractArray, v::Number)            u     v .* dy
@diffrule *(u::AbstractArray, v::AbstractArray)    u     dy * transpose(v)

@diffrule *(u::Number, v::Number)            v     u * dy
@diffrule *(u::Number, v::AbstractArray)    v     u .* dy
@diffrule *(u::AbstractArray, v::Number)            v     sum(u .* dy)
@diffrule *(u::AbstractArray, v::AbstractArray)    v     transpose(u) * dy
