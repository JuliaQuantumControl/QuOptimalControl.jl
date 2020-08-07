using QuOptimalControl
using Test

@testset "QuOptimalControl.jl" begin
    # Write your tests here.

    # okay we want to have some interface that looks like this

    prob = ClosedStateTransfer(
        [sx, sy], # controls
        [sz], # drift
        n_ensemble = 1, # number of ensemble members
        drift_range = 0., # range of the coefficients of the input ensemble members
    
    )

    sol = solve(prob, alg = GRAPE(), integration_method = pw_evolve()) # copy interface from DiffEq because its beautiful


end


function pw_evolve_T(H₀::T, Hₓ_array::Array{T,1}, x_arr::Array{Float64}, n_pulses, timestep, timeslices)::T where T
    x_arr = real.(complex.(x_arr))
    D = size(H₀)[1] # get dimension of the system
    K = n_pulses
    U0 = I(D)
    for i = 1:timeslices
        # compute the propagator
        Htot = H₀ + sum(Hₓ_array .* x_arr[:, i])
        U0 = exp(-1.0im * timestep * Htot) * U0
    end
    U0
end

using Zygote

pw_evolve_T(0 * sz, [sx, sy], rand(2, 10), 2, 1 / 10, 10)

function test(x)
    U = pw_evolve_T(0 * sz, [sx, sy], x, 2, 1 / 10, 10)
    real(tr(sz * (U * ρ1 * U')))
end
test(rand(2, 10))

Zygote.gradient(test, rand(2, 10))