

using QuOptimalControl
using Test

@testset "QuOptimalControl.jl" begin


    # test GRAPE initially
    # closed state trnasfer problem initially

    ρinit = [1.0 0.0]' * [1.0 0.0+0im]
    ρfin = [0.0 1.0]' * [0.0 1.0+0im]
    
    prob = ClosedStateTransfer(
        B = [σx, σy],
        A = [σz],
        X_init = [ρinit],
        X_target = [ρfin],
        duration = 1.0,
        n_timeslices = 10,
        n_controls = 2,
        n_ensemble = 1,
        norm2 = 1.0,
        alg = GRAPE_approx(),
        initial_guess = rand(2, 10)
    )

    sol = solve(prob)

    # check if the result is close to the target or not
    


    # check if we can do unitary synthesis for a pi pulse
    U_init = Matrix{ComplexF64}(I(2))
    ϕ = π
    U_final = Matrix{ComplexF64}([cos(ϕ) sin(ϕ); -sin(ϕ) cos(ϕ)])

    prob = UnitarySynthesis(
        B = [σx, σy],
        A = [σz],
        X_init = [U_init],
        X_target = [U_final],
        duration = 1.0,
        n_timeslices = 10,
        n_controls = 2,
        n_ensemble = 1,
        norm2 = 1.0,
        alg = GRAPE_approx(),
        initial_guess = rand(2, 10)
    )


    sol = solve(prob)

    # we need to test the open system

    # prob = OpenSystemCoherenceTransfer(
    #     B = [σx, σy],
    #     A = [σz],
    #     X_init = [ρinit],
    #     X_target = [ρfin],
    #     duration = 1.0,
    #     n_timeslices = 10,
    #     n_controls = 2,
    #     n_ensemble = 1,
    #     norm2 = 1.0,
    #     alg = GRAPE_approx(),
    #     initial_guess = rand(2, 10)
    # )


    # sol = solve(prob)



    # lets test the dCRAB algorithms

    prob = ClosedStateTransfer(
        B = [σx, σy],
        A = [σz],
        X_init = [ρinit],
        X_target = [ρfin],
        duration = 1.0,
        n_timeslices = 10,
        n_controls = 2,
        n_ensemble = 1,
        norm2 = 1.0,
        alg = dCRAB_options(),
        initial_guess = rand(2, 10)
    )

    sol = solve(prob)




    # lets test the AD algorithms

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