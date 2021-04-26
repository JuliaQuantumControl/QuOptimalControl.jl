



prob = StateTransferProblem(
    B = [Sx, Sy],
    A = Sz,
    X_init = ρinit,
    X_target = ρfin,
    duration = 5.0,
    n_timeslices = 25,
    n_controls = 2,
    initial_guess = rand(2, 25)
)


ens = ClosedEnsembleProblem(prob, 5, A_gens, B_gens, X_init_gens, X_target_gens, ones(5)/5)

sol = GRAPE(ens, inplace=true)
@test sol.result[1].minimum - C1(ρfin, ρfin) < tol * 10







# lets do some sort of exact gradient test



prob = StateTransferProblem(
    B = [Sx, Sy],
    A = Sz,
    X_init = ρinit,
    X_target = ρfin,
    duration = 1.0,
    n_timeslices = 10,
    n_controls = 2,
    initial_guess = rand(2, 10)
)
dt = prob.duration/prob.n_timeslices

pw_ham_save_bad(prob.A, prob.B, prob.initial_guess, prob.n_controls, dt, prob.n_timeslices
)

function pw_ham_save_bad(H₀::T, Hₓ_array::Array{T,1}, x_arr::Array{Float64}, n_pulses, timeslices) where T
    out = Vector{T}(undef, timeslices)
    for i = 1:timeslices
        Htot = H₀
        for j = 1:n_pulses
            @views Htot = Htot + Hₓ_array[j] * x_arr[j, i]
        end
        out[i] = Htot
    end
    out
end
