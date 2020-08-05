include("./problems.jl")
include("./algorithms.jl")

using QuantumInformation


struct ClosedStateTransferTest2 <: ClosedSystem
    control_Hamiltonians
    drift_Hamiltonians
    state_init
    state_target
    duration
    timestep
    timeslices
    number_pulses
end

ψ1 = [1.0 + 0.0im 0.0]
ψt = [0.0 + 0.0im 1.0]

ρ1 = ψ1' * ψ1
ρt = ψt' * ψt

# need to figure out how keyword constructors work
prob = ClosedStateTransferTest2(
    [sx, sy],
    [0.0 * sz],
    ρ1,
    ρt,
    1.0,
    1 / 10,
    10,
    2 
)

function solve(problem)
    println("no algorithm, using default for problem type")
    # then use approx GRAPE for everything
end


function solve(problem::ClosedSystem, alg)
    # use the provided algorithm to solve the problem
    
    # in this case lets use GRAPE as an example
    test = (F, G, x) -> alg(F, G, problem.drift_Hamiltonians, problem.control_Hamiltonians, problem.state_init, problem.state_target, x, problem.number_pulses, problem.timestep, problem.timeslices)

    # generate a random initial guess for the algorithm if the user hasn't provided one
    init = rand(problem.number_pulses, problem.timeslices) .* 0.01

    res = Optim.optimize(Optim.only_fg!(test), init, Optim.LBFGS(), Optim.Options(show_trace = true, allow_f_increases = false))
    return res
end

sol = solve(prob, GRAPE)

using Plots
bar(sol.minimizer[1, :], ylabel = "Control amplitude", xlabel = "Index", label = "1")
bar!(sol.minimizer[2, :], label = "2")



# implement dCRAB
# okay so a few thoughts here, dCRAB optimises the coefficients of a Fourier series
# traditionally it uses the Nelder-Mead algorithm to do this. 

# had this idea last night about providing alg_options to the algorithms that's different to the information that we pass to the step taking algorithm, this is about the actual algorithm

# testing params
n_pulses = 2
duration = 1
timeslices = 10
dt = 1 / 10


# alg specific params
n_freq = 10 # number of frequency components
n_coeff = 2 # number of coefficients


# the users functional should take a drive as an input and return the infidelity
function user_functional(x)
    # we get a list of pulses for now, so lets cat them into an array
    U = pw_evolve(0 * sz, [sx, sy], vcat(x...), 1, dt, timeslices)
    1 - C1(ρt, U * ρ1 * U')
end

user_functional([rand(1, 10), rand(1, 10)])


# starting over from the a fresh repl

function dCRAB(n_pulses, dt, timeslices, duration, n_freq, n_coeff, user_func)

    # lets set up an ansatz that will currently be for Fourier series
    ansatz(coeffs, ω, t) = coeffs[1] * cos(ω * t) + coeffs[2] * sin(ω * t)

    # initially randomly chosen frequencies (can refine this later)
    init_freq = rand(n_freq, n_pulses)

    # not sure the best way to handle multiple pulses at the moment sadly
    # we do this per pulse?
    init_coeffs = rand(n_freq, n_coeff, n_pulses)

    optimised_coeffs = []

    pulses = [zeros(1, timeslices) for i in n_pulses]

    pulse_time = 0:dt:duration - dt

    # now we loop over everything

    for i = 1:n_freq
        freqs = init_freq[i, :] # so this contains the frequencies for all of the pulses

        # wrap the user defined function, which should accept a list of pulses, ofc. it'll be 1dim if there's just one pulse
        function to_minimize(x)
            # copy pulses 
            copy_pulses = copy(pulses)

            # I find getting indices hard, want to divide up the array x into n_coeff chunks
            first(j) = (j - 1) * n_coeff + 1
            second(j) = j * n_coeff

            [copy_pulses[j] += reshape(ansatz.((x[first(j):second(j)],), freqs[j], pulse_time), (1, timeslices)) for j = 1:n_pulses]
            user_func(copy_pulses)
        end

        # now optimise with nelder mead

        result = Optim.optimize(to_minimize, reshape(init_coeffs[i, :, :], 4), Optim.NelderMead(), Optim.Options(show_trace = true, allow_f_increases = false))

        # update the pulses, save the coefficients
        [pulses[j] += reshape(ansatz.((x[first(j):second(j)],), freqs[j], pulse_time), (1, timeslices)) for j = 1:n_pulses]
        append!(optimised_coeffs, [result.minimizer])
    end
    return optimised_coeffs, pulses

end






