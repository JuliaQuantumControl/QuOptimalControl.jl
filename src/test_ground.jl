include("./problems.jl")
include("./algorithms.jl")

using QuantumInformation


mutable struct ClosedStateTransferTest2 <: ClosedSystem
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
n_pulses = 1
duration = 1
timeslices = 10
dt = 1 / 10


# alg specific params
n_freq = 10 # number of frequency components
n_coeff = 2 # number of coefficients

# lets create a little ansatz, can tidy this up to just make it vector * vector
ansatz(coeffs, ω, t) = coeffs[1] * cos(ω * t) + coeffs[2] * sin(ω * t)


# generate some random initial values, for each frequency we need n_coeffs

# initial frequencies
init_freq = rand(n_freq)

# initial coefficients
init_coeffs = rand(n_freq, n_coeff)

# unsure how this should be stored really
optimised_coeffs = []

# previously evaluated pulse, this is the trick I think
pulse = zeros(timeslices)

pulse_time = 0:dt:duration
# now lets set up the actual optimisation



pw_evolve(H₀::T, Hₓ_array::Array{T,1}, x_arr::Array{Float64}, n_pulses, timestep, timeslices)::T where T
# for each frequency we perform the following procedure

for (i, freq) in enumerate(init_freq)
    

    # we need to write this to be of the form func(x) = res


    pulse = pulse + ansatz.(init_coeffs[i, :], freq, t)

    pw_evolve(0 * sz, [sx], )
end


#
