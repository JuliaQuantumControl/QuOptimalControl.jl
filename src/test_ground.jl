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
pulse = zeros(1,  timeslices)

# now lets set up the actual optimisation

# the users functional should take a drive as an input and return the infidelity
function user_functional(x)
    U = pw_evolve(0 * sz, [sx], x, 1, dt, timeslices)
    1 - C1(ρt, U * ρ1 * U')
end

user_functional(rand(1, 10))

# for each frequency we perform the following procedure, we assume some given functional by the user
# we wrap the user given functional in a little bit of code to track things
for (i, freq) in enumerate(init_freq)
    pulse_time = 0:dt:duration - dt # fix this I guess

    # since we use NM to optimise only the coefficients
    # pulse = pulse + ansatz.(init_coeffs[i, :], freq, t)

    function to_minimizer(x)
        # what comes in here is from Nelder Mead
        
        # firstly lets create an internal copy of the pulse as it exists just now
        test_pulse = copy(pulse)
        test_pulse += ansatz.(x, freq, pulse_time)
        user_functional(test_pulse)
    end

    # now we optimise
    res = Optim.optimize(to_minimizer, init_coeffs[i, :], Optim.NelderMead(), Optim.Options(show_trace = true, allow_f_increases = false))

    # we need some sort of rejection here but.... assuming for now that it succeeds in optimising

    # then we can firstly store the result
    append!(optimised_coeffs, [res.minimizer])

    # but also update the pulse
    pulse += reshape(ansatz.(res.minimizer, freq, pulse_time), 1, 10)

end

# lets step through it

i, freq = 1, init_freq[1]
pulse_time = 0:dt:duration - dt
function to_minimizer(x)
    # what comes in here is from Nelder Mead
    
    test_pulse = copy(pulse)
    @show (x,)
    test_pulse += reshape(ansatz.((x,), freq, pulse_time), (1, 10))
    user_functional(test_pulse)
end



result = Optim.optimize(to_minimizer, init_coeffs[i, :], Optim.NelderMead(), Optim.Options(show_trace = true, allow_f_increases = false))

pulse += ansatz.(result.minimizer, freq, pulse_time)

# what does dCRAB actually entail? 
# use nelder mead to optimise some coefficients, you need to provide it a handwritten functional, that functional should return the infidelity
