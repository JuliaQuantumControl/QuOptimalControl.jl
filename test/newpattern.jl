

# so GRAPE methods use n_slices and so they can store that with themselves
# but dCRAB can be either continuous or pw defined and we can choose what to do 
# we will start with something piecewise since we have the integrators written for that
using Parameters
abstract type Basis end

# think about what defines a basis and the fourier basis in particular
# its always going to have the form A1 * sin(omega1 * t + phi1) + A2 * cos(omega1 * t + phi2)
Base.@kwdef mutable struct Fourier2{F,C} <: Basis
    frequencies::F = [0.0]
    coefficients::C = [(0.0, 0.0)]
end

Base.@kwdef mutable struct Sigmoid{F} <: Basis
    random_stuff::F
end

# this is only true for the fourier basis, but the point is, we could define this for every basis individually
# that gives us a method to have a unified interface
function evaluate_basis!(basis::Fourier2, out, time_axis)
    # might also want to return the lambda sometimes?

    # lam = get_lambda(basis)
    lam = function (t)
        res = 0.0
        @unpack frequencies, coefficients = basis
        # lets imagine we store tuples
        n_freq = length(frequencies)
        for i = 1:n_freq
            coeffs_i = coefficients[i]
            res =
                res +
                coeffs_i[1] * sin(frequencies[i] * t) +
                coeffs_i[2] * cos(frequencies[i] * t)
        end
        return res
    end

    out .= lam.(time_axis)
    return
end

# now lets set up an example using the bases
my_fourier_components = Fourier2([1.0, 2.0], [(1.0, 2.0), (3.0, 4.0)])


my_fourier_components.frequencies[2] = rand()

append!(my_fourier_components.coefficients, [(4.0, 5.0)])
my_fourier_components.coefficients


# using Plots
# using BenchmarkTools

time_axis = 0:0.001:5.0
hold = similar(time_axis)
evaluate_basis!(my_fourier_components, hold, time_axis)


using Plots
plot(hold)
# @benchmark evaluate_basis($my_fourier_components, $time_axis)
# @code_warntype evaluate_basis(my_fourier_components, time_axis)


# @benchmark evaluate_basis($my_fourier_components, $hold, $time_axis)

Base.@kwdef mutable struct dCRAB{ITG,B,FD,IIP,OPTS,MI,SI}
    # characteristic of the integrator 
    integrator_type::ITG # or if you want to do it continuously we have another method!
    basis::B = Fourier() # store bases
    freq_dist::FD = rand # distribution of frequencies
    isinplace::IIP = true
    optim_options::OPTS = Optim.Options()
    max_iters::MI = 100
    num_SI::SI = 5 # number of superiterations you want to do
end



# separate struct that defines the different solvers that are possible


prob = Problem(
    B = [Sx, Sy],
    A = Sz,
    Xi = ρinit,
    Xt = ρfin,
    T = 1.0,
    n_controls = 2,
    guess = rand(2, 10),
    sys_type = StateTransfer(),
)

# sol = solve(prob, GRAPE(n_slices = 10, isinplace = true))


# we need to get a functional f(x) which maps a sampled array to a figure of merit
# the array x will be the previous Fourier alongside the 
function _get_functional(prob)
    @unpack B, A, Xi, Xt, T, n_controls, guess, sys_type = prob
    D = size(A, 1)
    u0 = typeof(A)(I(D))

    topt = function (x)
        U = pw_evolve(A, B, x, n_controls, T / 10, 10, u0)
        ev = U * Xi * U'
        return C1(Xt, ev)
    end
    return topt
end


func = _get_functional(prob)

algg = dCRAB(n_timeslices = 10)

@unpack n_timeslices, basis, freq_dist, isinplace, optim_options, max_iters, num_SI = algg
time_axis = collect(range(0.0, prob.T, length = n_timeslices))
# for n_si = 1:num_SI
n_si = 1
# draw a frequency
ω = freq_dist()
# now from the basis we need to know that we have two coefficients, we draw them at randomly
init_coeffs = Tuple(rand(2))
init_phases = Tuple(rand(2))

# now the point is that these are our parameters in the optimisation

# so we can update the basis function I guess, since we know that we're indexing into it
# at n_si every time
basis.frequencies[n_si] = ω
basis.coefficients[n_si] = init_coeffs
basis.phases[n_si] = init_phases

test_store = zeros(n_timeslices)

evaluate_basis!(basis, test_store, time_axis)

func(test_store)
#end



struct testnew3{T,K}
    name::T
    age::K
end


mytest = testnew3("Alastair", 11)
mytest


using Parameters

@unpack name, age = mytest

function getname(test::testnew)
    @show "hi world " * test.name
end

getname(testnew("Alastair"))
