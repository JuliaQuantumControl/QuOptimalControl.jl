# QuOptimalControl

Quantum optimal control essentially tries to provide numerically optimised solutions to quantum problems in as efficient a manner as possible. 

**If you'd like to help, please get in touch with me! You can reach me @nv_alastair on Twitter or I guess by opening an issue on here...?**

## Usage

Trying to mimic the interface that DifferentialEquations.jl uses we offer several problem definitions. Problems define the dynamics of the quantum system.

Once a problem has been constructed we can use any of the available algorithms (ADGRAPE, GRAPE and dCRAB) to solve the problem. Using multiple dispatch the solver should then get to work.

```julia
using QuantumInformation # provides Pauli matrices for example

ψ1 = [1.0 + 0.0im 0.0]
ψt = [0.0 + 0.0im 1.0]

ρ1 = ψ1' * ψ1
ρt = ψt' * ψt

prob_GRAPE = ClosedStateTransfer(
    B = [sx, sy],
    A = [0.0 * sz],
    X_init = [ρ1],
    X_target = [ρt],
    duration = 1.0,
    n_timeslices = 10,
    n_controls = 2,
    n_ensemble = 1,
    initial_guess = rand(2, 10)
)

sol = GRAPE(prob_GRAPE, inplace = false)
```

And in this case our chosen algorithm, the approximate GRAPE algorithm, will solve the problem automatically, there's no need to define anything else!

Now we can visualise the output pulse:

```julia
visualise_pulse(sol.optimised_pulses, duration = prob.duration)
```

![Bar plot of pulse amplitudes](https://raw.githubusercontent.com/alastair-marshall/QuOptimalControl.jl/master/assets/pulsevis.png "Pulse output")

### Algorithms

For the defined problem types within the package (ClosedStateTransfer, UnitarySynthesis, OpenSystemCoherenceTransfer, ExperimentInterface) there are several predefined solver methods to make it easy to construct and solve common problems. All of the implemented algorithms are also available if the predefined solver methods aren't suitable for the problem type. 



## How does it work?

### What problem are we trying to solve?


![\begin{align*}
\dot{X(t)} = (A + B u_c(t))X(t)
\end{align*}
](https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%5Cbegin%7Balign%2A%7D%0A%5Cdot%7BX%28t%29%7D+%3D+%28A+%2B+B+u_c%28t%29%29X%28t%29%0A%5Cend%7Balign%2A%7D%0A)

Where A is the "drift" term and B is the "control" term and u(t) are time dependent control amplitudes that allow us to modify the state of the system.

**Need to add citations, Shai Machnes, Ville Bergholm, Steffen Glaser, Thomas Schulte-Herbruggen et al.**


## Functionality
* Closed system problems (using GRAPE, dCRAB and Autodiff)
    * State transfer
    * Synthesise unitary propagators
* Open system problems (solving in Liouville space using GRAPE and dCRAB, Autodiff coming)
    * State transfer 
* Closed loop optimisation using dCRAB
    * Use at own risk, improvements coming
* Ensemble optimisation
* Lots of examples (in progress but I'm only one person)

## Why Julia?

Julia is the perfect language to develop these tools in because:
1. it's easy to express mathematics in
2. it's really fast and thats what we care about
3. multiple dispatch makes it easy to write flexible code
4. good python interop (package coming)
5. automatic differentiation tooling is great and improving

This package is a rework of the OCToolbox.jl package but with extra functionality, and more algorithms.


## MVP

What I consider the requirements for a minimimum viable package!

We need to be able to use all of the following algorithms in both open and closed systems efficiently.

### GRAPE

- [ ] Citation of relevant papers
- [ ] Exact gradients from Shai
- [ ] Exact gradients from Steffen
- [ ] Scaling and squaring method

#### ADGRAPE

- [ ] Citation of relevant papers
- [ ] Yota implementation open+closed system (incl. different expm methods)
- [ ] Zygote implementation open system


David Schuster's lab pioneered this idea, essentially use autodiff to compute the gradient of the functional so that you can follow it. Really simple with Zygote but its pretty memory hungry, we can make this more efficient.

### GOAT

- [ ] Citation of relevant papers
- [ ] Implement possibly working prototype

### dCRAB

- [ ] Citation of relevant papers
- [ ] Fix initial conditions
- [ ] Open system master equation
- [ ] Non Nelder-Mead optimisation


**Remote usage** should be possible with this package, see the example where the user defined function simply saves a pulse and then runs some experiment before returning something.

### Krotov

- [ ] Citation of relevant papers
- [ ] Working implementation needed

Monotonic convergence is the name of the game here, not sure that I have the skills to implement it!

### GROUP

- [ ] Citation of relevant papers
- [ ] Combine this with work by Dennis Lucarelli
- [ ] working implementation needed!


Curious to see how this works
