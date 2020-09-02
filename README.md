# QuOptimalControl

Quantum optimal control essentially tries to provide numerically optimised solutions to quantum problems in as efficient a manner as possible. 

**Current status: very WIP!**

**If you'd like to help, please get in touch with me! You can reach me @nv_alastair on Twitter or I guess by opening an issue on here...?**

**Need to add citations, Shai Machnes, Ville Bergholm, Steffen Glaser, Thomas Schulte-Herbruggen et al.**

## Usage

Trying to mimic the interface that DifferentialEquations.jl uses we offer several problem definitions. Problems define the dynamics of the quantum system.

Included in the problem definition (for now) are some algorithm options to tune the algorithms. 

Once a problem has been constructed and an algorithm tuned we can simply solve the problem and through the wonders of multiple dispatch the code should just work!

```julia
using QuantumInformation # provides Pauli matrices for example

ψ1 = [1.0 + 0.0im 0.0]
ψt = [0.0 + 0.0im 1.0]

ρ1 = ψ1' * ψ1
ρt = ψt' * ψt

prob_GRAPE = ClosedStateTransfer(
    H_ctrl = [sx, sy],
    H_drift = [0.0 * sz],
    X_init = ρ1,
    X_target = ρt,
    duration = 1.0,
    timestep = 1 / 10,
    n_timeslices = 10,
    n_pulses = 2,
    n_ensemble = 1,
    norm2 = 1.0,
    GRAPE_approx()
)

sol = solve(prob_GRAPE)
```

And the approximate GRAPE algorithm will solve the problem automatically, there's no need to define anything else!

Now we can visualise the output pulse:

```julia
visualise_pulse(sol.optimised_pulses, duration = 1.0)
```

![Bar plot of pulse amplitudes](https://raw.githubusercontent.com/alastair-marshall/QuOptimalControl.jl/master/assets/pulsevis.png "Pulse output")



### How does it work?

We utilise Julia's multiple dispatch (where the Julia compiler decides which code to run based on the types of the problem) to keep the code clean. This means that when the solve function is called it passes over to a specific implementation of the algorithm for the problem in hand.

Using GRAPE as an example, there is a lot of similarity between different problem types when calculating the propagators but the gradient, figure of merit and evolution functions are different. We use multiple dispatch to choose the correct set of functions (this doesn't incur any time cost because the types are known). 


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
3. multiple dispatch makes it easy to write flexile code
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
