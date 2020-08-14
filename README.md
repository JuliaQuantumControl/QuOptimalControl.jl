# QuOptimalControl

Quantum optimal control essentially tries to provide numerically optimised solutions to quantum problems in as efficient a manner as possible. 

**Current status: very WIP!**
**Need to add citations, Shai Machnes, Ville Bergholm, Steffen Glaser, Thomas Schulte-Herbruggen et al.**

## Usage

Trying to mimic the interface that DifferentialEquations.jl uses we offer several problem definitions. Problems define the dynamics of the quantum system.

Included in the problem definition (for now) are some algorithm options to tune the algorithms. 

Once a problem has been constructed and an algorithm tuned we can simply solve the problem and through the wonders of multiple dispatch the code should just work!

```julia
using QuantumInformation

ψ1 = [1.0 + 0.0im 0.0]
ψt = [0.0 + 0.0im 1.0]

ρ1 = ψ1' * ψ1
ρt = ψt' * ψt

prob_GRAPE = ClosedStateTransfer(
    [sx, sy],
    [0.0 * sz],
    ρ1,
    ρt,
    1.0,
    1 / 10,
    10,
    2,
    GRAPE_approx(GRAPE)
)

sol = solve(prob_GRAPE)
```

Then we can analyse the output pulse that is returned!

## Functionality
* Create a closed system problem using a piecewise control and solve using GRAPE (approx. gradients and autodiff) or dCRAB
* Closed loop optimisation (at own risk) using dCRAB
* See Examples.jl for examples!

## Why Julia?

Julia is the perfect language to develop these tools in because:
1. it's easy to express mathematics in
2. it's really fast and thats what we care about
3. multiple dispatch makes it easy to code in
4. Python is inflexible (see multiple dispatch and speed)
5. The automatic differentiation tooling is really good and improving quickly

This package is a rework of the OCToolbox.jl package but with extra functionality, and more algorithms.


## MVP
What I consider the requirements for a minimimum viable package!

We need to be able to use all of the following algorithms in both open and closed systems efficiently.

### GRAPE

**Citation needed**

Algorithm originally developed by the Glaser group, approximate gradient implementation is already here. Exact gradients from the Dynamo package need to be implemented for all of the cases of interest

#### ADGRAPE

David Schuster's lab pioneered this idea, essentially use autodiff to compute the gradient of the functional so that you can follow it. Really simple with Zygote but its pretty memory hungry, we can make this more efficient.

Also want to implement it using Yota.jl because it seems that it's pretty optimised for speed. (We can use Zygote as the fallback for the open system case)

### GOAT

Shai's new algorithm, given the ability of DiffEq.jl we should be able to have the fastest version of this code.


### dCRAB

Fourier (or other basis) expansion of pulse and then optimisation of coefficients using Nelder-Mead. Currently seems to work.

**Remote usage** should be possible with this package, see the example where the user defined function simply saves a pulse and then runs some experiment before returning something.

### Krotov

Monotonic convergence is the name of the game here, not sure that I have the skills to implement it but since I'm a member of QuSCo it's important!

### GROUP

Jacob Sherson's group used gradient optimisation of the coefficients of a dCRAB expansion. Impressive work and curious if it's easy to implement, especially given how easy it is to use Zygote. 

