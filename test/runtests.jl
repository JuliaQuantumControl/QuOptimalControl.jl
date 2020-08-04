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
