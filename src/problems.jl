abstract type Problem end
abstract type ClosedSystem <: Problem end
abstract type OpenSystem <: Problem end
abstract type Experiment <: Problem end

import Base.@kwdef

"""
Description of the dynamics of a closed state transfer problem. This also applies method also applies to optimisation of Hermitian operators
"""
@kwdef struct ClosedStateTransfer <: ClosedSystem
    H_ctrl = nothing
    H_drift = nothing
    X_init = nothing
    X_target = nothing
    duration = 1
    timestep = 0.1 # this is extra info
    n_timeslices = 10
    n_pulses = 1
    n_ensemble = 1
    norm2 = 1.0
    alg = nothing # choose from the struct atm
    initial_guess = nothing
end

"""
Description of the dynamics of a unitary synthesis problem. Although this contains the same information as another closed state problem by defining it separately we can choose the evolution methods that make sense for the problem
"""
@kwdef struct UnitarySynthesis <: ClosedSystem
    H_ctrl = nothing
    H_drift = nothing
    X_init = nothing
    X_target = nothing
    duration = 1
    timestep = 0.1
    n_timeslices = 10
    n_pulses = 1
    n_ensemble = 1
    norm2 = 1.0
    alg = nothing
    initial_guess = nothing
end

"""
Working in Liouville space we can do an optimisation in the presence of relaxation, we can also reuse the gradient and goal functions from before. Assuming Hermitian operators on input, need to deal with non-Hermitian.

Following the Khaneja and Glaser paper provided we can use the same gradient and fom functions as in the relaxation free case but the evolution is now governed by the Liovuillian superoperator.
"""
@kwdef struct OpenSystemCoherenceTransfer <: OpenSystem
    H_ctrl = nothing # these might need changed TODO
    H_drift = nothing
    X_init = nothing # in this case some density matrix
    X_target = nothing # Hermitian operator, density matrix
    duration = 1
    timestep = 0.1
    n_timeslices = 10
    n_pulses = 1
    n_ensemble = 1
    norm2 = 1.0
    alg = nothing
    initial_guess = nothing
end

"""
Working with an experiment
"""
@kwdef struct ExperimentInterface <: Experiment
    duration = 1
    timestep = 0.1
    n_timeslices = 10
    n_pulses = 1
    start_exp # function to start exp
    pulse_path = ""
    infidelity_path = ""
    timeout = 10
    alg
end