abstract type Problem end
abstract type ClosedSystem <: Problem end
abstract type Experiment <: Problem end

import Base.@kwdef

# useful little thing in REPL: fieldnames(ClosedStateTransfer) or fieldnames(typeof(ClosedStateTransfer()))

"""
Contains all of the information needed to perform a closed state transfer
"""
# TODO need to decide whether or not we work with density matrices or pure states
# in theory we can work with both and dispatch to the correct algorithm given the dimensions of the user input

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


# gate syntehsis closed open ClosedSystem
# closed state transfer -> Shai exact gradient
# state transfer open system

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