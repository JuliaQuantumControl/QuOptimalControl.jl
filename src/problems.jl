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
@kwdef struct ClosedStateTransfer <: ClosedSystem
    H_ctrl = nothing
    H_drift = nothing
    X_init = nothing
    X_target = nothing
    duration = 1
    timestep = 0.1 # this is extra info
    timeslices = 10
    number_pulses = 1
    n_ensemble = 1
    norm2 = 1.0
    alg = nothing# choose from the struct atm
end

@kwdef struct UnitarySynthesis <: ClosedSystem
    H_ctrl = nothing
    H_drift = nothing
    X_init = nothing
    X_target = nothing
    duration = 1
    timestep = 0.1
    timeslices = 10
    number_pulses = 1
    n_ensemble = 1
    norm2 = 1.0
    alg = nothing
end


# gate syntehsis closed open ClosedSystem
# closed state trnasfer -> Shai exact gradient
# state transfer open system

"""
Working with an experiment
"""
@kwdef struct ExperimentInterface <: Experiment
    duration = 1
    timestep = 0.1
    timeslices = 10
    number_pulses = 1
    start_exp # function to start exp
    pulse_path = ""
    infidelity_path = ""
    timeout = 10
    alg
end