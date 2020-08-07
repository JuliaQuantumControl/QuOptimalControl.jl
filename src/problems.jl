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
    control_Hamiltonians = nothing
    drift_Hamiltonians = nothing
    state_init = nothing
    state_target = nothing
    duration = 1
    timestep = 0.1
    timeslices = 10
    number_pulses = 1
    n_ensemble = 1
    alg = nothing# choose from the struct atm
end

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
    alg
end