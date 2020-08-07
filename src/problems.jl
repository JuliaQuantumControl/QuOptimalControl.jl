abstract type Problem end
abstract type ClosedSystem <: Problem end
abstract type Experiment <: Problem end

"""
Contains all of the information needed to perform a closed state transfer
"""
# TODO need to decide whether or not we work with density matrices or pure states
# in theory we can work with both and dispatch to the correct algorithm given the dimensions of the user input
struct ClosedStateTransfer <: ClosedSystem
    control_Hamiltonians
    drift_Hamiltonians
    state_init
    state_target
    duration
    timestep
    timeslices
    number_pulses
    alg # choose from the struct atm
end

"""
Working with an experiment
"""
struct ExperimentInterface <: Experiment
    duration
    timestep
    timeslices
    number_pulses
    start_exp # function to start exp
    pulse_path
    infidelity_path
    alg
end