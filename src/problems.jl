abstract type Problem end
abstract type ClosedSystem <: Problem end

"""
Contains all of the information needed to perform a closed state transfer
"""
# TODO need to decide whether or not we work with density matrices or pure states
# in theory we can work with both and dispatch to the correct algorithm given the dimensions of the user input
struct ClosedStateTransfer <: ClosedSystem
    control_Hamiltonians
    state_init # rho (?) 
    state_target
    duration
    timestep
    timeslices
    number_pulses
end