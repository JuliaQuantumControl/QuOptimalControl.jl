abstract type Problem end
abstract type ClosedSystem <: Problem end

mutable struct ClosedStateTransfer <: ClosedSystem
    control_Hamiltonians
    duration
    timestep
    timeslices
    number_pulses
    integration_method
    alg
end