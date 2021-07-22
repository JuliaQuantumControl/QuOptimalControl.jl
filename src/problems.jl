abstract type SystemType end
# what should these be? I want to dispatch on them but never create them
# so maybe abstract type is fine?
abstract type ClosedSystem <: SystemType end
abstract type OpenSystem <: SystemType end

# these are the problem types
struct StateTransfer <: ClosedSystem end
struct UnitaryGate <: ClosedSystem end
struct CoherenceTransfer <: OpenSystem end

abstract type Experiment <: SystemType end

import Base.@kwdef # unsure about using this so much


# could do something like this
# and then we do the dispatch on the system type rather than the problem itself
@kwdef struct Problem{BT,AT,XT,TT,NC,IG,ST}
    B::BT # control terms
    A::AT # drift terms
    Xi::XT # initial state
    Xt::XT # target operator or state
    T::TT # duration of pulse
    n_controls::NC # number of pulses
    guess::IG # guess at controls
    sys_type::ST
end

"""
Define an ensemble of problems, can provide any method for computing the new drift or control Hamiltonians
"""
@kwdef struct EnsembleProblem{T,NE,AG,BG,XIG,XTG,WTS}
    prob::T # for an ensemble of systems you'll provide one "template" problem
    n_ens::NE # we need to know the number of ensemble members
    A_g::AG # need to find a better name but these will give us some discretisation
    B_g::BG # similarly to above
    XiG::XIG
    XtG::XTG
    wts::WTS # weights for calculating the figure of merit and gradient
end


"""
Working in Liouville space we can do an optimisation in the presence of relaxation, we can also reuse the gradient and goal functions from before. Assuming Hermitian operators on input, need to deal with non-Hermitian.

Following the Khaneja and Glaser paper provided we can use the same gradient and fom functions as in the relaxation free case but the evolution is now governed by the Liovuillian superoperator.
# """
# @kwdef struct OpenSystemCoherenceTransfer{BT, AT, XT, T, NS, NC, IG} <: OpenSystem
#     B_control::BT # control terms
#     A_drift::AT # drift terms
#     X_init::XT # initial state
#     X_target::XT # target operator or state
#     duration::T # duration of pulse
#     n_timeslices::NS # slices
#     n_controls::NC # number of pulses
#     initial_guess::IG # guess at controls
# end

"""
Working with an experiment
"""
@kwdef struct ExperimentInterface{T,TS,NC,FS,PP,IP,TO} <: Experiment
    duration::T
    timestep::TS
    n_controls::NC
    start_exp::FS # function to start exp
    pulse_path::PP
    infidelity_path::IP
    timeout::TO
end
