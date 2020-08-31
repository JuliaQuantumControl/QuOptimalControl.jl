using QuOptimalControl

using BenchmarkTools
using ExponentialUtilities
using QuantumInformation
using LinearAlgebra

const sz3 = [1.0+0.0im 0.0; 0.0 -1.0]


drift = [sz3, sz3, sz3]
ctrl = [sx, sy]
n_pulses = 2
timestep = 1/1000
timeslices = 1000
n_ensemble = 3
u0input = Array{ComplexF64,2}(I(2))
input = rand(n_pulses, timeslices)

prob = ClosedStateTransfer(ctrl, drift, [u0input,u0input,u0input], [sz,sz,sz], 1.0, timeslices, n_pulses, n_ensemble, 1, GRAPE_approx(), input)
const dt = prob.duration/prob.n_timeslices


