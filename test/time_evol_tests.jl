# want to test and benchmark easily all the time evolution methods
# will do this for both StaticArrays and non StaticArrays, (will assume some 5 spin system to make it sufficiently big)
using BenchmarkTools

# Static Stuff!

SSxI2 = SMatrix{4,4}(kron(SSx, I(2)))
SSyI2 = SMatrix{4,4}(kron(SSy, I(2)))
I4 = SMatrix{4,4}(I(4) .+ 0.0im)

const Hsys_static = SMatrix{4,4}(kron(SSx, SSz))
const HCtrl_static2 = [SSxI2, SSyI2]

# Normal Array (small)

SSxI2_mutable = Array(SSxI2)
SSyI2_mutable = Array(SSyI2)
I4_mutable = Array(I4)

const Hsys_mutable = Array(SMatrix{4,4}(kron(SSx, SSz)))
const HCtrl_mutable2 = Array.([SSxI2, SSyI2])

# general settings
n_slices = 500
n_pulses = 2
input_arr = rand(2, 500)

T = 2.0
dt = T/n_slices

# benchmark static
BenchmarkTools.@benchmark pw_evolve($Hsys_static, $HCtrl_static2, $input_arr, $n_pulses, $dt, $n_slices, $I4)

# benchmark mutable
BenchmarkTools.@benchmark pw_evolve($Hsys_mutable, $HCtrl_mutable2, $input_arr, $n_pulses, $dt, $n_slices, $I4_mutable)

# benchmark static _T
BenchmarkTools.@benchmark pw_evolve_T($Hsys_static, $HCtrl_static2, $input_arr, $n_pulses, $dt, $n_slices, $I4)

# benchmark mutable
BenchmarkTools.@benchmark pw_evolve_T($Hsys_mutable, $HCtrl_mutable2, $input_arr, $n_pulses, $dt, $n_slices, $I4_mutable)

# now we test some of the stuff that actually gets used
BenchmarkTools.@benchmark pw_evolve_save($Hsys_static, $HCtrl_static2, $input_arr, $n_pulses, $dt, $n_slices)

BenchmarkTools.@benchmark pw_evolve_save($Hsys_mutable, $HCtrl_mutable2, $input_arr, $n_pulses, $dt, $n_slices)

BenchmarkTools.@benchmark pw_evolve_save($Hsys_mutable, $HCtrl_mutable2, $input_arr, $n_pulses, $dt, $n_slices)


