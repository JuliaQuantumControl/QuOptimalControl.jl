# want to test and benchmark easily all the time evolution methods
# will do this for both StaticArrays and non StaticArrays, (will assume some 5 spin system to make it sufficiently big)
using BenchmarkTools

SSxI2 = SMatrix{4,4}(kron(SSx, I(2)))
SSyI2 = SMatrix{4,4}(kron(SSy, I(2)))
I4 = SMatrix{4,4}(I(4) .+ 0.0im)


const Hsys_static = SMatrix{4,4}(kron(SSx, SSz))
const HCtrl_static2 = [SSxI2, SSyI2]


n_slices = 500
n_pulses = 2
input_arr = rand(2, 500)

T = 2.0
dt = T/n_slices


BenchmarkTools.@benchmark pw_evolve($Hsys_static, $HCtrl_static2, $input_arr, $n_pulses, $dt, $n_slices)

BenchmarkTools.@benchmark pw_evolve($Hsys_static, $HCtrl_static2, $input_arr, $n_pulses, $dt, $n_slices, $I4)


@code_warntype pw_evolve(Hsys_static, HCtrl_static2, input_arr, n_pulses, dt, n_slices)
