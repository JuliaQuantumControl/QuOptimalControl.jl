module QuOptimalControl

# more to come
include("cost_functions.jl")
include("tools.jl")
include("evolution.jl")
include("algorithms.jl")
include("solve.jl")

export C1, C2, C3, C4, C5, C6, C7
export commutator, eig_factors, expm_exact_gradient, trace_matmul
export pw_evolve, pw_evolve_save, pw_evolve_T, pw_ham_save
export algorithm, gradientBased, gradientFree, GRAPE_approx, GRAPE_AD, dCRAB_type, ADGRAPE, GRAPE, dCRAB
export solve


end
