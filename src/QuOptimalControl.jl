module QuOptimalControl

include("problems.jl")
include("cost_functions.jl")
include("tools.jl")
include("timeevolution.jl")
include("algorithms.jl")
include("dCRAB.jl")
include("GRAPE.jl")
include("solve.jl")
include("visualisation.jl")

export C1, C2, C3, C4, C5, C6, C7, fom_func
export commutator, eig_factors, expm_exact_gradient, trace_matmul, save, load
export pw_evolve, pw_evolve_save, pw_evolve_T, pw_ham_save, pw_ham_save!, pw_gen_save!, evolve_func, evolve_func!
export algorithm, gradientBased, gradientFree, GRAPE_approx, GRAPE_AD, dCRAB_options, ADGRAPE, GRAPE, dCRAB, init_GRAPE, GRAPE_new, GRAPE!
export solve
export grad_func
export Problem, ClosedSystem, Experiment, StateTransferProblem, UnitaryProblem, ClosedEnsembleProblem, OpenSystemCoherenceTransfer, ExperimentInterface
export visualise_expt_val, visualise_pulse

end
