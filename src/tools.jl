"""
Just some useful functions
"""

using BSON
using DelimitedFiles
using StaticArrays
using Setfield

#const σₓ = SMatrix{} # etc



"""
Compute the commutator between two matrices A and B
"""
function commutator(A, B)
    A * B - B * A
end

# mul!ing over a in-place commutator :D (I need to get out more)
function commutator!(A, B, store)
    mul!(store, A, B)

    A * B - B * A
end


"""
Function that will take an optimal pulse and a solution result and simulate the pulse again returning the output so you can check the gate is correct
"""
function test_pulse(prob, solres)
    @show "not implemented"


end


"""
Initialise an ensemble from an ensemble problem definition. Right now this creates an array of problems just now, not sure if this is the best idea though.
"""
function init_ensemble(ens)
    ensemble_problem_array = [deepcopy(ens.prob) for k = 1:ens.n_ens]
    for k = 1:ens.n_ens
        prob_to_update = ensemble_problem_array[k]
        prob_to_update = @set prob_to_update.A = ens.A_g(k)
        prob_to_update = @set prob_to_update.B = ens.B_g(k)
        prob_to_update = @set prob_to_update.Xi = ens.XiG(k)
        prob_to_update = @set prob_to_update.Xt = ens.XtG(k)
        ensemble_problem_array[k] = prob_to_update
    end
    ensemble_problem_array # vector{ens.problem}
end


"""
Save a SolutionResult to file
"""
function save(solres, file_path)
    # convert the solres file into a dict for saving

    solres_dict = Dict()
    solres_keys = fieldnames(typeof(solres))[2:end]

    for k in solres_keys
        push!(solres_dict, k => getfield(solres, k))
    end


    bson(file_path, solres_dict)
end


"""
Load a SolutionResult from file
"""
function load(file_path)
    solres_dict = BSON.load(file_path)
    sol = SolutionResult(
        nothing,
        solres_dict[:fidelity],
        solres_dict[:problem_info],
        solres_dict[:optimised_pulses],
    )
end

"""
Write an array called pulse to a file, could also write a time array if that's necessary.
"""
function pulse_to_file(pulse, file_path)
    # can maybe check the shape and reshape so that its always time going down the file
    open(file_path, "w") do io
        writedlm(io, pulse')
    end
end

function pulse_to_file(pulse, file_path, duration)
    # can maybe check the shape and reshape so that its always time going down the file
    N = length(pulse)
    time_array = collect(range(0, duration, length = N))
    open(file_path, "w") do io
        writedlm(io, [time_array pulse'])
    end
end
