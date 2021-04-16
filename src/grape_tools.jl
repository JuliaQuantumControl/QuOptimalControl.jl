

"""
Initialise all the storage arrays that will be used in a GRAPE optimisation
"""
function init_GRAPE(X, n_timeslices, n_ensemble, A, n_controls)
        # states
        states = [similar(X) for i = 1:n_timeslices + 1, j = 1:n_ensemble]
        # costates
        costates = [similar(X) for i = 1:n_timeslices + 1, j = 1:n_ensemble]
        # exp(gens)
        propagators = [similar(A) for i = 1:n_timeslices, j = 1:n_ensemble]

        fom = 0.0
        gradient = zeros(n_ensemble, n_controls, n_timeslices)
    # end
    return (states, costates, propagators, fom, gradient)
end

"""
Initialise an ensemble from an ensemble problem definition. Right now this creates an array of problems just now, not sure if this is the best idea though.
"""
function init_ensemble(ens)
    ensemble_problem_array = [deepcopy(ens.problem) for k = 1:ens.n_ensemble]
    for k = 1:ens.n_ensemble
        prob_to_update = ensemble_problem_array[k]
        prob_to_update = @set prob_to_update.A = ens.A_generators(k)
        prob_to_update = @set prob_to_update.B = ens.B_generators(k)
        prob_to_update = @set prob_to_update.X_init = ens.X_init_generators(k)
        prob_to_update = @set prob_to_update.X_target = ens.X_target_generators(k)
        ensemble_problem_array[k] = prob_to_update
    end
    ensemble_problem_array # vector{ens.problem}
end
