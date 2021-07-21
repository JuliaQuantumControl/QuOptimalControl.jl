using Setfield

"""
Initialise all the storage arrays that will be used in a GRAPE optimisation
"""
function init_GRAPE(X, n_timeslices, n_ensemble, A, n_controls)
    # states
    states = [similar(X) .* 0.0 for i = 1:n_timeslices+1, j = 1:n_ensemble]
    # costates
    costates = [similar(X) .* 0.0 for i = 1:n_timeslices+1, j = 1:n_ensemble]
    # exp(gens)
    propagators = [similar(A) .* 0.0 for i = 1:n_timeslices, j = 1:n_ensemble]

    fom = 0.0
    gradient = zeros(n_ensemble, n_controls, n_timeslices)
    # end
    return (states, costates, propagators, fom, gradient)
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
