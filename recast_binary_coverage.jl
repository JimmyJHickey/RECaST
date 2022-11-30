using Distributions
using Statistics
using StatsBase
using DelimitedFiles
include("glm_regression.jl")
include("expit.jl")
include("roc.jl")
include("mse.jl")

include("colmeans_missing.jl")


function recast_binary_coverage(source_glm,
                            source_nn,
                            wiens_pred_func,
                            dnn,
                            unfreeze_nn,
                            directory_structure,
                            seed_string,
                            trace_mat_glm,
                            trace_mat_nn,
                            num_beta,
                            num_y,
                            true_theta_T)



    delta_trace_glm = trace_mat_glm[:,1]
    gamma_trace_glm = exp.(trace_mat_glm[:,2])

    delta_trace_nn = trace_mat_nn[:,1]
    gamma_trace_nn = exp.(trace_mat_nn[:,2])

    trace_len = length(delta_trace_glm)

	p = length(true_theta_T)-1

    level = 0.95

    n_cov = 0
    counter = 0
    cov_vec_recast_glm = []
    cov_vec_recast_nn = []
    cov_vec_dnn = []
    cov_vec_unfreeze = []
    cov_vec_wiens = []

    X_mat = repeat([-1.0], p)'
    Y_vec = [-1]

	X_mat = Array{Float64}(undef, (1,p))

    while n_cov < 500 && counter < 10_000
        println("counter:\t$(counter)")
        println("n_cov:\t$(n_cov)")
        cov_i = false
        counter += 1
        X_T = randn(length(true_theta_T)-1)
        g_T = (vcat(1, X_T)' * true_theta_T)[1]
        Y_T = rand(Bernoulli( expit( g_T ) ))

        # if glm else nn
        f_X_theta_S_glm = (vcat(1, X_T)' * source_glm)[1]

        post_pred_sample_glm = [rand(Bernoulli(expit(f_X_theta_S_glm * beta)))
                            for j in 1:trace_len
                            for beta in rand(Cauchy(delta_trace_glm[j], gamma_trace_glm[j]), num_beta)
                            for _ in 1:num_y]

        p_recast_glm = mean(post_pred_sample_glm)

        println(p_recast_glm)

        cov_bool, pred_class = check_coverage(p_recast_glm, level)
        if cov_bool
            append!(cov_vec_recast_glm, mean(post_pred_sample_glm .== pred_class))
            cov_i = true
        end


        f_X_theta_S_nn = logit.(source_nn(X_T)[1])
        post_pred_sample_nn = [rand(Bernoulli(expit(f_X_theta_S_nn * beta)))
                            for j in 1:trace_len
                            for beta in rand(Cauchy(delta_trace_nn[j], gamma_trace_nn[j]), num_beta)
                            for _ in 1:num_y]

        p_recast_nn = mean(post_pred_sample_nn)

        cov_bool, pred_class = check_coverage(p_recast_nn, level)
        if cov_bool
            append!(cov_vec_recast_nn, mean(post_pred_sample_nn .== pred_class))
        #    cov_i = true
        end


        if cov_i
            X_mat = vcat(X_mat, X_T')

            Y_vec=hcat(Y_vec, Y_T)
        end


        n_cov += cov_i

    end # while


	if length(Y_vec) != 1
		Y_vec = Y_vec[2:end]
		X_mat = X_mat[2:end,:]

		println("before wiens")
	    p_T = wiens_pred_func(X_mat)
		println(p_T)

	    cov_vec_wiens = is_covered.(p_T, Y_vec, level)

		println("before dnn")

	    p_T = dnn(X_mat')[1,:]

		println(p_T)
	    cov_vec_dnn = is_covered.(p_T, Y_vec, level)

		println("before unfreeze")

	    p_T = unfreeze_nn(X_mat')[1,:]
	    cov_vec_unfreeze = is_covered.(p_T, Y_vec, level)
	end



    coverage = length(cov_vec_recast_glm) == 0 ? -1 : mean(cov_vec_recast_glm)
    writedlm("$(directory_struct.tl_coverage_glm_dir)post_pred_coverage$(seed_string).csv", coverage, ',')

    coverage = length(cov_vec_recast_nn) == 0 ? -1 : mean(cov_vec_recast_nn)
    writedlm("$(directory_struct.tl_coverage_nn_dir)post_pred_coverage$(seed_string).csv", coverage, ',')

    coverage = length(cov_vec_dnn) == 0 ? -1 : mean(cov_vec_dnn[cov_vec_dnn .!= -1])
    writedlm("$(directory_struct.nn_coverage_dir)post_pred_coverage$(seed_string).csv", coverage, ',')

    coverage = length(cov_vec_unfreeze) == 0 ? -1 : mean(cov_vec_unfreeze[cov_vec_unfreeze .!= -1])
    writedlm("$(directory_struct.unfreeze_coverage_dir)post_pred_coverage$(seed_string).csv", coverage, ',')

    coverage = length(cov_vec_wiens) == 0 ? -1 : mean(cov_vec_wiens[cov_vec_wiens .!= -1])
    writedlm("$(directory_struct.wiens_coverage_dir)post_pred_coverage$(seed_string).csv", coverage, ',')


end


function check_coverage(p_T, level)

    cov = false
    pred_class = 0

    if level - 0.025 <= p_T <= level + 0.025 ||
        level - 0.025 <= (1-p_T) <= level + 0.025

        cov = true

        pred_class = dom_class = p_T <= 0.50 ? 0 : 1


    end # if covered

    return(cov, pred_class)

end

function is_covered(p_T, Y_T, level)

	out = -1
	if level - 0.025 <= p_T <= level + 0.025 ||
		level - 0.025 <= (1-p_T) <= level + 0.025

		out = round(p_T) == Y_T

	end

    return(out)
end
