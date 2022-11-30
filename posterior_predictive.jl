using Distributions
using Statistics
using StatsBase
using DelimitedFiles
include("glm_regression.jl")
include("expit.jl")
include("roc.jl")
include("mse.jl")

include("colmeans_missing.jl")

function posterior_predictive(model,
                                directory_structure,
                                seed_string,
                                trace_mat,
                                f_X_theta_S,
                                num_beta,
                                num_y,
                                Yt,
                                source_model_type)


    if source_model_type == "glm"
        tpr_dir = directory_struct.tpr_tl_glm_dir
        fpr_dir = directory_struct.fpr_tl_glm_dir
        cov_dir = directory_struct.tl_coverage_glm_dir
        len_dir = directory_struct.tl_length_glm_dir
        mse_dir = directory_struct.glm_tl_mse_dir
    elseif source_model_type == "nn"
        tpr_dir = directory_struct.tpr_tl_nn_dir
        fpr_dir = directory_struct.fpr_tl_nn_dir
        cov_dir = directory_struct.tl_coverage_nn_dir
        len_dir = directory_struct.tl_length_nn_dir
        mse_dir = directory_struct.tl_nn_mse_dir
    end

    sample_size = length(Yt)

    delta_trace = trace_mat[:,1]
    delta_mean = mean(delta_trace)
    gamma_trace = exp.(trace_mat[:,2])
    gamma_mean = mean(gamma_trace)

    trace_len = length(delta_trace)

    if model=="continuous"
        coverage_levels = collect(0.5:0.05:0.95)
        n_cov_levels = length(coverage_levels)
        Y_cov = zeros(sample_size, n_cov_levels)
        cov_len = zeros(sample_size, n_cov_levels)


        Y_post_pred_means = zeros(sample_size)

        sigma_trace = exp.(trace_mat[:,3])
        sigma_mean = mean(sigma_trace)
    end

    p_T = zeros(sample_size)

    for i in 1:sample_size
        println("Yt: $(i)")

        # posterior predicitve sampling

        if model=="continuous"
            Y_post_pred = [rand(Normal(f_X_theta_S[i] * beta, sigma_trace[j]))
                                    for j in 1:trace_len
                                    for beta in rand(Cauchy(delta_trace[j], gamma_trace[j]), num_beta)
                                    for _ in 1:num_y]

            Y_post_pred_means[i] = median(Y_post_pred)
            lower_vec = [percentile(Y_post_pred, 100 * (0.50 - (level/2))) for level in coverage_levels]
            upper_vec = [percentile(Y_post_pred, 100 * (0.50 + (level/2))) for level in coverage_levels]

            Y_cov[i,:], cov_len[i,:] = calculate_coverage_continuous(Yt[i], lower_vec, upper_vec)

        elseif model=="discrete"

            p_T[i] = mean([rand(Bernoulli(expit(f_X_theta_S[i] * beta)))
                                    for j in 1:trace_len
                                    for beta in rand(Cauchy(delta_trace[j], gamma_trace[j]), num_beta)
                                    for _ in 1:num_y])


        end # if model
    end # end iteration over Yt

    # make ROC curve
    if model == "discrete"

        tpr_tl, fpr_tl = roc(Yt, p_T)

        writedlm("$(tpr_dir)tpr_tl$(seed_string).csv", tpr_tl, ',')
        writedlm("$(fpr_dir)fpr_tl$(seed_string).csv", fpr_tl, ',')

    elseif model == "continuous"

        mse = calculate_mse(Yt, Y_post_pred_means)
        writedlm("$(mse_dir)tl_mse$(seed_string).csv", mse, ',')


        writedlm("$(cov_dir)post_pred_coverage$(seed_string).csv", mean(Y_cov, dims=1)', ',')
        writedlm("$(len_dir)post_pred_length$(seed_string).csv", mean(cov_len, dims=1)', ',')

    end

end


function calculate_coverage_continuous(Yt, lower_vec, upper_vec)

    coverage_levels = collect(0.5:0.05:0.95)
    n_cov_levels = length(coverage_levels)

    Y_cov = zeros(n_cov_levels)
    cov_length = zeros(n_cov_levels)

    for i in 1:n_cov_levels

        lower = lower_vec[i]
        upper = upper_vec[i]

        Y_cov[i] = (lower <= Yt <= upper)
        cov_length[i] = upper - lower
    end

    return(Y_cov, cov_length)
end
