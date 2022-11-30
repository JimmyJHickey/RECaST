using DataFrames
using GLM
include("expit.jl")

function predict_glm(glm, test_X, model = "")
        prediction = test_X * coef(glm)
        # replace with Y_pred = X * coef(glm)
        if model == "discrete"
            prediction = expit.(prediction)
        end
        return(prediction)
end

function fit_glm_discrete(train)
    out_glm = glm(Term(:shock_response) ~ sum(Term.(Symbol.(names(train[:, Not(:shock_response)])))), train, Binomial(), LogitLink())
    return(out_glm)
end

function fit_glm_continuous(train)
    out_glm = lm(Term(:shock_response) ~ sum(Term.(Symbol.(names(train[:, Not(:shock_response)])))), train)
    return(out_glm)
end

function continuous_prediction_interval(train_X, train_Y, test_X, test_Y, glm_model, level)
    n_train, p = size(train_X)
    X_train_intercept = hcat(ones(n_train), train_X)

    level += (1-level)/2

    b_hat = coef(glm_model)
    X_train_intercept = hcat(ones(n_train), train_X)
    t_alpha2 = quantile(TDist(n_train-p), level)

    sigma_hat = sqrt(1/(n_train - p) * sum((train_Y - X_train_intercept * b_hat).^2))

    n_test = length(test_Y)
    length_vec = Array{Float64}(undef, n_test)
    cov_vec = Array{Float64}(undef, n_test)

    for ii in 1:size(test_X)[1]
        # x_star = vcat(1, test_X[ii,:])
        x_star = test_X[ii,:]
        lower_est = x_star' * b_hat - t_alpha2 * sigma_hat * ( x_star' * (X_train_intercept' * X_train_intercept)^(-1) * x_star + 1 )^(1/2)
        upper_est = x_star' * b_hat + t_alpha2 * sigma_hat * ( x_star' * (X_train_intercept' * X_train_intercept)^(-1) * x_star + 1 )^(1/2)

        length_vec[ii] = upper_est - lower_est
        cov_vec[ii] = lower_est <= test_Y[ii] <= upper_est
    end

    return(length_vec, cov_vec)
end
