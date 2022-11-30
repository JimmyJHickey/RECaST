function prepare_simulated_data(directory_struct,
                                seed,
                                model,
                                theta_S_path,
                                n_S,
                                n_T_train,
                                noise,
                                true_sigma_T
                                )
    seed_string = lpad(seed, 4, "0")

    Random.seed!(seed)
    theta_S = readdlm(theta_S_path)

    p = length(theta_S)

    if noise == 0
        theta_T = theta_S
    else
        theta_T = theta_S + rand(Normal(0, noise), p)
    end
    
    writedlm("$(directory_struct.theta_T_dir)theta_T$(seed_string).csv", theta_T, ",")
    writedlm("$(directory_struct.theta_diff_norm_dir)theta_diff_norm$(seed_string).csv", norm(theta_S-theta_T), ",")

    # calculate true delta and gamma
    sigma_T = norm(theta_T)
    sigma_S = norm(theta_S)
    rho = (theta_T' * theta_S / (sigma_T * sigma_S))[1]

    true_delta = rho * sigma_T / sigma_S
    true_gamma = sigma_T * sqrt(1 - rho^2) / sigma_S
    writedlm("$(directory_struct.true_delta_dir)true_delta$(seed_string).csv", true_delta, ",")
    writedlm("$(directory_struct.true_gamma_dir)true_gamma$(seed_string).csv", true_gamma, ",")


    println("create source data")
    calib_prop = 0.2
    X_S = randn(n_S, p-1)
    g_S = hcat(ones(n_S), X_S) * theta_S
    if model == "discrete"
        Y_S = [rand(Bernoulli( expit( gi) )) for gi in g_S]
    elseif model == "continuous"
        true_sigma_S = 0.5
        Y_S = g_S .+ rand( Normal( 0, true_sigma_S), n_S)
    end
    source_data = hcat(Y_S, X_S)

    # train, test
    source_split = [0.80, 0.20]

    source_data = source_data[shuffle(axes(source_data, 1)), :]
    source_train_cutoff = Int32(round(n_S * source_split[1]))

    source_train = source_data[1:source_train_cutoff, :]
    source_test = source_data[source_train_cutoff+1:end, :]

    source_train_Y = source_train[:,1]
    source_train_X = source_train[:,2:end]
    source_train_df = hcat(DataFrame(shock_response = source_train_Y), DataFrame(source_train_X, :auto))

    source_calib_cutoff = Int(round(length(source_train_Y) * calib_prop))
    source_calib = source_train[1:source_calib_cutoff, :]
    source_train_nn = source_train[source_calib_cutoff+1:end, :]

    source_train_nn_Y = source_train_nn[:,1]
    source_train_nn_X = source_train_nn[:,2:end]

    source_calib_Y = source_calib[:,1]
    source_calib_X = source_calib[:,2:end]

    source_test_Y = source_test[:,1]
    source_test_X = source_test[:,2:end]
    source_test_X_intercept = hcat(ones(size(source_test[:,2:end])[1]), source_test[:,2:end])

    println("create target data")
    n_T_test = 250
    n_T = n_T_test + n_T_train
    X_T = randn(n_T, p-1)
    g_T = hcat(ones(n_T), X_T) * theta_T

    if model == "discrete"
        Y_T = [rand(Bernoulli( expit( gi) )) for gi in g_T]
    elseif model == "continuous"
        Y_T = g_T .+ rand( Normal( 0, true_sigma_T), n_T);
    end
    target_data = hcat(Y_T, X_T)

    target_data = target_data[shuffle(axes(target_data, 1)), :]

    target_test = target_data[1:n_T_test, :]
    target_train = target_data[n_T_test+1:end, :]

    calib_cutoff = Int(round(calib_prop * n_T_train))
    target_calib = target_data[1:calib_cutoff, :]
    target_train_nn = target_train[1+calib_cutoff:end, :]

    target_train_Y = target_train[:,1]
    target_train_X = target_train[:,2:end]
    target_train_df = hcat(DataFrame(shock_response = target_train_Y), DataFrame(target_train_X, :auto))

    target_train_nn_Y = target_train_nn[:,1]
    target_train_nn_X = target_train_nn[:,2:end]
    #
    # target_calib_Y = target_calib[:,1]
    # target_calib_X = target_calib[:,2:end]

    target_test_Y = target_test[:,1]
    target_test_X = target_test[:,2:end]


    return( source_train_df,
            source_test,
            source_train_Y,
            source_train_X,
            source_test_Y,
            source_test_X,
            source_test_X_intercept,
            target_train_df,
            target_test,
            target_train_Y,
            target_train_X,
            target_test_Y,
            target_test_X,
		theta_T)
end

# seed = 1
# model = "continuous"
# post_pred_beta = 10
# post_pred_y = 10
# n_S = 1_000
# n_T_train = 60
# noise = 2.0
# true_sigma_T = 0.5
# theta_S_path = "transfer_learning/theta_S.csv"
# dir = "output/test/"
# directory_struct = Directories(dir)
# make_directories(directory_struct)
#
# source_train_Y,
#         source_train_X,
#         source_calib_Y,
#         source_calib_X,
#         source_test_Y,
#         source_test_X,
#         source_test_X_intercept,
#         target_train_Y,
#         target_train_X,
#         target_test_Y,
#         target_test_X = prepare_simulated_data(directory_struct,
#                                 seed,
#                                 model,
#                                 theta_S_path,
#                                 n_S,
#                                 n_T_train,
#                                 noise,
#                                 true_sigma_T
#                                 )
#
