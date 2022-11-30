    using CSV
    using DataFrames
    using Query
    using Random
    using StatsBase
    using ArgParse
    using DelimitedFiles
    using TickTock
    using Statistics
    using LinearAlgebra
    using Flux

    include("glm_regression.jl")
    include("roc.jl")
    include("mse.jl")
    include("make_directories.jl")


argtable = ArgParseSettings()

@add_arg_table argtable begin
"--out_dir"
    help = "output directory"
"--seed"
    help = "add seed"
    arg_type = Int
    default = 1978
"--steps"
    help = "number of MCMC steps"
    arg_type = Int
    default = 100_000
"--burnin"
    arg_type = Int
    default = 20_000
"--start_trace"
    help = "where to start using trace values (after burn in)"
    arg_type = Int
    default = 30_000
"--thinning"
    help = "thin MCMC chains to this many points"
    arg_type = Int
    default = 100
"--post_pred_beta"
    help = "number of betas per delta, gamma pair in trace"
    arg_type = Int
    default = 100
"--post_pred_y"
    help = "number of ys per beta in poster predictive"
    arg_type = Int
    default = 100
# simualted data specific inputs
"--theta_S_path"
    help = "output directory"
"--model"
    help = "continuous or discrete"
"--n_S"
    help = "number of source data points"
    arg_type = Int
    default = 1_000
"--n_T_train"
    help = "number of target training data points"
    arg_type = Int
    default = 100
"--noise"
    help = "standard deviation of noise to be added to theta_S to generate the real theta_T"
    arg_type = Float64
    default = 0.5
"--true_sigma_T"
    help = "true sigma for continuous case"
    arg_type = Float64
    default = 0.5
end

parsed_args = parse_args(ARGS, argtable)
seed = parsed_args["seed"]
seed_string = lpad(seed, 4, "0")
dir = parsed_args["out_dir"]
steps = parsed_args["steps"]
burnin = parsed_args["burnin"]
start_trace = parsed_args["start_trace"]
thinning = parsed_args["thinning"]
post_pred_beta = parsed_args["post_pred_beta"]
post_pred_y = parsed_args["post_pred_y"]
# simulated data specific inputs
theta_S_path = parsed_args["theta_S_path"]
model = parsed_args["model"]
n_S = parsed_args["n_S"]
n_T_train = parsed_args["n_T_train"]
noise = parsed_args["noise"]
true_sigma_T = parsed_args["true_sigma_T"]


directory_struct = Directories(dir)
make_directories(directory_struct)



include("prepare_simulated_data.jl")
source_train,
        source_test,
        source_train_Y,
        source_train_X,
        source_test_Y,
        source_test_X,
        source_test_X_intercept,
        target_train,
        target_test,
        target_train_Y,
        target_train_X,
        target_test_Y,
        target_test_X,
        theta_T = prepare_simulated_data(directory_struct,
                                seed,
                                model,
                                theta_S_path,
                                n_S,
                                n_T_train,
                                noise,
                                true_sigma_T
                                )

Random.seed!(seed)

####################
# run TL model
####################
source_glm_model = nothing

# source glm
if model == "discrete"
    global source_glm_model = fit_glm_discrete(source_train)
elseif model == "continuous"
    global source_glm_model = fit_glm_continuous(source_train)
end

theta_hat_S = coef(source_glm_model)

if model == "discrete"
    source_glm_pred = predict_glm(source_glm_model, source_test_X_intercept, model)
    tpr_glm_source, fpr_glm_source = roc(source_test_Y, source_glm_pred)
    auc_glm_source = calc_auc(tpr_glm_source, fpr_glm_source)

    writedlm("$(directory_struct.tpr_source_glm_dir)tpr_source$(seed_string).csv", tpr_glm_source, ",")
    writedlm("$(directory_struct.fpr_source_glm_dir)fpr_source$(seed_string).csv", fpr_glm_source, ",")
    writedlm("$(directory_struct.source_glm_auc_dir)source_auc$(seed_string).csv", auc_glm_source, ',')
elseif model == "continuous"
    source_glm_pred = predict_glm(source_glm_model, source_test_X_intercept, model)
    source_glm_mse = calculate_mse(source_test_Y, source_glm_pred)
    writedlm("$(directory_struct.source_glm_mse_dir)source_glm_mse$(seed_string).csv", source_glm_mse, ",")
end

target_train_X_intercept = hcat(ones(size(target_train_X)[1]), target_train_X)

f_glm = target_train_X_intercept * theta_hat_S

init_param = [0.0, 1.0]

# sigma^2 parameter initialization
if model == "continuous"
    append!(init_param, 0.1)
end


include("TL_routine.jl")
output_glm = mcmc_routine( target_train_Y, f_glm, model, init_param, steps, burnin)
chain_glm = output_glm[1]
delta_trace_glm = chain_glm[:,1]
log_gamma_trace_glm = chain_glm[:,2]
acceptance_ratio_glm = output_glm[2]

writedlm("$(directory_struct.acceptance_glm_dir)acceptance_ratio$(seed_string).csv", acceptance_ratio_glm, ",")
writedlm("$(directory_struct.delta_trace_glm_dir)delta_trace$(seed_string).csv", delta_trace_glm, ",")
writedlm("$(directory_struct.gamma_trace_glm_dir)log_gamma_trace$(seed_string).csv", log_gamma_trace_glm, ",")

if model == "continuous"
	log_sigma_trace_glm = chain_glm[:,3]
	sigma_cov = Int32(percentile(log_sigma_trace_glm , 2.5) <= log(true_sigma_T)  <= percentile(log_sigma_trace_glm, 97.5))
	writedlm("$(directory_struct.log_sigma_trace_glm_dir)log_sigma_trace$(seed_string).csv", log_sigma_trace_glm, ",")
	writedlm("$(directory_struct.log_sigma_coverage_glm_dir)log_sigma_coverage$(seed_string).csv", sigma_cov, ",")
end


# source NN
include("$(model)_nn.jl")

include("train_nn.jl")
p = size(target_train_X)[2]

source_nn = make_nn(p)
source_ps = Flux.params(source_nn)

train_split = [0.9, 0.1]

tt_cutoff = Int32(round(size(source_train_X)[1] * train_split[1]))

source_train_train_X = source_train_X[1:tt_cutoff, :]
source_train_train_Y = source_train_Y[1:tt_cutoff, :]

source_train_calib_X = source_train_X[tt_cutoff+1:end, :]
source_train_calib_Y = source_train_Y[tt_cutoff+1:end, :]

source_nn,
    train_loss,
    calib_loss,
    source_epoch = train_nn(source_nn,
                                source_ps,
                                source_train_train_Y',
                                source_train_train_X',
                                source_train_calib_Y',
                                source_train_calib_X',
                                2500,
                                1e-10)

source_nn_pred = source_nn(source_test_X')

if model == "discrete"
    tpr_nn_source, fpr_nn_source = roc(source_test_Y, source_nn_pred')
    auc_nn_source = calc_auc(tpr_nn_source, fpr_nn_source)

    writedlm("$(directory_struct.tpr_source_nn_dir)tpr_source$(seed_string).csv", tpr_nn_source, ",")
    writedlm("$(directory_struct.fpr_source_nn_dir)fpr_source$(seed_string).csv", fpr_nn_source, ",")
    writedlm("$(directory_struct.source_nn_auc_dir)source_auc$(seed_string).csv", auc_nn_source, ',')

elseif model == "continuous"
    source_nn_mse = calculate_mse(source_test_Y, source_nn_pred')
    writedlm("$(directory_struct.source_nn_mse_dir)source_nn_mse$(seed_string).csv", source_nn_mse, ',')
end


writedlm("$(directory_struct.source_train_loss_dir)source_train_loss$(seed_string).csv", train_loss, ",")
writedlm("$(directory_struct.source_calib_loss_dir)source_calib_loss$(seed_string).csv", calib_loss, ",")
writedlm("$(directory_struct.source_epoch_dir)source_epoch$(seed_string).csv", calib_loss, ",")


f_nn = source_nn(target_train_X')[1,:]

if model == "discrete"
	f_nn = logit.(f_nn)
end

output_nn = mcmc_routine(target_train_Y, f_nn, model, init_param, steps, burnin)
chain_nn = output_nn[1]
delta_trace_nn = chain_nn[:,1]
log_gamma_trace_nn = chain_nn[:,2]
acceptance_ratio_nn = output_nn[2]

writedlm("$(directory_struct.acceptance_nn_dir)acceptance_ratio$(seed_string).csv", acceptance_ratio_nn, ",")
writedlm("$(directory_struct.delta_trace_nn_dir)delta_trace$(seed_string).csv", delta_trace_nn, ",")
writedlm("$(directory_struct.gamma_trace_nn_dir)log_gamma_trace$(seed_string).csv", log_gamma_trace_nn, ",")

if model == "continuous"
	log_sigma_trace_nn = chain_nn[:,3]
	sigma_cov = Int32(percentile(log_sigma_trace_nn , 2.5) <= log(true_sigma_T)  <= percentile(log_sigma_trace_nn, 97.5))
	writedlm("$(directory_struct.log_sigma_trace_nn_dir)log_sigma_trace$(seed_string).csv", log_sigma_trace_nn, ",")
	writedlm("$(directory_struct.log_sigma_coverage_nn_dir)log_sigma_coverage$(seed_string).csv", sigma_cov, ",")
end


if model == "discrete"
	include("run_wiens_glmnet.jl")

	wiens_pred_func(x) = wiens_method_glmnet(target_train_X,
				target_train_Y,
				source_train_X,
				source_train_Y,
				x)
end




#############
# run target train nn
#############

tt_cutoff = Int32(round(size(target_train_X)[1] * train_split[1]))

target_train_train_X = target_train_X[1:tt_cutoff, :]
target_train_train_Y = target_train_Y[1:tt_cutoff, :]

target_train_calib_X = target_train_X[tt_cutoff+1:end, :]
target_train_calib_Y = target_train_Y[tt_cutoff+1:end, :]

target_train_nn = make_nn(p)


target_train_ps = Flux.params(target_train_nn)


target_train_nn,
	train_loss,
	calib_loss,
	tt_epoch = train_nn(target_train_nn,
						target_train_ps,
						target_train_train_Y',
						target_train_train_X',
						target_train_calib_Y',
						target_train_calib_X',
						2500,
						1e-10)

writedlm("$(directory_struct.target_train_train_loss_dir)target_train_train_loss$(seed_string).csv", train_loss, ",")
writedlm("$(directory_struct.target_train_calib_loss_dir)target_train_calib_loss$(seed_string).csv", calib_loss, ",")
writedlm("$(directory_struct.target_train_epoch_dir)target_train_epoch$(seed_string).csv", calib_loss, ",")


######################
# Freeze/unfreeze TL
######################

if model == "discrete"
    target_unfreeze_nn = Chain(
            source_nn[1:end-2],
    		Dense(25,1),
    		x->sigmoid.(x)
    		)
elseif model == "continuous"
    target_unfreeze_nn = Chain(
            source_nn[1:end-2],
            Dense(25,1)
            )
end

target_unfreeze_ps = Flux.params(target_unfreeze_nn[2:end])

target_unfreeze_nn,
	train_loss,
	calib_loss,
	target_unfreeze_epoch = train_nn(target_unfreeze_nn,
								target_unfreeze_ps,
								target_train_train_Y',
								target_train_train_X',
								target_train_calib_Y',
								target_train_calib_X',
								2500,
								1e-10)

writedlm("$(directory_struct.target_unfreeze_train_loss_dir)target_unfreeze_train_loss$(seed_string).csv", train_loss, ",")
writedlm("$(directory_struct.target_unfreeze_calib_loss_dir)target_unfreeze_calib_loss$(seed_string).csv", calib_loss, ",")
writedlm("$(directory_struct.target_unfreeze_epoch_dir)target_unfreeze_epoch$(seed_string).csv", calib_loss, ",")


nn_pred = target_train_nn(target_test_X')[1,:]
unfreeze_pred = target_unfreeze_nn(target_test_X')[1,:]

if model == "discrete"

	tpr_unfreeze, fpr_unfreeze = roc(target_test_Y, unfreeze_pred)
	writedlm("$(directory_struct.tpr_unfreeze_dir)tpr_unfreeze$(seed_string).csv", tpr_unfreeze, ",")
	writedlm("$(directory_struct.fpr_unfreeze_dir)fpr_unfreeze$(seed_string).csv", fpr_unfreeze, ",")

	tpr_nn, fpr_nn = roc(target_test_Y, nn_pred)
	writedlm("$(directory_struct.tpr_nn_dir)tpr_nn$(seed_string).csv", tpr_nn, ",")
	writedlm("$(directory_struct.fpr_nn_dir)fpr_nn$(seed_string).csv", fpr_nn, ",")

elseif model == "continuous"

	nn_mse = calculate_mse(target_test_Y, nn_pred)
	writedlm("$(directory_struct.nn_mse_dir)mse_nn$(seed_string).csv", nn_mse)

	unfreeze_mse = calculate_mse(target_test_Y, unfreeze_pred)
	writedlm("$(directory_struct.unfreeze_mse_dir)mse_unfreeze$(seed_string).csv", unfreeze_mse)

end


if thinning != 0
    thin_indices = Int.(floor.(collect(range(start_trace, length(delta_trace_nn), length = thinning))))
	chain_nn = chain_nn[thin_indices, :]
    chain_glm = chain_glm[thin_indices, :]
end


if model == "discrete"
    include("recast_binary_coverage.jl")

    recast_binary_coverage(theta_hat_S,
                                source_nn,
                                wiens_pred_func,
                                target_train_nn,
                                target_unfreeze_nn,
                                directory_struct,
                                seed_string,
                                chain_glm,
                                chain_nn,
                                post_pred_beta,
                                post_pred_y,
                                theta_T)
end


include("posterior_predictive.jl")

target_test_X_intercept = hcat(ones(size(target_test_X)[1]), target_test_X)
f_X_theta_S = target_test_X_intercept * theta_hat_S

posterior_predictive(model,
                    directory_struct,
                    seed_string,
                    chain_glm,
                    f_X_theta_S,
                    post_pred_beta,
                    post_pred_y,
                    target_test_Y,
                    "glm"
                    )


f_nn_test = source_nn(target_test_X')[1,:]

if model == "discrete"
	f_nn_test = logit.(f_nn_test)
end

posterior_predictive(model,
                    directory_struct,
                    seed_string,
                    chain_nn,
                    f_nn_test,
                    post_pred_beta,
		    post_pred_y,
                    target_test_Y,
                    "nn"
		   )



# Wiens prediction
if model == "discrete"
   include("run_wiens_glmnet.jl")
   wiens_pred = wiens_method_glmnet(target_train_X,
                           target_train_Y,
                           source_train_X,
                           source_train_Y,
                           target_test_X)

   tpr_wiens, fpr_wiens = roc(target_test_Y, wiens_pred)

   writedlm("$(directory_struct.tpr_wiens_dir)tpr_wiens$(seed_string).csv", tpr_wiens, ",")
   writedlm("$(directory_struct.fpr_wiens_dir)fpr_wiens$(seed_string).csv", fpr_wiens, ",")
   wiens_auc = calc_auc(tpr_wiens, fpr_wiens)

end
