#!/bin/bash

mkdir -p output
mkdir -p output/discrete
mkdir -p output/continuous

declare -a model_array=(discrete continuous)
declare -a n_T_train_array=(20 40 60 100 250)
declare -a noise_array=(0.0 0.25 1.0 2.0)

for model in "${model_array[@]}"
do

	for n_T_train in "${n_T_train_array[@]}"
	do

		for noise in "${noise_array[@]}"
		do


		out_dir="output/${model}/${n_T_train}_${noise}/"
		echo ${out_dir}

			for seed in {1..300}
			do

			julia run_file.jl \
				--out_dir ${out_dir} \
				--seed ${seed} \
				--theta_S_path theta_S.csv \
				--model ${model} \
				--steps 100000 \
				--burnin 20000 \
				--start_trace 30000 \
				--thinning 300 \
				--post_pred_beta 300 \
				--post_pred_y 300 \
				--n_S 1000 \
				--n_T_train ${n_T_train} \
				--noise ${noise} \
				--true_sigma_T 0.5

			done # seed
		done # noise
	done # n_T_train
done # model
