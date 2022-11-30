# directory structure
mutable struct Directories
    base_dir::String
    true_delta_dir::String
    true_gamma_dir::String
    theta_T_dir::String
    theta_diff_norm_dir::String
    target_train_pd_dir::String
    delta_trace_glm_dir::String
    delta_coverage_glm_dir::String
    gamma_trace_glm_dir::String
    log_gamma_coverage_glm_dir::String
    delta_trace_nn_dir::String
    delta_coverage_nn_dir::String
    gamma_trace_nn_dir::String
    log_gamma_coverage_nn_dir::String
    log_sigma_trace_glm_dir::String
    log_sigma_coverage_glm_dir::String
    log_sigma_trace_nn_dir::String
    log_sigma_coverage_nn_dir::String
    acceptance_glm_dir::String
    tl_coverage_glm_dir::String
    tl_length_glm_dir::String
    acceptance_nn_dir::String
    tl_coverage_nn_dir::String
    tl_length_nn_dir::String
    source_size_dir::String
    source_glm_auc_dir::String
    source_nn_auc_dir::String
    target_train_nn_auc_dir::String
    target_train_size_dir::String
    target_test_size_dir::String
    analysis_dir::String
    tpr_tl_glm_dir::String
    fpr_tl_glm_dir::String
    tpr_tl_nn_dir::String
    fpr_tl_nn_dir::String
    tpr_wiens_dir::String
    fpr_wiens_dir::String
    tpr_source_glm_dir::String
    fpr_source_glm_dir::String
    tpr_source_nn_dir::String
    tpr_target_train_nn_dir::String
    fpr_source_nn_dir::String
    fpr_target_train_nn_dir::String
    tpr_glm_dir::String
    fpr_glm_dir::String
    tpr_nn_dir::String
    fpr_nn_dir::String
    tpr_unfreeze_dir::String
    fpr_unfreeze_dir::String
    unfreeze_coverage_dir::String
    unfreeze_length_dir::String
    glm_coverage_dir::String
    glm_length_dir::String
    nn_coverage_dir::String
    nn_length_dir::String
    wiens_coverage_dir::String
    source_train_loss_dir::String
    source_calib_loss_dir::String
    source_epoch_dir::String
    target_train_train_loss_dir::String
    target_train_calib_loss_dir::String
    target_train_epoch_dir::String
    target_unfreeze_train_loss_dir::String
    target_unfreeze_calib_loss_dir::String
    target_unfreeze_epoch_dir::String
    source_glm_mse_dir::String
    source_nn_mse_dir::String
    glm_tl_mse_dir::String
    glm_mse_dir::String
    nn_mse_dir::String
    tl_nn_mse_dir::String
    unfreeze_mse_dir::String
end

# build directory structure off of base directory
function Directories(base_dir::String)

    true_delta_dir = "$(base_dir)true_delta/"
    true_gamma_dir = "$(base_dir)true_gamma/"
    theta_T_dir = "$(base_dir)theta_T/"
    theta_diff_norm_dir = "$(base_dir)theta_diff_norm/"
    target_train_pd_dir = "$(base_dir)target_train_pd/"
    delta_trace_glm_dir = "$(base_dir)delta_trace_glm/"
    delta_coverage_glm_dir = "$(base_dir)delta_coverage_glm/"
    gamma_trace_glm_dir = "$(base_dir)gamma_trace_glm/"
    log_gamma_coverage_glm_dir = "$(base_dir)log_gamma_coverage_glm/"
    delta_trace_nn_dir = "$(base_dir)delta_trace_nn/"
    delta_coverage_nn_dir = "$(base_dir)delta_coverage_nn/"
    gamma_trace_nn_dir = "$(base_dir)gamma_trace_nn/"
    log_gamma_coverage_nn_dir = "$(base_dir)log_gamma_coverage_nn/"
    log_sigma_trace_glm_dir = "$(base_dir)log_sigma_trace_glm/"
    log_sigma_coverage_glm_dir = "$(base_dir)log_sigma_coverage_glm/"
    log_sigma_trace_nn_dir = "$(base_dir)log_sigma_trace_nn/"
    log_sigma_coverage_nn_dir = "$(base_dir)log_sigma_coverage_nn/"
    acceptance_glm_dir = "$(base_dir)acceptance_glm/"
    tl_coverage_glm_dir = "$(base_dir)coverage_tl_glm/"
    tl_length_glm_dir = "$(base_dir)length_tl_glm/"
    acceptance_nn_dir = "$(base_dir)acceptance_nn/"
    tl_coverage_nn_dir = "$(base_dir)coverage_tl_nn/"
    tl_length_nn_dir = "$(base_dir)length_tl_nn/"
    source_size_dir = "$(base_dir)source_size/"
    source_glm_auc_dir = "$(base_dir)source_glm_auc/"
    source_nn_auc_dir = "$(base_dir)source_nn_auc/"
    target_train_nn_auc_dir = "$(base_dir)target_train_nn_auc/"
    target_train_size_dir = "$(base_dir)target_train_size/"
    target_test_size_dir = "$(base_dir)target_test_size/"
    analysis_dir = "$(base_dir)analysis/"
    tpr_tl_glm_dir = "$(base_dir)tpr_tl_glm/"
    fpr_tl_glm_dir = "$(base_dir)fpr_tl_glm/"
    tpr_tl_nn_dir = "$(base_dir)tpr_tl_nn/"
    fpr_tl_nn_dir = "$(base_dir)fpr_tl_nn/"
    tpr_wiens_dir = "$(base_dir)tpr_wiens/"
    fpr_wiens_dir = "$(base_dir)fpr_wiens/"
    tpr_source_glm_dir = "$(base_dir)tpr_source_glm/"
    fpr_source_glm_dir = "$(base_dir)fpr_source_glm/"
    tpr_source_nn_dir = "$(base_dir)tpr_source_nn/"
    tpr_target_train_nn_dir = "$(base_dir)tpr_target_train_nn/"
    fpr_source_nn_dir = "$(base_dir)fpr_source_nn/"
    fpr_target_train_nn_dir = "$(base_dir)fpr_target_train_nn/"
    tpr_glm_dir = "$(base_dir)tpr_glm/"
    fpr_glm_dir = "$(base_dir)fpr_glm/"
    tpr_nn_dir = "$(base_dir)tpr_nn/"
    fpr_nn_dir = "$(base_dir)fpr_nn/"
    tpr_unfreeze_dir = "$(base_dir)tpr_unfreeze/"
    fpr_unfreeze_dir = "$(base_dir)fpr_unfreeze/"
    unfreeze_coverage_dir = "$(base_dir)coverage_unfreeze/"
    unfreeze_length_dir = "$(base_dir)length_unfreeze/"
    glm_coverage_dir = "$(base_dir)coverage_glm/"
    glm_length_dir = "$(base_dir)length_glm/"
    nn_coverage_dir = "$(base_dir)coverage_nn/"
    nn_length_dir = "$(base_dir)length_nn/"
    wiens_coverage_dir = "$(base_dir)coverage_wiens/"
    source_train_loss_dir = "$(base_dir)source_train_loss/"
    source_calib_loss_dir = "$(base_dir)source_calib_loss/"
    source_epoch_dir = "$(base_dir)source_epoch/"
    target_train_train_loss_dir = "$(base_dir)target_train_train_loss/"
    target_train_calib_loss_dir = "$(base_dir)target_train_calib_loss/"
    target_train_epoch_dir = "$(base_dir)target_train_epoch/"
    target_unfreeze_train_loss_dir = "$(base_dir)target_unfreeze_train_loss/"
    target_unfreeze_calib_loss_dir = "$(base_dir)target_unfreeze_calib_loss/"
    target_unfreeze_epoch_dir = "$(base_dir)target_unfreeze_epoch/"
    source_glm_mse_dir = "$(base_dir)source_glm_mse/"
    source_nn_mse_dir = "$(base_dir)source_nn_mse/"
    glm_tl_mse_dir = "$(base_dir)mse_tl_glm/"
    glm_mse_dir = "$(base_dir)mse_glm/"
    nn_mse_dir = "$(base_dir)mse_nn/"
    tl_nn_mse_dir = "$(base_dir)mse_tl_nn/"
    unfreeze_mse_dir = "$(base_dir)mse_unfreeze/"


    Directories(base_dir,
        true_delta_dir,
        true_gamma_dir,
        theta_T_dir,
        theta_diff_norm_dir,
        target_train_pd_dir,
        delta_trace_glm_dir,
        delta_coverage_glm_dir,
        gamma_trace_glm_dir,
        log_gamma_coverage_glm_dir,
        delta_trace_nn_dir,
        delta_coverage_nn_dir,
        gamma_trace_nn_dir,
        log_gamma_coverage_nn_dir,
        log_sigma_trace_glm_dir,
        log_sigma_coverage_glm_dir,
        log_sigma_trace_nn_dir,
        log_sigma_coverage_nn_dir,
        acceptance_glm_dir,
        tl_coverage_glm_dir,
        tl_length_glm_dir,
        acceptance_nn_dir,
        tl_coverage_nn_dir,
        tl_length_nn_dir,
        source_size_dir,
        source_glm_auc_dir,
        source_nn_auc_dir,
        target_train_nn_auc_dir,
        target_train_size_dir,
        target_test_size_dir,
        analysis_dir,
        tpr_tl_glm_dir,
        fpr_tl_glm_dir,
        tpr_tl_nn_dir,
        fpr_tl_nn_dir,
        tpr_wiens_dir,
        fpr_wiens_dir,
        tpr_source_glm_dir,
        fpr_source_glm_dir,
        tpr_source_nn_dir,
        tpr_target_train_nn_dir,
        fpr_source_nn_dir,
        fpr_target_train_nn_dir,
        tpr_glm_dir,
        fpr_glm_dir,
        tpr_nn_dir,
        fpr_nn_dir,
        tpr_unfreeze_dir,
        fpr_unfreeze_dir,
        unfreeze_coverage_dir,
        unfreeze_length_dir,
        glm_coverage_dir,
        glm_length_dir,
        nn_coverage_dir,
        nn_length_dir,
        wiens_coverage_dir,
        source_train_loss_dir,
        source_calib_loss_dir,
        source_epoch_dir,
        target_train_train_loss_dir,
        target_train_calib_loss_dir,
        target_train_epoch_dir,
        target_unfreeze_train_loss_dir,
        target_unfreeze_calib_loss_dir,
        target_unfreeze_epoch_dir,
        source_glm_mse_dir,
        source_nn_mse_dir,
        glm_tl_mse_dir,
        glm_mse_dir,
        nn_mse_dir,
        tl_nn_mse_dir,
        unfreeze_mse_dir)

end

# make a directory if it does not exist
function make_not_empty_dir(dir)
    if !ispath(dir)
        mkdir(dir)
    end
end

# make all directories
function make_directories(dir_struct::Directories)
    make_not_empty_dir(dir_struct.base_dir)
    make_not_empty_dir(dir_struct.true_delta_dir)
    make_not_empty_dir(dir_struct.true_gamma_dir)
    make_not_empty_dir(dir_struct.theta_T_dir)
    make_not_empty_dir(dir_struct.theta_diff_norm_dir)
    make_not_empty_dir(dir_struct.target_train_pd_dir)
    make_not_empty_dir(dir_struct.delta_trace_glm_dir)
    make_not_empty_dir(dir_struct.delta_coverage_glm_dir)
    make_not_empty_dir(dir_struct.gamma_trace_glm_dir)
    make_not_empty_dir(dir_struct.log_gamma_coverage_glm_dir)
    make_not_empty_dir(dir_struct.delta_trace_nn_dir)
    make_not_empty_dir(dir_struct.delta_coverage_nn_dir)
    make_not_empty_dir(dir_struct.gamma_trace_nn_dir)
    make_not_empty_dir(dir_struct.log_gamma_coverage_nn_dir)
    make_not_empty_dir(dir_struct.log_sigma_trace_glm_dir)
    make_not_empty_dir(dir_struct.log_sigma_coverage_glm_dir)
    make_not_empty_dir(dir_struct.log_sigma_trace_nn_dir)
    make_not_empty_dir(dir_struct.log_sigma_coverage_nn_dir)
    make_not_empty_dir(dir_struct.acceptance_glm_dir)
    make_not_empty_dir(dir_struct.tl_coverage_glm_dir)
    make_not_empty_dir(dir_struct.tl_length_glm_dir)
    make_not_empty_dir(dir_struct.acceptance_nn_dir)
    make_not_empty_dir(dir_struct.tl_coverage_nn_dir)
    make_not_empty_dir(dir_struct.tl_length_nn_dir)
    make_not_empty_dir(dir_struct.source_size_dir)
    make_not_empty_dir(dir_struct.source_glm_auc_dir)
    make_not_empty_dir(dir_struct.source_nn_auc_dir)
    make_not_empty_dir(dir_struct.target_train_nn_auc_dir)
    make_not_empty_dir(dir_struct.target_train_size_dir)
    make_not_empty_dir(dir_struct.target_test_size_dir)
    make_not_empty_dir(dir_struct.analysis_dir)
    make_not_empty_dir(dir_struct.tpr_tl_glm_dir)
    make_not_empty_dir(dir_struct.fpr_tl_glm_dir)
    make_not_empty_dir(dir_struct.tpr_tl_nn_dir)
    make_not_empty_dir(dir_struct.fpr_tl_nn_dir)
    make_not_empty_dir(dir_struct.tpr_wiens_dir)
    make_not_empty_dir(dir_struct.fpr_wiens_dir)
    make_not_empty_dir(dir_struct.tpr_source_glm_dir)
    make_not_empty_dir(dir_struct.fpr_source_glm_dir)
    make_not_empty_dir(dir_struct.tpr_source_nn_dir)
    make_not_empty_dir(dir_struct.tpr_target_train_nn_dir)
    make_not_empty_dir(dir_struct.fpr_source_nn_dir)
    make_not_empty_dir(dir_struct.fpr_target_train_nn_dir)
    make_not_empty_dir(dir_struct.tpr_glm_dir)
    make_not_empty_dir(dir_struct.fpr_glm_dir)
    make_not_empty_dir(dir_struct.tpr_nn_dir)
    make_not_empty_dir(dir_struct.fpr_nn_dir)
    make_not_empty_dir(dir_struct.glm_coverage_dir)
    make_not_empty_dir(dir_struct.glm_length_dir)
    make_not_empty_dir(dir_struct.tpr_unfreeze_dir)
    make_not_empty_dir(dir_struct.fpr_unfreeze_dir)
    make_not_empty_dir(dir_struct.unfreeze_coverage_dir)
    make_not_empty_dir(dir_struct.unfreeze_length_dir)
    make_not_empty_dir(dir_struct.nn_coverage_dir)
    make_not_empty_dir(dir_struct.wiens_coverage_dir)
    make_not_empty_dir(dir_struct.nn_length_dir)
    make_not_empty_dir(dir_struct.source_train_loss_dir)
    make_not_empty_dir(dir_struct.source_calib_loss_dir)
    make_not_empty_dir(dir_struct.source_epoch_dir)
    make_not_empty_dir(dir_struct.target_train_train_loss_dir)
    make_not_empty_dir(dir_struct.target_train_calib_loss_dir)
    make_not_empty_dir(dir_struct.target_train_epoch_dir)
    make_not_empty_dir(dir_struct.target_unfreeze_train_loss_dir)
    make_not_empty_dir(dir_struct.target_unfreeze_calib_loss_dir)
    make_not_empty_dir(dir_struct.target_unfreeze_epoch_dir)
    make_not_empty_dir(dir_struct.source_glm_mse_dir)
    make_not_empty_dir(dir_struct.source_nn_mse_dir)
    make_not_empty_dir(dir_struct.glm_tl_mse_dir)
    make_not_empty_dir(dir_struct.glm_mse_dir)
    make_not_empty_dir(dir_struct.nn_mse_dir)
    make_not_empty_dir(dir_struct.tl_nn_mse_dir)
    make_not_empty_dir(dir_struct.unfreeze_mse_dir)

end
