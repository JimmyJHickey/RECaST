using RCall
include("expit.jl")


function wiens_method_glmnet(target_train_X,
                        target_train_Y,
                        source_train_X,
                        source_train_Y,
                        target_test_X)

    # combined source and target used to train Wiens method
    X_train_combine = vcat(target_train_X, source_train_X)
    Y_train_combine = vcat(target_train_Y, source_train_Y)

    # fits with liblinear R implementation
    R"source('wiens_method_glmnet.R')"
    R"wiens_pred = wiens_glmnet($X_train_combine, $Y_train_combine, $target_test_X)"
    wiens_pred = rcopy(R"wiens_pred")[:,1]

    return(wiens_pred)
end
