####
# L2 regression for Wiens method
####

wiens_glmnet = function(xTrain, yTrain, xTest){
    library(glmnet)

    cv_model = cv.glmnet(xTrain, yTrain, alpha = 0)
    best_lambda <- cv_model$lambda.min

    best_model <- glmnet(xTrain, yTrain, alpha = 0, lambda = best_lambda)
    coef(best_model)
    # Make prediction
    p=predict(best_model, s = best_lambda, newx = xTest, , type="response")


    return(p)
}
