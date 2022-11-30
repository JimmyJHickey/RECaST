function expit(x)
    return( 1 / (1 + exp(-x)))
end

function logit(x)
	return log(x / (1-x))
end
