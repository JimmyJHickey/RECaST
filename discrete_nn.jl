using Flux

function make_nn(p)
    return(
        Chain(
            Dense(p, 25),
            x->relu.(x),
            Dense(25,1),
            x->sigmoid.(x)
		)
    )
end
