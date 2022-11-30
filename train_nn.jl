using Flux
using Distributions

function train_nn(m, ps, y_train, X_train, y_calib, X_calib, max_epochs = 20, tolerance = 1e-10)

    data = Flux.Data.DataLoader((X_train, y_train), batchsize=64, shuffle=true)
    # intialize weights as page 33
    # https://arxiv.org/pdf/1510.00726.pdf
    d_in = size(ps[:][1])[2]
    d_out= size(ps[:][end-1])[1]
    bound = sqrt(6)/(sqrt(d_in + d_out))

    for param in ps

        try
            w_mat = rand(Uniform(-bound, bound), size(param))
            param .= w_mat
        catch
        end
    end

    param_vec = Array{Any}(undef, max_epochs+1)
    param_vec[1] = deepcopy(ps)


    # Defining our model, optimization algorithm and loss function
    opt = ADAM()
    # opt = Descent(0.05)

    loss(x, y) = Flux.Losses.mse(m(x), y )
    # loss(x, y) = Flux.Losses.binarycrossentropy(m(x), y )

    # set these to max epoch and return length
    train_loss = [loss(X_train, y_train)]
    calib_loss = [loss(X_calib, y_calib)]

    global epoch = 1

    while true
        Flux.train!(loss, ps, data, opt)
        global epoch += 1

        append!(train_loss, loss(X_train, y_train))
        append!(calib_loss, loss(X_calib, y_calib))
        param_vec[epoch] = deepcopy(ps)

        println("epoch:\t" * string(epoch))
        println("training loss:\t" * string(train_loss[end]))
        println("calib loss:\t" * string(calib_loss[end]))

        stop = epoch >= max_epochs

        if epoch >= 3
            stop |= tolerance > abs(train_loss[end] - train_loss[end-1]) / (1 + abs(train_loss[end]))
        end

        # if epoch >= 8
        #     stop |= mean(calib_loss[end-4:end]) > mean(calib_loss[end-5:end-1]) > mean(calib_loss[end-6:end-2]) > mean(calib_loss[end-7:end-3])
        # end

        stop && break
    end

    # roll back NN parameters to those where calibration loss was minimized
    min_calib_loss_index = findmin(calib_loss)[2]
    println("minimim calibration loss achieved at epoch: $min_calib_loss_index")
    optimal_params = param_vec[min_calib_loss_index]

    for param_i in 1:length(ps)
        ps[param_i] .= optimal_params[param_i]
    end

    return(m, train_loss, calib_loss, epoch)
end
