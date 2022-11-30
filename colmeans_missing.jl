function colmeans_missing(mat)
    out = Array{Float64}(undef, size(mat)[2])

    for i in 1:length(out)
        if all(ismissing.(mat[:,i]))
            out[i] = -1
        else
            out[i] = mean(skipmissing(mat[:,i]))
        end
    end

    return(out)
end

# test = [1 missing 2;
#         missing 4 1]
#
# colmeans_missing(test)
