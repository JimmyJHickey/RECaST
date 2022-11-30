using Plots

function plot_roc(tpr, fpr, title="", color="blue", shape="circle", marker_size=3, diagline=false,prev_plot=nothing)

    if isnothing(prev_plot)
        fig = scatter()
    else
        fig = prev_plot
    end

    auc = calc_auc(tpr, fpr)
    fig = plot!(fpr, tpr,
            label="",
            legend = :bottomright,
            linewidth = 1,
            color = color)

    fpr = fpr[collect(1:5:length(fpr))]
    tpr = tpr[collect(1:5:length(tpr))]



    fig = scatter!(fpr, tpr,
            label="AUC "*title*"= " * string(round(auc, sigdigits = 3)),
            legend = :bottomright,
            markersize = marker_size,
            markerstrokecolor = color,
            # color=RGBA(0,0,0,0),
            color="white",
            markershape = shape)

    fig = title!("ROC Curve")
    fig = xlabel!("False Positive Rate")
    fig = ylabel!("True Positive Rate")

    if diagline

        diag_line = collect(0:1/length(tpr):1)
        fig = plot!(diag_line, diag_line, label="", linewidth = 2, linecolor="gray", alpha=0.25)
    end

    return(fig)
end

# calculate auc with trapezoid rule
function calc_auc(tpr, fpr)

    combine = hcat(tpr, fpr)

    unique_pairs = unique(combine, dims = 1)

    tpr2 = unique_pairs[:,1]
    fpr2 = unique_pairs[:,2]

    y_left = tpr2[1:end-1]
    y_right = tpr2[2:end]

    x = -diff(fpr2)

    integral = sum(1/2 * -diff(fpr2) .*  (y_left + y_right))

    return(integral)
end


######################
# calculate ROC curve
#
# truth     = vector of true labels
# prob      = vector of predicted probablilities
# n_points  = number of evenly space thresholds
#
# returns vector of true positive rates and false positive rates
#######################
function roc(truth, prob, n_points = 100)
    truth = BitArray(truth)
	len = length(truth)
    # true positive rate and false positive rate vectors
    tpr = Array{Float64}(undef, n_points)
    fpr = Array{Float64}(undef, n_points)

    # predicitions based on threshold
    pred_thresh = Array{Float64}(undef, len)
    thresholds = collect(range(-0.001, stop = 1.001, length = n_points))

    for ii in 1:length(thresholds)
        # evenly spaced thresholds from 0.0 to 1.0
        threshold = thresholds[ii]
        # all probabilities greater than threshold are 1, else 0
        pred_thresh = BitArray(prob .>= threshold)
        # thresh+1 to adress 1-based indexing
        tpr[ii] = sum(truth .& pred_thresh) / sum(truth)
        fpr[ii] = sum(.!truth .& pred_thresh) / (length(truth) - sum(truth))

    end

    return(tpr, fpr)
end



# truth = [0,0,0,0,1,1,1,1]
# p = [0.1, 0.15, 0.3, 0.7, 0.4, 0.8, 0.89, 0.7]
# p = [0.01, 0.01, 0.01, 0.01, 0.9, 1.0, 1.0, 1.0]
# #
# # #
# tpr, fpr = roc(truth, p, 10)
# calc_auc(tpr, fpr)
#
#
# plot_roc(tpr, fpr)
