function calculate_mse(Yt, Y_post_pred)
    return( 1/length(Yt) * sum( (Yt .-Y_post_pred ).^2 ) )
end
