mel_rtable <- melt(est_rtable, id.vars = "userId", variable.name = "movieId", 
                   value.name = "resid" , variable.factor = FALSE)
mel_rtable<- mel_rtable[, movieId := as.double(movieId)]

m_bias <- sample_train[u_bias, on = .(userId)][
  , .(sum(rating - g_mean - V1)/(1.9 + .N)), by = .(movieId)]

est <- sample_test[u_bias, on = .(userId)][
  m_bias, on = .(movieId)][mel_rtable, on = .(userId, movieId)][
    !is.na(rating)][
    , .(V1 + i.V1 + resid + g_mean), by = .(userId, movieId)]

err <- sample_test[est, on = .(userId, movieId)][
  , .(rating - V1), by = .(userId, movieId)]

rmse_test <- sqrt(mean(err$V1^2))