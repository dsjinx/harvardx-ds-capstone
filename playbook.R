library(tidyverse)
library(data.table)
library(caret)
library(Matrix)
library(doParallel)
library(glmnet)
library(RcppArmadillo)

#load the edx data from misc folder
#create a sample out of edx, and make a pair of train/test sets  

load("~/Work/Rproj/PH125.9x/harvardx-ds-capstone/misc/edx.Rdata")
edx %>% tibble()
set.seed(9 ,sample.kind = "Rounding")
sample <- sample(1:9000000, 9999, replace = FALSE)
edx_sample <- edx[sample,]

set.seed(2 ,sample.kind = "Rounding")
ind <- createDataPartition(edx_sample$rating, 1, p = 0.1, list = FALSE)
sample_train <- edx_sample[-ind,]
sample_test <- edx_sample[ind,] %>% 
  semi_join(sample_train, by = "movieId") %>%
  semi_join(sample_train, by = "userId")
sample_train %>% tibble()
sample_test %>% tibble()

rm(edx, edx_sample, ind, sample)

#de-mean 
#global mean
g_mean <- mean(sample_train$rating)

#user and movie biases with count regulation lambda_u and lambda_m respectively
#tune the lambda by cross validation
set.seed(0, sample.kind= "Rounding")
ind_cv <- createFolds(sample_train$userId, k = 5)
train_cv <- list()
test_cv <- list()
for(k in 1:5){
  train_cv[[k]] <- sample_train[-ind_cv[[k]],]
  test_cv[[k]] <- sample_train[ind_cv[[k]],] %>% 
    semi_join(train_cv[[k]], by = "movieId") %>% 
    semi_join(train_cv[[k]], by = "userId")
}

rm(k, ind_cv)

#u_bias
#guessing and reviewing with plotting
lambda_search <- seq(5, 12, 0.1)
#implement parallel computing to save time
registerDoParallel(cores = 3)
ub_tune <- foreach(l = lambda_search, .combine = "cbind.data.frame") %:% 
  foreach(k = 1:5, .combine = "c", .packages = "data.table") %dopar% {
    u_bias <- train_cv[[k]][
      , .(u_bias = sum(rating - g_mean) / (l + .N)), by = .(userId)]
    
    pred <- u_bias[test_cv[[k]], on = .(userId)][
      , .(err = g_mean + u_bias - rating)]
    
    sqrt(mean(pred$err * pred$err))
  }

setDT(ub_tune)
setnames(ub_tune, as.character(lambda_search))
ub_rmse <- ub_tune[, lapply(.SD, mean)]
qplot(lambda_search, as.numeric(ub_rmse[1, ]), geom = c("point", "line"))
#try different ranges of lambda_search to find the bottom on plot
lambda_u <- lambda_search[which.min(ub_rmse[1,])]
u_bias <- sample_train[, .(u_bias = sum(rating - g_mean) / (lambda_u + .N)), 
                        by = .(userId)]

rm(ub_rmse, ub_tune, lambda_search, lambda_u)
#rm(mb_rmse, mb_tune, lambda_search)
#m_bias (repeat the u_bias tuning method)
lambda_search <- seq(1, 5, 0.1)
mb_tune <- foreach(l = lambda_search, .combine = "cbind.data.frame") %:% 
  foreach(k = 1:5, .combine = "c", .packages = "data.table") %dopar% {
    m_bias <- u_bias[train_cv[[k]], on = .(userId)][
      , .(m_bias = sum(rating - g_mean - u_bias) / (l + .N)), 
      by = .(movieId)]
    
    pred <- m_bias[u_bias[test_cv[[k]], 
                          on = .(userId)], 
                   on = .(movieId)][
                     , .(err = g_mean + u_bias + m_bias - rating)]
    
    sqrt(mean(pred$err * pred$err))
  }
  
setDT(mb_tune)
setnames(mb_tune, as.character(lambda_search))
mb_rmse <- mb_tune[, lapply(.SD, mean)]
qplot(lambda_search, as.numeric(mb_rmse[1,]), 
      geom = c("point", "line"))
lambda_m <- lambda_search[which.min(mb_rmse[1,])]
m_bias <- sample_train[u_bias, on = .(userId)][
  , .(m_bias = sum(rating - g_mean - u_bias) / (lambda_m + .N)), 
            by = .(movieId)]

rm(mb_tune, mb_rmse, lambda_search, lambda_m, train_cv, test_cv)

#genres
genres <- str_split(sample_train$genres, "\\|")
gen_cat <- genres %>% unlist() %>% unique()
n <- length(gen_cat)
gen_mean <- foreach(g = 1:n, .combine = "c", 
                    .packages = "data.table") %dopar% {
                    m_bias[u_bias[sample_train, on = .(userId)], 
                           on = .(movieId)][genres %like% gen_cat[g],
                    mean(g_mean + u_bias + m_bias - rating)]
                  }

rm(n)

m_id <- unique(sample_train$movieId)
m_n <- length(m_id)
m_gen <- foreach(i = 1:m_n, .packages = c("stringr", "data.table")) %dopar% {
            gens <- sample_train[movieId == m_id[i], genres][1] 
            gens <- str_split(gens, "\\|") %>% unlist()
            }
names(m_gen) <- m_id

ind <- foreach(g = 1:m_n) %dopar% {
          foreach(i = 1:length(m_gen[[g]]), .combine = "c", 
                  .packages = "stringr") %do% {
                  ind <- str_which(gen_cat, m_gen[[g]][i])}
              }

gen <- data.frame(gen_cat)
j <- 1
while(j <= m_n){
      gen[, j+1] <- gen_mean
      gen[-ind[[j]], j+1] <- 0
      j <- j + 1
      }
colnames(gen)[-1] <- m_id

rm(j, genres, ind, m_gen, gen_cat, gen_mean, m_n)

#rating residual table from the train set: rating - (g_mean + u_bias + m_bias)
residual_train <- m_bias[u_bias[sample_train, on = .(userId)]
                         , on = .(movieId)][
                           , resid := g_mean + u_bias + m_bias - rating]
sum(residual_train$resid == 0) #check overfittings 

#QQ plot to check the distribution
r_mean <- residual_train[, mean(resid)]
r_sd <- residual_train[, sd(resid)]
p <- seq(0.05, 0.95, 0.05)
r_quantile <- residual_train[, quantile(resid, p)]
n_quantile <- qnorm(p, r_mean, r_sd)
qplot(n_quantile, r_quantile) + geom_abline()

rm(p, r_quantile, n_quantile)

rtable <- dcast(residual_train, userId ~ movieId, value.var = "resid")
sum(!is.na(rtable[,-1]))/(dim(rtable)[1]*(dim(rtable)[2] - 1)) #sparsity
movieId <- names(rtable[,-1])
userId <- rtable[,1]

############sgd0
rtable_y0 <- setnafill(rtable[,-1], fill = 0)
rtable_y0 <- as(as.matrix(rtable_y0), "sparseMatrix")
R0 <- rtable_y0@x #training resid
U_i0 <- rtable_y0@i + 1 #row index of resid (user) 
#col index of resid (movie)
M_j0 <- rep(1:rtable_y0@Dim[2], diff(rtable_y0@p))

rm(rtable, residual_train)

#P Q starter matrix for sgd
f <- 200
set.seed(3, sample.kind = "Rounding")
P <- matrix(runif(f*rtable_y0@Dim[1], 0, 1), nrow = f)
set.seed(4, sample.kind = "Rounding")
Q <- matrix(runif(f*rtable_y0@Dim[2], 0, 1), nrow = f)

sgd <- function(P, Q, y, L_rate, lambda, batch_size, epochs){
  n <- length(y) 
  learning_log <- vector("list", epochs)
  
  for (t in 1:epochs){
    
    batch_id <- sample(1:n, batch_size, replace = FALSE)
    
    for (ui in batch_id){
      
      err_ui <- c(P[, U_i0[ui]] %*% Q[, M_j0[ui]] - y[ui]) 
      nabla_p <- err_ui * Q[, M_j0[ui]]  + lambda * P[,U_i0[ui]]
      nabla_q <- err_ui * P[, U_i0[ui]]  + lambda * Q[,M_j0[ui]]
      
      P[, U_i0[ui]] <- P[, U_i0[ui]] - L_rate * nabla_p
      Q[, M_j0[ui]] <- Q[, M_j0[ui]] - L_rate * nabla_q
    }
    
    err <- sapply(1:n, function(j){
      P[, U_i0[j]] %*% Q[, M_j0[j]] - y[j]
    })
    learning_log[[t]] <- sqrt(mean(err * err))
  }
  return(learning_log)
}

sgd0 <- sgd(P = P, Q = Q , y = R0, 
            L_rate = 0.02, lambda = 1, 
            batch_size = 30, epochs = 5000)
sgd0 <- unlist(sgd0)
qplot(x = c(1:(5000)), y = sgd0)
rm(sgd0)

#run the algo to search opt P,Q

L_rate = 0.02
lambda = 1 
batch_size = 30
epochs = 5000*5
n <- length(R0) #change all the y in below into R
learning_log0 <- vector("list", epochs)

for (t in 1:epochs){
  
  batch_id <- sample(1:n, batch_size, replace = FALSE)
  
  for (ui in batch_id){
    
    err_ui <- c(P[, U_i0[ui]] %*% Q[, M_j0[ui]] - R0[ui]) 
    nabla_p <- err_ui * Q[, M_j0[ui]]  + lambda * P[,U_i0[ui]]
    nabla_q <- err_ui * P[, U_i0[ui]]  + lambda * Q[,M_j0[ui]]
    
    P[, U_i0[ui]] <- P[, U_i0[ui]] - L_rate * nabla_p
    Q[, M_j0[ui]] <- Q[, M_j0[ui]] - L_rate * nabla_q
    
    rm(err_ui, nabla_p, nabla_q)
  }
  
  err <- sapply(1:n, function(j){
    P[, U_i0[j]] %*% Q[, M_j0[j]] - R0[j]
  })
  learning_log0[[t]] <- sqrt(mean(err * err))
  rm(err, batch_id)
}
rm(t,ui)

learning_log0 <- unlist(learning_log0)
qplot(x = c(1:epochs), y = learning_log0[1:epochs])
rm(L_rate, lambda)

colnames(P) <- userId$userId %>% as.character()
colnames(Q) <- movieId

#val
uid_test <- sample_test$userId %>% as.character()
mid_test <- sample_test$movieId %>% as.character()

i <- length(uid_test)
f_sgd <- foreach(i = 1:i, .combine ="c") %dopar% {
  P[, uid_test[i]] %*% Q[, mid_test[i]]
}
rm(i)

pred <- m_bias[u_bias[sample_test[, .(userId, movieId, rating)][
  , f_sgd := f_sgd], on = .(userId)], on = .(movieId)][
    , ":="(pred = pred <- g_mean + u_bias + m_bias - 0*f_sgd, 
           err = pred - rating)]

sqrt(mean(pred$err * pred$err))
rm(f_sgd, pred, learning_log0)
rm(P, Q)
###########

rtable_tr <- transpose(rtable, keep.names = "movieId", 
                      make.names = "userId")
rtable_tr$movieId <- as.numeric(rtable_tr$movieId)
m_id_dt <- data.table(movieId = m_id)
rtable <- rtable_tr[m_id_dt, on = .(movieId)]

rm(rtable_tr, movieId, m_id, m_id_dt)

rtable_y <- setnafill(rtable[,-1], fill = 0)
rtable_y <- as(as.matrix(rtable_y), "sparseMatrix")
gen_x <- as(as.matrix(gen[,-1]), "sparseMatrix")
gen_x <- t(gen_x)
set.seed(1, sample.kind = "Rounding")
ind_y<- createFolds(1: rtable_y@Dim[2], k = ceiling(rtable_y@Dim[2]/1000))

rm(rtable)
#choice between fitting data in limited ram with slow for loop
#speed up the learning with lapply but require bit ram space
#some how the lappy version does not work on windows
b <- length(ind_y)
system.time(u_beta <- lapply(1:b, function(k){
  fit <- cv.glmnet(gen_x, rtable_y[, ind_y[[k]]],
                   family = "mgaussian", 
                   type.measure = "mse", 
                   nfolds = 5, alpha = 0.5,
                   parallel = TRUE, trace.it = TRUE)
  coef(fit, s= "lambda.min")}
  )
)
rm(b)
u_beta <- unlist(u_beta)

u_betas <- list()
system.time(for(k in 1:b){
  fit <- cv.glmnet(gen_x, rtable_y[, ind_y[[k]]],
                     family = "mgaussian", 
                     type.measure = "mse", 
                     nfolds = 5, alpha = 0.5, 
                     parallel = TRUE, trace.it = TRUE)
  u_betas[[k]] <- coef(fit, s= "lambda.min")
  rm(fit)
  gc()
})
rm(k)
u_beta <- unlist(u_beta)

uid_test <- sample_test$userId %>% as.character()
mid_test <- sample_test$movieId %>% as.character()

gl <- length(sample_test$genres)
gen_bias <- foreach(i = 1:gl, .combine = "c") %dopar% {
  gen[,mid_test[i]] %*% u_beta[[uid_test[i]]][-1] + u_beta[[uid_test[i]]][1]
}
rm(gl)

pred <- m_bias[u_bias[sample_test[, .(userId, movieId, rating)][
    , gen_bias := gen_bias], on = .(userId)], on = .(movieId)][
    , ':='(pred = pred <- g_mean + u_bias + m_bias - gen_bias, 
           err = pred - rating)]

sqrt(mean(pred$err * pred$err))

<<<<<<< HEAD
<<<<<<< HEAD
predb <- m_bias[u_bias[validation[, .(userId, movieId, rating)][
  , gen_bias := gen_bias], on = .(userId)], on = .(movieId)][
    , ':='(pred = pred <- g_mean + u_bias + m_bias, 
           err = pred - rating)]
sqrt(mean(predb$err * predb$err))


=======
i <- length(u_beta)
u_beta_int <- foreach(k = 1:i, .combine = "c") %dopar% {
  u_beta[[k]][1]
=======
rm(gen, gen_x, ind_y, rtable_y, gen_bias, pred)

#update the gen_x with trained beta###################
k <- length(u_beta)
u_beta_int <- foreach(k = 1:k, .combine = "c") %dopar% {
  u_beta[[k]][1] + 0
}
uid <- names(u_beta) %>% as.numeric()
u_beta_int <- data.table(uid, u_beta_int)
rm(k)

residual_alt <- u_beta_int[residual_train, on = .(uid = userId)][
  , resid_alt := resid - u_beta_int
]
rtable_alt <- dcast(residual_alt, uid ~ movieId, value.var = "resid_alt")
uid_dt <- as.data.table(uid)
rtable_alt <- rtable_alt[uid_dt, on = .(uid)]
u_beta_alt <- Matrix(0, nrow = 7612, ncol = (length(u_beta[[1]]) - 1))
for(i in 1:7612){
  u_beta_alt[i, ] <- u_beta[[i]][-1]
>>>>>>> 85845a2d7b252e1bc14a66fe9ea55f65eea33bc9
}
>>>>>>> 8cfddb9fa9ed038526fd8b6deb346ede505d652a

rtable_alt_y <- setnafill(rtable_alt[,-1], fill = 0)
rtable_alt_y <- as(as.matrix(rtable_alt_y), "sparseMatrix")

rm(i, uid, uid_dt, residual_alt, residual_train, rtable_alt)

set.seed(2, sample.kind = "Rounding")
ind_alt_y <- createFolds(1: rtable_alt_y@Dim[2], 
                     k = ceiling(rtable_alt_y@Dim[2]/1000))
k <- length(ind_alt_y)
gen_alt_x <- lapply(1:k, function(m){
  temp <- cv.glmnet(u_beta_alt, rtable_alt_y[, ind_alt_y[[m]]], 
                    family = "mgaussian", intercept = FALSE, 
                    type.measure = "mse", 
                    parallel = TRUE, trace.it = TRUE)
  coef(temp, s = "lambda.min")
}) #method wont work, predictors are too sparse ->sgd
rm(k, ind_alt_y)

#sgd
##################################
uid_train <- sample_train$userId %>% as.character()
mid_train <- sample_train$movieId %>% as.character()
gl <- length(sample_train$genres)
gen_bias_train <- foreach(i = 1:gl, .combine = "c") %dopar% {
  gen[,mid_train[i]] %*% u_beta[[uid_train[i]]][-1] + u_beta[[uid_train[i]]][1]
}
rm(gl, uid_train, mid_train)

resid_gen <- m_bias[u_bias[sample_train[, .(userId, movieId, rating)][
  , gen_bias := gen_bias_train], on = .(userId)], on = .(movieId)][
    , err := g_mean + u_bias + m_bias - gen_bias - rating]
rtable_gen <- dcast(resid_gen, userId ~ movieId, value.var = "err")
uid_gen <- rtable_gen[,1]
mid_gen <- names(rtable_gen)[-1]
rtable_gen <- setnafill(rtable_gen[,-1], fill = 0)
rtable_gen <- Matrix(as.matrix(rtable_gen), "sparseMatrix")

R <- rtable_gen@x #training resid
U_i <- rtable_gen@i + 1 #row index of resid (user) 
#col index of resid (movie)
M_j <- rep(1:rtable_gen@Dim[2], diff(rtable_gen@p))

#P Q starter matrix for sgd
f <- 30
set.seed(3, sample.kind = "Rounding")
P <- matrix(runif(f*rtable_gen@Dim[1], 0, 1), nrow = f)
set.seed(4, sample.kind = "Rounding")
Q <- matrix(runif(f*rtable_gen@Dim[2], 0, 1), nrow = f)

<<<<<<< HEAD
#warm start P Q matrix for sgd
k <- 20
f_mean <- sqrt(r_mean/k)
f_sd<- r_sd/sqrt(k) #by LNN
set.seed(0, sample.kind = "Rounding")
P <- matrix(rnorm(k*rtable_sparse@Dim[1], f_mean, f_sd), nrow = k)
set.seed(0, sample.kind = "Rounding")
Q <- matrix(rnorm(k*rtable_sparse@Dim[2], f_mean, f_sd), nrow = k)

sgd <- function(P, Q, y, L_rate, lambda_p, lambda_q, batch_size, epochs){
  #y: resid to be trained on (rtable_sparse in dgCMatrix sparse format)
  #lambda_p/q: not tuned, try 
  #batch_size: sample size out of total number of training resid 
  #batch_size * epochs should be much larger than length(y) 
  r <- y
  n <- length(y) #number of training resid
  learning_log <- list()
=======
sgd <- function(P, Q, y, L_rate, lambda, batch_size, epochs){
  n <- length(y) 
  learning_log <- vector("list", epochs)
>>>>>>> 85845a2d7b252e1bc14a66fe9ea55f65eea33bc9
  
  for (t in 1:epochs){
    
    batch_id <- sample(1:n, batch_size, replace = FALSE)
    
    for (ui in batch_id){
        
        err_ui <- c(P[, U_i[ui]] %*% Q[, M_j[ui]] - y[ui]) 
        nabla_p <- err_ui * Q[, M_j[ui]]  + lambda * P[,U_i[ui]]
        nabla_q <- err_ui * P[, U_i[ui]]  + lambda * Q[,M_j[ui]]
        
        P[, U_i[ui]] <- P[, U_i[ui]] - L_rate * nabla_p
        Q[, M_j[ui]] <- Q[, M_j[ui]] - L_rate * nabla_q
    }
    
    err <- sapply(1:n, function(j){
      P[, U_i[j]] %*% Q[, M_j[j]] - y[j]
      })
    learning_log[[t]] <- sqrt(mean(err * err))
  }
  return(learning_log)
}

sgdl <- sgd(P = P, Q = Q , y = R, 
                       L_rate = 0.02, lambda = 1, 
                       batch_size = 30, epochs = 1000)
sgdl <- unlist(sgdl)
qplot(x = c(1:5000), y = sgdl)

ftbl <- data.frame(factors = f, 
                   rmse = min(sgdl))
ipdate <- c(f, min(sgdl))
ftbl <- rbind(ftbl, update)

rm(sgdl, update)

#####hyperparameters tunning
###factor length
lambda <- 1
L_rate <- 0.05
epochs <- 1000
batch_size <- 1000
n <- length(R)
factors <- seq(10, 20, 1)
f_tune <- foreach(f = factors, .combine = "c") %dopar% {
  set.seed(3, sample.kind = "Rounding")
  P <- matrix(runif(f*rtable_gen@Dim[1], 0, 1), nrow = f)
  set.seed(4, sample.kind = "Rounding")
  Q <- matrix(runif(f*rtable_gen@Dim[2], 0, 1), nrow = f)
  
  for (t in 1:epochs){
    
    batch_id <- sample(1:n, batch_size, replace = FALSE)
    
    for (ui in batch_id){
      
      err_ui <- c(P[, U_i[ui]] %*% Q[, M_j[ui]] - R[ui]) 
      nabla_p <- err_ui * Q[, M_j[ui]]  + lambda * P[,U_i[ui]]
      nabla_q <- err_ui * P[, U_i[ui]]  + lambda * Q[,M_j[ui]]
      
      P[, U_i[ui]] <- P[, U_i[ui]] - L_rate * nabla_p
      Q[, M_j[ui]] <- Q[, M_j[ui]] - L_rate * nabla_q
    }
  }
  err <- sapply(1:n, function(j){
      P[, U_i[j]] %*% Q[, M_j[j]] - R[j]
    })
  sqrt(mean(err * err))
}

qplot(x = factors, y = f_tune)
f_opt <- factors[which.min(f_tune)]
rm(f_tune, factors)

###learning rate
lambda <- 1
epochs <- 1000
batch_size <- 1000
n <- length(R)

set.seed(3, sample.kind = "Rounding")
P <- matrix(runif(f_opt*rtable_gen@Dim[1], 0, 1), nrow = f_opt)
set.seed(4, sample.kind = "Rounding")
Q <- matrix(runif(f_opt*rtable_gen@Dim[2], 0, 1), nrow = f_opt)

Lrts <- c(seq(0.1, 1, 0.1), seq(1:5, 1))
Lr_tune <- foreach(L_rate = Lrts, .combine = "c") %dopar% {
  
  for (t in 1:epochs){
    
    batch_id <- sample(1:n, batch_size, replace = FALSE)
    
    for (ui in batch_id){
      
      err_ui <- c(P[, U_i[ui]] %*% Q[, M_j[ui]] - R[ui]) 
      nabla_p <- err_ui * Q[, M_j[ui]]  + lambda * P[,U_i[ui]]
      nabla_q <- err_ui * P[, U_i[ui]]  + lambda * Q[,M_j[ui]]
      
      P[, U_i[ui]] <- P[, U_i[ui]] - L_rate * nabla_p
      Q[, M_j[ui]] <- Q[, M_j[ui]] - L_rate * nabla_q
    }
  }
  err <- sapply(1:n, function(j){
    P[, U_i[j]] %*% Q[, M_j[j]] - R[j]
  })
  sqrt(mean(err * err))
}

qplot(x = Lrts, y = Lr_tune)
L_rate_opt <- Lrts[which.min(Lr_tune)]
rm(Lrts, Lr_tune)

#####lambda
epochs <- 1000
batch_size <- 1000

set.seed(0, sample.kind = "Roudning")
Rid_cv <- createFolds(n, 5)
Rid_cv <- Rid_cv[[1]] %>% unlist()

train_R <- R[-Rid_cv]
test_R <- R[Rid_cv]

Ui_tr <- U_i[-Rid_cv]
Ui_tst <- U_i[Rid_cv]

Mj_tr <- M_j[-Rid_cv]
Mj_tst <- M_j[Rid_cv]

lmds <- c(seq(0.1, 1, 0.1), seq(1, 5, 1))
lmd_tune <- foreach(lambda = lmds, .combine = "c") %dopar% {
    tr_n <- length(train_R)
    tst_n <- length(test_R)
      
    for (t in 1:epochs){
      
      batch_id <- sample(1:tr_n, batch_size, replace = FALSE)
      
      for (ui in batch_id){
        
        err_ui <- c(P[, Ui_tr[ui]] %*% Q[, Mj_tr[ui]] - train_R[ui]) 
        nabla_p <- err_ui * Q[, Mj_tr[ui]]  + lambda * P[,Ui_tr[ui]]
        nabla_q <- err_ui * P[, Ui_tr[ui]]  + lambda * Q[,Mj_tr[ui]]
        
        P[, Ui_tr[ui]] <- P[, Ui_tr[ui]] - L_rate_opt * nabla_p
        Q[, Mj_tr[ui]] <- Q[, Mj_tr[ui]] - L_rate_opt * nabla_q
      }
    }
    err <- sapply(1:tst_n, function(j){
      P[, Ui_tst[j]] %*% Q[, Mj_tst[j]] - test_R[j]
    })
    sqrt(mean(err * err))
}

qplot(x = lmds, y = lmd_tune)
lambda_opt <- lmds[which.min(lmd_tune)]
rm(lmds, lmd_tune)

######################
L_rate = 0.01
lambda = 1 
batch_size = 30
epochs = 1000
n <- length(R) #change all the y in below into R
learning_log <- vector("list", epochs)

for (t in 1:epochs){
  
  batch_id <- sample(1:n, batch_size, replace = FALSE)
  
  for (ui in batch_id){
    
    err_ui <- c(P[, U_i[ui]] %*% Q[, M_j[ui]] - R[ui]) 
    nabla_p <- err_ui * Q[, M_j[ui]]  + lambda * P[,U_i[ui]]
    nabla_q <- err_ui * P[, U_i[ui]]  + lambda * Q[,M_j[ui]]
    
    P[, U_i[ui]] <- P[, U_i[ui]] - L_rate * nabla_p
    Q[, M_j[ui]] <- Q[, M_j[ui]] - L_rate * nabla_q
    
    rm(err_ui, nabla_p, nabla_q)
  }
  
  err <- sapply(1:n, function(j){
    P[, U_i[j]] %*% Q[, M_j[j]] - R[j]
  })
  learning_log[[t]] <- sqrt(mean(err * err))
  rm(err, batch_id)
}
rm(t, ui)

learning_log <- unlist(learning_log)
qplot(x = c(1:epochs), y = learning_log)
rm(L_rate, lambda)

colnames(P) <- uid_gen$userId %>% as.character()
colnames(Q) <- mid_gen

######gd

for (t in 1:epochs){
  
  batch_id <- sample(1:n)
  
  for (ui in batch_id){
    
    err_ui <- c(P[, U_i[ui]] %*% Q[, M_j[ui]] - R[ui]) 
    nabla_p <- err_ui * Q[, M_j[ui]]  + lambda * P[,U_i[ui]]
    nabla_q <- err_ui * P[, U_i[ui]]  + lambda * Q[,M_j[ui]]
    
    P[, U_i[ui]] <- P[, U_i[ui]] - L_rate * nabla_p
    Q[, M_j[ui]] <- Q[, M_j[ui]] - L_rate * nabla_q
    
    rm(err_ui, nabla_p, nabla_q)
  }
  
  err <- sapply(1:n, function(j){
    P[, U_i[j]] %*% Q[, M_j[j]] - R[j]
  })
  learning_log[[t]] <- sqrt(mean(err * err))
  rm(err, batch_id)
}
rm(t, ui)
learning_log <- unlist(learning_log)
qplot(x = 1:epochs, y = learning_log)

#####cpp

pq <- gd(U_i = U_i, M_j = M_j, y = R, u_n = 7612, m_n = 2867, 
   factor_n = 130, L_rate = 0.01, lambda = 1, epochs = 10000)
sum(is.nan(pq$P))
sum(is.nan(pq$Q))

colnames(pq$P) <- uid_gen$userId %>% as.character()
colnames(pq$Q) <- mid_gen

######################
#val
uid_test <- sample_test$userId %>% as.character()
mid_test <- sample_test$movieId %>% as.character()

i <- length(uid_test)
gen_bias <- foreach(i = 1:i, .combine = "c") %dopar% {
  gen[,mid_test[i]] %*% u_beta[[uid_test[i]]][-1] + u_beta[[uid_test[i]]][1]
}

f_gd <- foreach(i = 1:i, .combine ="c") %dopar% {
  pq$P[, uid_test[i]] %*% pq$Q[, mid_test[i]]
}
rm(i)

pred <- m_bias[u_bias[sample_test[, .(userId, movieId, rating)][
  , ":="(gen_bias = gen_bias, f_gd = f_gd)], on = .(userId)], 
  on = .(movieId)][
    , ":="(pred = pred <- g_mean + u_bias + m_bias - gen_bias - f_gd, 
            err = pred - rating)]

sqrt(mean(pred$err * pred$err))
rm(f_gd, pred, pq)
rm(pred)
###########
#validation
P <- as.data.frame(P) %>% setNames(u_id)
Q <- as.data.frame(Q) %>% setNames(m_id[-1])

rmse <- function(g_mean, u_bias, m_bias, P, Q, valid){
            
            pred <- valid[u_bias, on = .(userId)][
                                m_bias, on = .(movieId)][!is.na(rating), 
                                .(pred = g_mean + u_bias + m_bias), 
                                by = .(userId, movieId)]
            pred_uid <- as.character(unlist(pred[,1]))
            pred_mid <- as.character(unlist(pred[,2]))
            n <- length(pred$pred)
            
            pred_resids <- foreach(i = 1:n, 
                                   .combine = "c", 
                                   .packages = "data.table") %dopar% {
                                    P[,pred_uid[i]] %*% Q[,pred_mid[i]]
                                  }

            pred <- pred[, resid := pred_resids]
            err_rmse <- valid[pred, on = .(userId, movieId)][
                                    , .(err = pred + resid - rating), 
                                    by = .(userId, movieId)]
            return(sqrt(mean(err_rmse$err * err_rmse$err)))
        }

rmse(g_mean = g_mean, u_bias = u_bias, m_bias = m_bias, P = P, Q = Q, 
     valid = sample_test)

###

  