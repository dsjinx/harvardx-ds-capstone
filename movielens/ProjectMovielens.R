##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", 
                                         repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", 
                                     repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", 
                                          repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", 
                             readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), 
                          "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 3.6 or earlier:
movies <- as.data.frame(movies) %>% 
  mutate(movieId = as.numeric(levels(movieId))[movieId],
         title = as.character(title),
         genres = as.character(genres))
# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
# if using R 3.5 or earlier, use `set.seed(1)`
set.seed(1, sample.kind="Rounding") 
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, 
                                  list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)
#############################################

#save data set(edx, validation) at local directory, so we do not need to repeat 
#above process, if we get stuck or crash during the following process
save(edx, validation, file = "mledx.rdata")
load("mledx.rdata")

library(tidyverse)
library(data.table)
library(caret)
library(Matrix)
library(doParallel)
library(glmnet)
library(RcppArmadillo)

#global mean
g_mean <- mean(edx$rating)

#biases
#making 5-folds cross validation sets
set.seed(2, sample.kind= "Rounding")
ind_cv <- createFolds(edx$userId, k = 5)
#utilize the multicores on our computer to parallel the loop
#I am using total cores -1 to avoid system frozen
#number of cores only need to be registered once globally
registerDoParallel(cores = 3)
train_cv <- foreach(k = 1:5) %dopar% {edx[-ind_cv[[k]],]}
test_cv <- foreach(k = 1:5, .packages = "tidyverse") %dopar% {
  edx[ind_cv[[k]],] %>% semi_join(train_cv[[k]], by = "movieId") %>%
    semi_join(train_cv[[k]], by = "userId")}
rm(ind_cv)

#user-bias with ridge regression penalty
lambda_search <- seq(4, 8, 0.1)
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

#try different ranges of lambda_search to find the turning bottom in plot
qplot(lambda_search, as.numeric(ub_rmse[1, ]), geom = c("point", "line"))

lambda_u <- lambda_search[which.min(ub_rmse[1,])]
u_bias <- edx[, .(u_bias = sum(rating - g_mean) / (lambda_u + .N)),
              by = .(userId)] #composite the u_bias with the opt lambda

rm(ub_rmse, ub_tune, lambda_search, lambda_u)

#movie-bias with ridge regression penalty
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
m_bias <- edx[u_bias, on = .(userId)][
  , .(m_bias = sum(rating - g_mean - u_bias) / (lambda_m + .N)),
  by = .(movieId)]#user the opt lambda to composite m_bias

rm(mb_tune, mb_rmse, lambda_search, lambda_m, train_cv, test_cv)

#genres-bias with elastic net penalty
#clean the | in genres
genres <- str_split(edx$genres, "\\|")
gen_cat <- genres %>% unlist() %>% unique()

n <- length(gen_cat) #only one movie(#8606) has no gen info,
#otherwise we have to exclude those from following process

gen_mean <- foreach(g = 1:n, .combine = "c",
                    .packages = "data.table") %dopar% {
                      u_bias[m_bias[edx, on = .(movieId)],
                             on = .(userId)][genres %like% gen_cat[g],
                             mean(g_mean + u_bias + m_bias - rating)]
                    }#vector contains all the genres means
rm(n)

m_id <- unique(edx$movieId)
m_n <- length(m_id)

m_gen <- foreach(i = 1:m_n, .packages = c("stringr", "data.table")) %dopar% {
  gens <- edx[movieId == m_id[i], genres][1]
  str_split(gens, "\\|") %>% unlist()
}#all movies genres name character vector
names(m_gen) <- m_id

#convert the movies genres name char vector into a numeric vector
#each element corresponds to the index number in the genres names vector
#the gen_cat we constructed in genre name cleaning code chunk
ind <- foreach(g = 1:m_n, .packages = "foreach") %dopar% {
  foreach(i = 1:length(m_gen[[g]]), .combine = "c",
          .packages = "stringr") %do% {
            str_which(gen_cat, m_gen[[g]][i])}
}

gen <- data.frame(gen_cat)
j <- 1
while(j <= m_n){
  gen[, j+1] <- gen_mean
  gen[-ind[[j]], j+1] <- 0
  j <- j + 1
}#combine all the movies genres vector into one data frame
colnames(gen)[-1] <- m_id

rm(j, genres, ind, m_gen, gen_cat, gen_mean, m_n)

#rating residual table from the train set:
#(g_mean + u_bias + m_bias) - rating
residual_train <- u_bias[m_bias[edx, on = .(movieId)]
                         , on = .(userId)][
                           , .(resid = g_mean + u_bias + m_bias - rating),
                           by = .(userId, movieId)]
sum(residual_train$resid == 0) #check any overfittings

rtable <- dcast(residual_train, userId ~ movieId, value.var = "resid")

sum(!is.na(rtable[,-1]))/(dim(rtable)[1]*(dim(rtable)[2] - 1))#sparsity

movieId <- names(rtable[,-1])
rtable_tr <- transpose(rtable, keep.names = "movieId",
                       make.names = "userId")
rtable_tr$movieId <- as.numeric(rtable_tr$movieId)

#the following codes are to rearrange movie ids to align with the 
#movie genres mean table's id order
m_id_dt <- data.table(movieId = m_id)#m_id is got in above after gen_mean
rtable <- rtable_tr[m_id_dt, on = .(movieId)]

rm(rtable_tr, movieId, m_id_dt)

#movie x user rating matrix as response y
#movie genres mean table: movie x 20 matrix as predictor x
#see https://glmnet.stanford.edu/articles/glmnet.html for 'glmnet' pkg manual 
rtable_y <- setnafill(rtable[,-1], type = "const", fill = 0)
rtable_y <- as(as.matrix(rtable_y), "sparseMatrix")
gen_x <- as(as.matrix(gen[,-1]), "sparseMatrix")
gen_x <- t(gen_x)

#break training responses into small groups of 1000 to be RAM friendly
#because the 'cv.glmnet' will internally tune 100 lambdas
#the process requuirs lots RAM, and mine PC's 16G got crashed without 
#the small grouping
set.seed(3, sample.kind = "Rounding")
ind_y<- createFolds(1: rtable_y@Dim[2], k = ceiling(rtable_y@Dim[2]/1000))

k <- length(ind_y)
u_beta <- list()
for(k in 1:k){
  fit <- cv.glmnet(gen_x, rtable_y[, ind_y[[k]]],
                   family = "mgaussian",
                   type.measure = "mse",
                   nfolds = 5, alpha = 0.5,
                   parallel = TRUE, trace.it = TRUE)
  u_beta[[k]] <- coef(fit, s= "lambda.min")
  rm(fit)
  gc()
}#be warned, it can take up to 20hrs to finish the training
rm(k, ind_y)

u_beta <- unlist(u_beta)
#use the result to compose the gen_bias to be used in 'edx' set
uid_train <- edx$userId %>% as.character()
mid_train <- edx$movieId %>% as.character()

ml <- length(edx$movieId)
gen_bias_train <- foreach(i = 1:ml, .combine = "c",
                          .packages = "Matrix") %dopar% {
              gen[,mid_train[i]] %*% u_beta[[uid_train[i]]][-1] + 
                              u_beta[[uid_train[i]]][1]
                          }
rm(ml, uid_train, mid_train)

#repeat the same way in genres to get the residual from 
#g_mean + u_bias + m_bias - gen_bias - rating
#minus the gen_bias because it was also derived from 
#residual of g_mean + u_bias + m_bias - rating
resid_gen <- m_bias[u_bias[edx[, .(userId, movieId, rating)][
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

rm(resid_gen, rtable_gen)

#####hyperparameters tunning#####
#create train-test set(1 fold val)
set.seed(0, sample.kind = "Rounding")
Rid_cv <- createFolds(1:length(R), 5)
Rid_cv <- Rid_cv[[1]]

train_R <- R[-Rid_cv]
test_R <- R[Rid_cv]

tr_n <- length(train_R)
tst_n <- length(test_R)

Ui_tr <- U_i[-Rid_cv]
Ui_tst <- U_i[Rid_cv]

Mj_tr <- M_j[-Rid_cv]
Mj_tst <- M_j[Rid_cv]

###factor length
lambda <- 1 #random choice
L_rate <- 0.05 #random choice
epochs <- 1000
batch_size <- 10000
factors <- seq(20, 30, 1)

#use stochastic gradient descent in here
f_tune <- foreach(f = factors, .combine = "c") %dopar% {
  set.seed(3, sample.kind = "Rounding")
  P <- matrix(runif(f*rtable_gen@Dim[1], 0, 1), nrow = f)
  set.seed(4, sample.kind = "Rounding")
  Q <- matrix(runif(f*rtable_gen@Dim[2], 0, 1), nrow = f)
  
  for (t in 1:epochs){
    
    batch_id <- sample(1:tr_n, batch_size, replace = FALSE)
    
    for (ui in batch_id){
      
      err_ui <- c(P[, Ui_tr[ui]] %*% Q[, Mj_tr[ui]] - train_R[ui])
      nabla_p <- err_ui * Q[, Mj_tr[ui]]  + lambda * P[,Ui_tr[ui]]
      nabla_q <- err_ui * P[, Ui_tr[ui]]  + lambda * Q[,Mj_tr[ui]]
      
      P[, Ui_tr[ui]] <- P[, Ui_tr[ui]] - L_rate * nabla_p
      Q[, Mj_tr[ui]] <- Q[, Mj_tr[ui]] - L_rate * nabla_q
    }
  }
  err <- sapply(1:tst_n, function(j){
    P[, Ui_tst[j]] %*% Q[, Mj_tst[j]] - test_R[j]
  })
  rm(P, Q)
  sqrt(mean(err * err))
}

qplot(x = factors, y = f_tune, geom = c("point", "line"))

f_opt <- factors[which.min(f_tune)]

rm(f_tune, factors)#save for plot

###learning rate
rm(L_rate)

Lrts <- seq(0.01, 0.1, 0.01)

#SGD method as well, but with the found factor length from above
Lr_tune <- foreach(L_rate = Lrts, .combine = "c") %dopar% {
  set.seed(3, sample.kind = "Rounding")
  P <- matrix(runif(f_opt*rtable_gen@Dim[1], 0, 1), nrow = f_opt)
  set.seed(4, sample.kind = "Rounding")
  Q <- matrix(runif(f_opt*rtable_gen@Dim[2], 0, 1), nrow = f_opt)
  
  for (t in 1:epochs){
    
    batch_id <- sample(1:tr_n, batch_size, replace = FALSE)
    
    for (ui in batch_id){
      
      err_ui <- c(P[, Ui_tr[ui]] %*% Q[, Mj_tr[ui]] - train_R[ui])
      nabla_p <- err_ui * Q[, Mj_tr[ui]]  + lambda * P[,Ui_tr[ui]]
      nabla_q <- err_ui * P[, Ui_tr[ui]]  + lambda * Q[,Mj_tr[ui]]
      
      P[, Ui_tr[ui]] <- P[, Ui_tr[ui]] - L_rate * nabla_p
      Q[, Mj_tr[ui]] <- Q[, Mj_tr[ui]] - L_rate * nabla_q
    }
  }
  err <- sapply(1:tst_n, function(j){
    P[, Ui_tst[j]] %*% Q[, Mj_tst[j]] - test_R[j]
  })
  rm(P, Q)
  sqrt(mean(err * err))
}
qplot(x = Lrts, y = Lr_tune, geom = c("point", "line"))
#there is no obvious turning point at bottom, so use slope to search
lr_slope <- sapply(1:(length(Lr_tune) - 1), function(k){
  abs(Lr_tune[k+1] - Lr_tune[k]) / 0.001})

qplot(x = Lrts[-length(Lrts)], y = lr_slope, geom = c("point","line"))
#define the slope <1 as selection threshold of the choice learning rate
L_rate_opt <- Lrts[which(lr_slope < 0.1)][1]

rm(Lrts, Lr_tune, lr_slope)#keep for plotting

###lambda
rm(lambda)

set.seed(3, sample.kind = "Rounding")
P <- matrix(runif(f_opt*rtable_gen@Dim[1], 0, 1), nrow = f_opt)
set.seed(4, sample.kind = "Rounding")
Q <- matrix(runif(f_opt*rtable_gen@Dim[2], 0, 1), nrow = f_opt)

lmds <- seq(0.01, 0.1, length.out = 10)

#instead of SGD, use the full gradient descent in here
#write a C++ function as 'gdtune' to speed up the gradient descent loop
#see the https://www.rcpp.org/ for detailed C++ in R manual
#https://dirk.eddelbuettel.com/code/rcpp.armadillo.html for details in 
#linear algebra C++ library 'RcppArmadillo' in R
#the C++ code is in "lmdtune.cpp" file in the github:
#https://github.com/dsjinx/harvardx-ds-capstone/tree/main/movielens

lmd_tune <- sapply(lmds, function(lmd){
  gdtune(P = P, Q = Q, ytr = train_R, ytst = test_R,
         Uitr = Ui_tr, Mjtr = Mj_tr, Uitst = Ui_tst, Mjtst = Mj_tst,
         L_rate = L_rate_opt, lambda = lmd, epochs = 5)
})#function returns err^2 for each lambda try
#the C++ code is ready for use in "lmdtune.cpp" file 

lmd_tune <- sqrt(lmd_tune / length(test_R))#compose the err^2 results into rmse

qplot(x = lmds, y = lmd_tune, geom = c("point", "line"))

lmd_opt <- lmds[which.min(lmd_tune)]

rm(P, Q, lmds, lmd_tune)#save for plot

#use the tuned factor length, learning rate ,and lambda to train by gradient
#descent for the matrix P, Q
#the gradient descent loop is also written in C++ as the function 'gd()'
#the the C++ code is ready for use in "gd.cpp" file in the github:
#https://github.com/dsjinx/harvardx-ds-capstone/tree/main/movielens

set.seed(5, sample.kind = "Rounding")
pq <- gd(U_i = U_i, M_j = M_j, y = R,
         u_n = rtable_gen@Dim[1], m_n = rtable_gen@Dim[2],
         factor_n = f_opt, L_rate = L_rate_opt,
         lambda = lmd_opt, epochs = 50)

#check any 'nan' result
#if any, then L_rate is too big, cause the training results exploded
sum(is.nan(pq$P))
sum(is.nan(pq$Q))

colnames(pq$P) <- uid_gen$userId %>% as.character()
colnames(pq$Q) <- mid_gen

###final validation###
#get the 'userId' and 'movieId' from validation dataset
uid_val <- validation$userId %>% as.character()
mid_val <- validation$movieId %>% as.character()

i <- length(validation$rating)
#use the 'userId' and 'movieId' to call 
#the respective movie and user gen vectors to construct gen_bias(b_gui)
gen_bias <- foreach(i = 1:i, .combine = "c") %dopar% {
  gen[,mid_val[i]] %*% u_beta[[uid_val[i]]][-1] + u_beta[[uid_val[i]]][1]
}

#call the user/movie id respective latent factor vectors for lf_ui
f_gd <- foreach(i = 1:i, .combine ="c") %dopar% {
  pq$P[, uid_val[i]] %*% pq$Q[, mid_val[i]]
}

#prediction composition and get the error for RMSE calculation
#!!be aware the "gen_bias" and "f_gd" are subtracted in prediction composition
#by the reason that both of them were trained from the residuals 
#mean + u_b + m_b - rating & mean + u_b + m_b - g_b - rating!!
pred <- m_bias[u_bias[validation[, .(userId, movieId, rating)][
  , ":="(gen_bias = gen_bias, f_gd = f_gd)], on = .(userId)],
  on = .(movieId)][
    , ":="(pred = pred <- g_mean + u_bias + m_bias - gen_bias - f_gd,
           err = pred - rating)]

sqrt(mean(pred$err * pred$err))#compose the final RMSE
