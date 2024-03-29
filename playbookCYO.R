library(tidyverse)
library(data.table)
library(caret)
library(doParallel)
library(e1071)
library(caretEnsemble)
library(GGally)

#kaggle: www.kaggle.com/datasets/uciml/adult-census-income
#github: github.com/dsjinx/harvardx-ds-capstone/blob/main/CYO/AdultCensusIncome.zip
data <- fread(unzip("AdultCensusIncome.zip"))
data <- fread("../CYO/adult.csv")
setwd("../CYO")

###########data exploration##########
str(data)

#inspect any strange and missing value
sum(is.na(data))
data[, lapply(.SD, function(j) sum(str_detect(j, "\\?")))]

#convert dependent income into two factor levels
data <- data[, income := as.factor(income)]
sapply(data, class)

#inspect the sample distribution over the to be predicted income var
ggplot(data, aes(x = income)) + geom_bar(width = 0.3) + 
  scale_y_log10() + labs(y = "log10(count)")
sum(data$income == "<=50K") / sum(data$income == ">50K")

plot_temp <- data[, .(pct = 100 * .N / dim(data)[1]), by = .(income)]
ggplot(plot_temp, aes(income, pct)) + 
  geom_bar(stat = "identity", width = 0.3) + 
  geom_text(aes(label = format(round(pct, 2), nsmall = 2)),
            vjust = 2, color = "white")

##numeric features 1st
names(data)
cols <- names(which(sapply(data, class) == "integer"))
summary(data[, ..cols]) #check for strange value

num_fp <- data[, ..cols][, lapply(.SD, 
              function(j) (j - mean(j)) / sd(j)), .SDcols = cols]
summary(num_fp) #check unusual outlier

###check the outliers in details
ind_min <- num_fp[, lapply(.SD, which.min)]
ind_max <- num_fp[, lapply(.SD, which.max)]
ind_out <- rbind(ind_min, ind_max)
outliers <- data.table(
  outlier = c("min", "max"),
  age = data$age[ind_out$age],
  fnlwgt = data$fnlwgt[ind_out$fnlwgt],
  ednum = data$education.num[ind_out$education.num],
  capgain = data$capital.gain[ind_out$capital.gain],
  caplos = data$capital.loss[ind_out$capital.loss],
  weekhrs = data$hours.per.week[ind_out$hours.per.week])

#plot to inspect any correlation exist across numeric features
num_fp <- num_fp[, income := data$income]

trellis.par.set("fontsize", list(text = 8.5))
featurePlot(x = num_fp[, 1:6], y = num_fp$income, plot = "box", 
            scales = list(y = list(relation = "free"),
                          x = list(rot = 90)),
            layout = c(3, 2), auto.key = list(columns = 2))
featurePlot(x = num_fp[, 1:6], y = num_fp$income, plot = "pairs", 
            auto.key = list(columns = 2))

##categorical/character features
char_cols <- names(data)[-which(names(data) %in% cols)][-9]

###check the missing ?
sum(data == "?") / (dim(data)[1] * dim(data)[2])
mis_locate <- data[, lapply(.SD, function(i) sum(i == "?")), 
                   .SDcols = char_cols]
mis_locate / dim(data)[1] * 100

char_fp <- data[, ..char_cols]
char_fp <- char_fp[, lapply(.SD, as.factor)] 
char_fp <- char_fp[, income := data$income]
str(char_fp)
sapply(char_fp, levels) #inspect any wired input

setDF(char_fp)
ggp1 <- lapply(1:3, function(j){
  ggplot(char_fp, aes(x = char_fp[,j], fill = income)) + 
    geom_bar() + scale_y_log10() + 
    theme(axis.text.x = element_text(angle = 90, size = 8)) + 
    facet_grid(income ~., scale = "free_y")
  })
ggmatrix(ggp1, nrow = 1, ncol = 3, xAxisLabels = char_cols[1:3])

ggp2 <- lapply(4:6, function(j){
  ggplot(char_fp, aes(x = char_fp[,j], fill = income)) + 
    geom_bar() + scale_y_log10() + 
    theme(axis.text.x = element_text(angle = 90, size = 8)) + 
    facet_grid(income ~., scale = "free_y")
})
ggmatrix(ggp2, nrow = 1, ncol = 3, xAxisLabels = char_cols[4:6])

ggp3 <- lapply(7:8, function(j){
  ggplot(char_fp, aes(x = char_fp[,j], fill = income)) + 
    geom_bar() + scale_y_log10() + 
    theme(axis.text.x = element_text(angle = 90, size = 8)) + 
    facet_grid(income ~., scale = "free_y")
})
ggmatrix(ggp3, nrow = 1, ncol = 2, xAxisLabels = char_cols[7:8])

rm(ggp1, ggp2, ggp3, char_fp, mis_locate, num_fp, outliers, 
   ind_out, ind_max, ind_min)

#methods details
#train/test split
set.seed(1, sample.kind = "Rounding")
ind <- createDataPartition(1:dim(data)[1], times = 1, list = FALSE, p = 0.2)
train <- data[-ind, ]
test <- data[ind, ]

#0. guessing
prev <- sum(train$income == "<=50K") / dim(train)[1]
set.seed(10, sample.kind = "Rounding")
pred_guess <- sample(c("<=50K", ">50K"), dim(test)[1], replace = TRUE, 
                     prob = c(prev, 1 - prev))
gauge_guess <- confusionMatrix(as.factor(pred_guess), test$income, 
                               mode = "prec_recall",
                               positive = ">50K")
print(gauge_guess)
gauge_guess$byClass["F1"]

#1. tree
registerDoParallel(cores = 4)

set.seed(2, sample.kind = "Rounding") #5 fold cv
ind_cv <- createFolds(1:dim(train)[1], k = 5, returnTrain = TRUE)

#define the cv index can make the train to be reproducible
treecontrol <- trainControl(method = "cv", index = ind_cv)
fit_tree <- train(income ~ ., data = train, method = "rpart",
              trControl = treecontrol, 
              tuneGrid = data.frame(cp = seq(0, 0.1, length.out = 10)))
plot(fit_tree)
fit_tree
fit_tree$finalModel
varImp(fit_tree, scale = FALSE) #get an idea of important predictors

pred_tree <- predict(fit_tree, test)
gauge_tree <- confusionMatrix(pred_tree, test$income, 
                              mode = "prec_recall",
                              positive = ">50K")
print(gauge_tree)
gauge_tree$byClass["F1"]

#2. random forest
#same cv index from the tree
rfcontrol <- trainControl(method = "cv", index = ind_cv)
tune_forest <- train(income ~., data = train, method = "rf",
                    trControl = rfcontrol, 
                    tuneGrid = data.frame(mtry = seq(2, 14, 2)))
plot(tune_forest)
tune_forest
best_mtry <- tune_forest$bestTune$mtry 
#tune the mtry 1st to decide how many predictors for each trees

nodesize <- seq(1, 20, 2)
tune_nds <- sapply(nodesize, function(nd){
  train(income ~., data = train, method = "rf",
        trControl = rfcontrol,
        tuneGrid = data.frame(mtry = best_mtry),
        nodesize = nd)$results$Accuracy
})
qplot(nodesize, tune_nds, geom = c("point", "line"))
best_node <- nodesize[which.max(tune_nds)]
#tune the nodesize to decide how big the leaf size

ntrees <- seq(10, 200, 10)
tune_ntree <- sapply(ntrees, function(nt){
  train(income ~., data = train, method = "rf",
        trControl = rfcontrol,
        tuneGrid = data.frame(mtry = best_mtry),
        nodesize = best_node,
        ntree = nt)$results$Accuracy})
qplot(ntrees, tune_ntree, geom = c("point", "line"))
best_ntree <- ntrees[which.max(tune_ntree)]
#tune the number of trees in the forest

fit_forest <- train(income ~., data = train, method = "rf",
                     tuneGrid = data.frame(mtry = best_mtry),
                     nodesize = best_node,
                     ntree = best_ntree)
varImp(fit_forest, scale = FALSE) #to see what criteria used in trees

pred_forest <- predict(fit_forest, test)
gauge_forest <- confusionMatrix(pred_forest, test$income, 
                                mode = "prec_recall",
                                positive = ">50K")
print(gauge_forest)
gauge_forest$byClass["F1"]

############TRY
set.seed(19, sample.kind = "Rounding")
try_ind <- sample(1:dim(train)[1], 3500, replace = FALSE)
try_train <- train[try_ind]

set.seed(92, sample.kind = "Rounding")
try_cv <- createFolds(1:3500, k = 5, returnTrain = TRUE)
trycontrol <- trainControl(method = "cv", index = try_cv)
trytune_forest <- train(income ~., data = try_train, method = "rf",
                    trControl = trycontrol, 
                    tuneGrid = data.frame(mtry = seq(2, 14, 2)))
plot(trytune_forest)
trytune_forest
trytune_forest$finalModel
best_mtry <- trytune_forest$bestTune$mtry

nodesize <- seq(1, 20, 5)
tune_nds <- sapply(nodesize, function(nd){
  train(income ~., data = try_train, method = "rf",
        trControl = trycontrol,
        tuneGrid = data.frame(mtry = best_mtry),
        nodesize = nd)$results$Accuracy})
qplot(nodesize, tune_nds, geom = c("point", "line"))
best_node <- nodesize[which.max(tune_nds)]

ntrees <- seq(10, 150, 10)
tune_ntree <- sapply(ntrees, function(nt){
  train(income ~., data = try_train, method = "rf",
        trControl = trycontrol,
        tuneGrid = data.frame(mtry = best_mtry),
        nodesize = best_node,
        ntree = nt)$results$Accuracy})
qplot(ntrees, tune_ntree, geom = c("point", "line"))
best_ntree <- ntrees[which.max(tune_ntree)]

try_forest <- train(income ~., data = try_train, method = "rf",
                    trControl = trycontrol,
                    tuneGrid = data.frame(mtry = seq(2, 14, 2)),
                    nodesize = best_node,
                    ntree = best_ntree)

############
#3. SVM
##digitise all categorical columns
data_digit <- data[, ..char_cols]
sapply(data_digit, uniqueN)
sapply(data_digit, unique)

data_digit[, lapply(.SD, function(j) sum(str_detect(j, "\\?")))]

data_digit <- data_digit[, 
              lapply(.SD, function(j) str_replace(j, "\\?", "0"))]

data_digit[, lapply(.SD, function(j) sum(str_detect(j, "\\?")))]
data_digit[, lapply(.SD, function(j) sum(str_detect(j, "0")))]

unique(data_digit$education)
sum(str_detect(data_digit$education, "10th"))

sapply(data_digit, unique)

##containing "?" cols mean "0"
wc <- 1:uniqueN(data_digit$workclass)
for(w in wc){data_digit$workclass[which(data_digit$workclass == 
                    unique(data_digit$workclass)[w])] <- wc[w] - 1}
rm(w, wc)

oc <- 1:uniqueN(data_digit$occupation)
for(o in oc){data_digit$occupation[which(data_digit$occupation == 
                    unique(data_digit$occupation)[o])] <- oc[o] - 1}
rm(o, oc)

#"0" is in 2nd place of uniqueN(), special treatment convert "2" -> "0"
nc <- 1:uniqueN(data_digit$native.country)
for(n in nc){data_digit$native.country[which(data_digit$native.country == 
                    unique(data_digit$native.country)[n])] <- nc[n]}
data_digit$native.country[which(data_digit$native.country == "2")] <- 0
rm(n, nc)

ed <- 1:uniqueN(data_digit$education)
for(e in ed){data_digit$education[which(data_digit$education == 
                    unique(data_digit$education)[e])] = ed[e]}
rm(e, ed)

ms <- 1:uniqueN(data_digit$marital.status)
for(m in ms){data_digit$marital.status[which(data_digit$marital.status == 
                    unique(data_digit$marital.status)[m])] = ms[m]}
rm(m, ms)

rl <- 1:uniqueN(data_digit$relationship)
for(r in rl){data_digit$relationship[which(data_digit$relationship == 
                    unique(data_digit$relationship)[r])] = rl[r]}
rm(r, rl)

rc <- 1:uniqueN(data_digit$race)
for(r in rc){data_digit$race[which(data_digit$race == 
                    unique(data_digit$race)[r])] = rc[r]}
rm(r, rc)

sx <- 1:uniqueN(data_digit$sex)
for(s in sx){data_digit$sex[which(data_digit$sex == 
                    unique(data_digit$sex)[s])] = sx[s]}
rm(s, sx)

data_digit <- data_digit[, lapply(.SD, as.numeric)]
data_digit <- data[, (char_cols) := data_digit]
str(data_digit)

#same partition index from trees
train_svm <- data_digit[-ind, ]
test_svm <- data_digit[ind, ]

#linear kernel
svmcontrol <- trainControl(method = "cv", index = ind_cv)
fit_svm <- train(income ~., data = train_svm, method = "svmLinear2",
                 trControl = svmcontrol, 
                 tuneGrid = data.frame(cost = c(
                      2^-13, 2^-10, 2^-8 ,2^-5, 2^-1, 1, 2^3)))
plot(fit_svm)
fit_svm
fit_svm$finalModel

pred_svm <- predict(fit_svm, test_svm)
gauge_svm <- confusionMatrix(pred_svm, test_svm$income, 
                             mode = "prec_recall",
                             positive = ">50K")
print(gauge_svm)
gauge_svm$byClass["F1"]

#2nd degree polynomial kernel, compare with linear kernel
c <- c(2^-5, 1, 2^5)
g <- c(2^-5, 1, 2^5)
para_grid <- expand.grid(cost = c, gamma = g)

#mini tuning cv set, mimic the income class distribution from original dataset
under_ind <- which(data$income == "<=50K")
above_ind <- which(data$income == ">50K")

set.seed(3, sample.kind = "Rounding")
mini_under <- sample(under_ind, length(under_ind) * 0.2, 
                     replace = FALSE)
set.seed(4, sample.kind = "Rounding")
mini_above <- sample(above_ind, length(above_ind) * 0.2,
                     replace = FALSE)
mini_ind <- c(mini_under, mini_above)

set.seed(5, sample.kind = "Rounding")
mini_cv <- createFolds(mini_ind, k = 5, returnTrain = TRUE)
mini_cv_ind <- lapply(1:5, function(k) mini_ind[mini_cv[[k]]])

tune_cg <- foreach(j = 1:dim(para_grid)[1], .combine = cbind.data.frame, 
                   .packages = "e1071") %:% 
  foreach(k = 1:5, .combine = c) %dopar% {
    cv_train <- svm(income ~., data = data_digit[mini_cv_ind[[k]],], 
                    cost = para_grid[j, 1], gamma = para_grid[j, 2], 
                    kernel = "polynomial", degree = 2)
    val_acc <- sum(predict(cv_train, data_digit[-mini_cv_ind[[k]],]) == 
      data_digit[-mini_cv_ind[[k]],]$income) / 
      dim(data_digit[-mini_cv_ind[[k]],])[1]
  }

tune_acc <- apply(tune_cg, 2, mean)
qplot(1:dim(tune_cg)[2], tune_acc, geom = c("point", "line"))
best_cg <- para_grid[which.min(tune_acc), ]

fit_svm_poly <- svm(income ~., data = train_svm, cost = best_cg$cost, 
              gamma = best_cg$gamma, kernel = "polynomial", degree = 2)

pred_svm_poly<- predict(fit_svm_poly, test_svm)
gauge_svm_poly <- confusionMatrix(pred_svm_poly, test_svm$income, 
                                  mode = "prec_recall",
                                  positive = ">50K")
print(gauge_svm_poly)
gauge_svm_poly$byClass["F1"]

polyfit_svm <- svm(income ~., data = data_digit[mini_ind,], 
                   cost = best_cg$cost, gamma = best_cg$gamma, 
                   kernel = "polynomial", degree = 2)

polypred_svm <- predict(polyfit_svm, test_svm)
polygauge_svm <- confusionMatrix(polypred_svm, test_svm$income, 
                                 mode = "prec_recall",
                                 positive = ">50K")
print(polygauge_svm)
polygauge_svm$byClass["F1"]

linfit_svm <- train(income ~., data = data_digit[mini_ind,], 
                    method = "svmLinear2", 
                    tuneGrid = data.frame(cost = c(2^-5)))

linpred_svm <- predict(linfit_svm, test_svm)
lingauge_svm <- confusionMatrix(linpred_svm, test_svm$income, 
                                mode = "prec_recall",
                                positive = ">50K")
print(lingauge_svm)
lingauge_svm$byClass["F1"]

sum_linpoly <- data.table(Kernel = c("polynomial", "linear"),
                          test_F_1 = c(polygauge_svm$byClass["F1"], 
                                       lingauge_svm$byClass["F1"]))
knitr::kable(sum_linpoly)
#mini sample proved in high dimension predictors, linear kernel is good enough 
#to produce a favourable result
#also proved large sample size can improve performance

#only use numeric predictors
numcols <- c(cols, "income")
numtrain_svm <- data[, ..numcols][-ind,]
numtest_svm <- data[, ..numcols][ind,]
numfit_svm <- train(income ~., data = numtrain_svm, method = "svmLinear2",
                    trControl = svmcontrol, 
                    tuneGrid = data.frame(cost = c(
                      2^-13, 2^-10, 2^-8 ,2^-5, 2^-1, 1, 2^3)))
plot(numfit_svm)
numfit_svm
numfit_svm$finalModel

numpred_svm <- predict(numfit_svm, numtest_svm)
numgauge_svm <- confusionMatrix(numpred_svm, numtest_svm$income,
                                mode = "prec_recall",
                                positive = ">50K")
print(numgauge_svm)
numgauge_svm$byClass["F1"]

sum_numfull <- data.table(Attributes = c("numeric", "full"), 
                          test_F_1 = c(numgauge_svm$byClass["F1"],
                                       gauge_svm$byClass["F1"]))
knitr::kable(sum_numfull)

#by eyeing the random forest important variables, 
#the "race" and "native.country" is not in the top list
#so exclude these two and try again
imptrain_svm <- data[, -c("race", "native.country")][-ind,]
imptest_svm <- data[, -c("race", "native.country")][ind,]
impfit_svm <- train(income ~., data = imptrain_svm, method = "svmLinear2",
                    trControl = svmcontrol, 
                    tuneGrid = data.frame(cost = c(
                      2^-13, 2^-10, 2^-8 ,2^-5, 2^-1, 1, 2^3)))
plot(impfit_svm)
impfit_svm
impfit_svm$finalModel

imppred_svm <- predict(impfit_svm, imptest_svm)
impgauge_svm <- confusionMatrix(imppred_svm, imptest_svm$income, 
                                mode = "prec_recall",
                                positive = ">50K")
print(impgauge_svm)
impgauge_svm$byClass["F1"]

sum_impfull <- data.table(Attributes = c("numeric", "full", "imp"), 
                          test_F_1 = c(numgauge_svm$byClass["F1"],
                                       gauge_svm$byClass["F1"],
                                       impgauge_svm$byClass["F1"]))

knitr::kable(sum_impfull)
#impsvm is best, slightly better than full predictor linear svm
#??table to summarise svm results

########try refer L177
set.seed(19, sample.kind = "Rounding")
try_ind <- sample(1:dim(train_svm)[1], 3500, replace = FALSE)
try_svtrain <- train_svm[try_ind, ]

#linear kernel
set.seed(92, sample.kind = "Rounding")
try_cv <- createFolds(1:3500, k = 5, returnTrain = TRUE)
trycontrol <- trainControl(method = "cv", index = try_cv)
try_svm <- train(income ~., data = try_svtrain, method = "svmLinear2",
                 trControl = trycontrol, 
                 tuneGrid = data.frame(
                   cost = c(1, 2^5)))
plot(try_svm)
try_svm
try_svm$finalModel

tst_svm <- predict(try_svm, test_svm)
cfm_svm <- confusionMatrix(tst_svm, test_svm$income, positive = ">50K")
print(cfm_svm)
F_meas(tst_svm, reference = test_svm$income)

#2nd degree polynomial kernel
c <- c(2^-2, 2^2)
g <- c(2^-5, 2^2)
para_grid <- expand.grid(cost = c, gamma = g)
tune_cg <- foreach(j = 1:dim(para_grid)[1], .combine = cbind.data.frame) %:% 
  foreach(k = 1:5, .combine = c) %dopar% {
    cv_train <- svm(income ~., data = train_svm[try_cv[[k]],], 
                    cost = para_grid[j, 1], gamma = para_grid[j, 2], 
                    kernel = "polynomial", degree = 2)
    val_acc <- sum(predict(cv_train, test_svm) == 
                     test_svm$income) / dim(test_svm)[1]
  } 
cv_acc <- apply(tune_cg, 2, mean)
qplot(1:4, cv_acc, geom = c("point", "line"))
best_cg <- para_grid[which.min(cv_acc), ]

cg_svm <- svm(income ~., data = train_svm, cost = best_cg$cost, 
              gamma = best_cg$gamma, kernel = "polynomial", degree = 2)
cg_pred <- predict(cg_svm, test_svm)
cfm_cg <- confusionMatrix(cg_pred, test_svm$income, positive = ">50K")
print(cfm_cg)
F_meas(cg_pred, reference = test_svm$income)

#saved draft 
tune_cg <- foreach(j = 1:dim(para_grid)[1], .combine = cbind.data.frame, 
                   .packages = "e1071") %:% 
  foreach(k = 1:5, .combine = c) %dopar% {
    cv_train <- svm(income ~., data = train_svm[ind_cv[[k]],], 
                    cost = para_grid[j, 1], gamma = para_grid[j, 2], 
                    kernel = "polynomial", degree = 2)
    val_acc <- sum(predict(cv_train, train_svm[-ind_cv[[k]],]) == 
            train_svm[-ind_cv[[k]],]$income) / dim(train_svm[-ind_cv[[k]],])[1]
  }

########

#4. glm
fit_glm <- train(income ~., data = train_svm, method = "glm", 
                 family = binomial)
fit_glm$finalModel #?

pred_glm <- predict(fit_glm, test_svm)
gauge_glm <- confusionMatrix(pred_glm, test_svm$income, 
                             mode = "prec_recall",
                             positive = ">50K")
print(gauge_glm)
gauge_glm$byClass["F1"]

impfit_glm <- train(income ~., data = imptrain_svm, method = "glm", 
                 family = binomial)
impfit_glm$finalModel #?

imppred_glm <- predict(impfit_glm, imptest_svm)
impgauge_glm <- confusionMatrix(imppred_glm, imptest_svm$income, 
                             mode = "prec_recall",
                             positive = ">50K")
print(impgauge_glm)
impgauge_glm$byClass["F1"] #about the same

sum_glm <- data.table(Attributes = c("full", "important"),
                      F_1 = c(gauge_glm$byClass["F1"], 
                              impgauge_glm$byClass["F1"]))
knitr::kable(sum_glm)

#5. ensemble
##caretEnsemble pkg does not recognise symbols, need to be replaced by charc
cnvt_income <- train_svm[, lapply(.SD, 
                             function (j) str_replace(j, "\\>", "gt")), 
                             .SDcols = c("income")]
cnvt_income <- cnvt_income[, lapply(.SD, 
                             function(j) str_replace(j, "\\<=", "loe"))]

cnvtst_income <- test_svm[, lapply(.SD, 
                            function (j) str_replace(j, "\\>", "gt")), 
                           .SDcols = c("income")]
cnvtst_income <- cnvtst_income[, lapply(.SD, 
                            function(j) str_replace(j, "\\<=", "loe"))]

train_svm[, income := cnvt_income] 
test_svm[, income := cnvtst_income]

##make sure rf is still consistence in the digitised dataset
enfit_rf <- train(income ~., data = train_svm, trControl = rfcontrol,
                  method = "rf", tuneGrid = data.frame(mtry = best_mtry),
                  nodesize = best_node, ntree = best_ntree)
enpred_rf <- predict(enfit_rf, test_svm)
engauge_rf <- confusionMatrix(enpred_rf, as.factor(test_svm$income), 
                              mode = "prec_recall",
                              positive = "gt50K")
engauge_rf$byClass["F1"]

##re-tune the rf with the digitised dateset
entune_rf <- train(income ~., data = train_svm, method = "rf",
                     trControl = rfcontrol, 
                     tuneGrid = data.frame(mtry = seq(1, 5, 1)))
plot(entune_rf)
entune_rf
entune_rf$finalModel
enbest_mtry <- entune_rf$bestTune$mtry

en_nodes <- seq(1, 20, 2)
entune_nds <- sapply(en_nodes, function(nd){
  train(income ~., data = train_svm, method = "rf",
        trControl = rfcontrol,
        tuneGrid = data.frame(mtry = enbest_mtry),
        nodesize = nd)$results$Accuracy
})
qplot(en_nodes, entune_nds, geom = c("point", "line"))
enbest_node <- en_nodes[which.max(entune_nds)]

en_ntrees <- seq(10, 200, 10)
entune_ntree <- sapply(en_ntrees, function(nt){
  train(income ~., data = train_svm, method = "rf",
        trControl = rfcontrol,
        tuneGrid = data.frame(mtry = enbest_mtry),
        nodesize = enbest_node,
        ntree = nt)$results$Accuracy})
qplot(en_ntrees, entune_ntree, geom = c("point", "line"))
enbest_ntree <- ntrees[which.max(entune_ntree)]

enfit_forest <- train(income ~., data = train_svm, method = "rf",
                    tuneGrid = data.frame(mtry = enbest_mtry),
                    nodesize = enbest_node,
                    ntree = enbest_ntree)
varImp(enfit_forest, scale = FALSE)

enpred_forest <- predict(enfit_forest, test_svm)
engauge_forest <- confusionMatrix(enpred_forest, as.factor(test_svm$income), 
                                  mode = "prec_recall",
                                  positive = "gt50K")
engauge_forest$byClass["F1"] #better than pre retuned

##now ensemble
crEnsem <- trainControl(method = "cv", index = ind_cv, 
                        savePredictions = "final", classProbs = TRUE)
ensemble_fit <- caretList(income ~., data = train_svm, trControl = crEnsem,
                          tuneList = list(
                          svm = caretModelSpec(method = "svmLinear2", 
                                  tuneGrid = data.frame(cost = 0.03125)),
                          glm = caretModelSpec(method = "glm", 
                                  family = binomial),
                          rf = caretModelSpec(method = "rf", 
                                  tuneGrid = data.frame(mtry = enbest_mtry),
                                  nodesize = enbest_node,
                                  ntree = enbest_ntree)))

modelCor(resamples(ensemble_fit))

stack <- caretStack(ensemble_fit, method = "glm", 
                    trControl = trainControl(method = "boot", number = 5, 
                                             savePredictions = "final"))
pred_stack <- predict(stack, test_svm)
gauge_stack <- confusionMatrix(pred_stack, as.factor(test_svm$income), 
                               mode = "prec_recall",
                               positive = "gt50K")
print(gauge_stack)
gauge_stack$byClass["F1"]

stack_boost <- caretStack(ensemble_fit, method = "AdaBoost.M1", 
                          trControl = trainControl(method = "boot", number = 5, 
                                                   savePredictions = "final"))
pred_stackbst <- predict(stack, test_svm)
gauge_stackbst <- confusionMatrix(pred_stack, as.factor(test_svm$income), 
                               mode = "prec_recall",
                               positive = "gt50K")
print(gauge_stackbst)
gauge_stackbst$byClass["F1"]

summary <- data.table(Methods = c("Guessing", "Tree.rpart", "Forest.rf", 
                                  "SVM.linear", "glm", 
                                  "Stacking.glm", "Stacking.boost"),
                      F_1_score = c(gauge_guess$byClass["F1"], 
                                    gauge_tree$byClass["F1"],
                                    gauge_forest$byClass["F1"],
                                    gauge_svm$byClass["F1"],
                                    gauge_glm$byClass["F1"],
                                    gauge_stack$byClass["F1"],
                                    gauge_stackbst$byClass["F1"]))
knitr::kable(summary)
