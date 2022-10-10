library(tidyverse)
library(data.table)
library(caret)
library(GGally)
library(doParallel)
library(e1071)

#https://www.kaggle.com/datasets/uciml/adult-census-income
data <- fread("adult.csv")

#data exploration
str(data)
sum(is.na(data))

data <- data[, income := as.factor(income)]
sapply(data, class)

ggplot(data, aes(x = income)) + geom_bar(width = 0.3) + 
  scale_y_log10() + labs(y = "log10(count)")
sum(data$income == "<=50K") / sum(data$income == ">50K")

##numeric features 1st
names(data)
cols <- c("age", "fnlwgt", "education.num", 
          "capital.gain", "capital.loss", "hours.per.week")
summary(data[, ..cols])
num_fp <- data[, ..cols][, lapply(.SD, 
              function(j) (j - mean(j)) / sd(j)), .SDcols = cols]
summary(num_fp)

###check the outliers
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

num_fp <- cbind(num_fp, data$income)

trellis.par.set("fontsize", list(text = 8.5))
featurePlot(x = num_fp[, 1:6], y = num_fp$income, plot = "box", 
            scales = list(y = list(relation = "free"),
                          x = list(rot = 90)),
            layout = c(3, 2), auto.key = list(columns = 2))
featurePlot(x = num_fp[, 1:6], y = num_fp$income, plot = "pairs", 
            auto.key = list(columns = 2))

##categorical features
char_cols <- names(data)[-which(names(data) %in% cols)][-9]

###check the missing ?
sum(data == "?") / (dim(data)[1] * dim(data)[2])
mis_locate <- data[, lapply(.SD, function(i) sum(i == "?")), 
                   .SDcols = char_cols]
mis_locate / dim(data)[1] * 100

char_fp <- data[, ..char_cols]
char_fp <- char_fp[, lapply(.SD, as.factor)] 
char_fp <- cbind(char_fp, income = data$income)
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

rm(ggp1, ggp2, ggp3, char_fp, mis_locate, char_cols, num_fp, 
   outliers, ind_out, ind_max, ind_min, cols)

#methods details
#train/test split
set.seed(1, sample.kind = "Rounding")
ind <- createDataPartition(1:dim(data)[1], times = 1, list = FALSE, p = 0.2)
train <- data[-ind, ]
test <- data[ind, ]

rm(ind)

#0. guessing
prev <- sum(train$income == "<=50K") / dim(train)[1]
set.seed(10, sample.kind = "Rounding")
pred_guess <- sample(c("<=50K", ">50K"), dim(test)[1], replace = TRUE, 
                     prob = c(prev, 1-prev))
confusionMatrix(as.factor(pred_guess), test$income, positive = ">50K")
F_meas(as.factor(pred_guess), reference = test$income)

#1. tree
registerDoParallel(cores = 3)

set.seed(2, sample.kind = "Rounding")
ind_cv <- createFolds(1:dim(train)[1], k = 5, returnTrain = TRUE)

treecontrol <- trainControl(method = "cv", index = ind_cv)
fit_tree <- train(income ~ ., data = train, method = "rpart",
              trControl = treecontrol, 
              tuneGrid = data.frame(cp = seq(0, 0.1, length.out = 10)))
plot(fit_tree)
fit_tree$results
fit_tree$bestTune
fit_tree$finalModel
varImp(fit_tree, scale = FALSE)

pred_tree <- predict(fit_tree, test)
gauge_tree <- confusionMatrix(pred_tree, test$income, positive = ">50K")
print(gauge_tree)
F_meas(pred_tree, reference = test$income)

#2. random forest
rfcontrol <- trainControl(method = "cv", index = ind_cv)
tune_forest <- train(income ~., data = train, method = "rf",
                    trControl = rfcontrol, 
                    tuneGrid = data.frame(mtry = seq(2, 14, 2)))
plot(tune_forest)
tune_forest$finalModel
best_mtry <- tune_forest$bestTune$mtry

nodesize <- c(10, 50, 100, 200, 500)
tune_nds <- sapply(nodesize, function(nd){
  train(income ~., data = train, method = "rf",
        trControl = rfcontrol,
        tuneGrid = data.frame(mtry = best_mtry),
        nodesize = nd)$results$Accuracy
})
qplot(nodesize, tune_nds, geom = c("point", "line"))
best_node <- nodesize[which.max(tune_nds)]

ntrees <- seq(10, 150, 10)
tune_ntree <- sapply(ntrees, function(nt){
  train(income ~., data = train, method = "rf",
        trControl = rfcontrol,
        tuneGrid = data.frame(mtry = best_mtry),
        nodesize = best_node,
        ntree = nt)$results$Accuracy})
qplot(ntrees, tune_ntree, geom = c("point", "line"))
best_ntree <- ntrees[which.max(tune_ntree)]

tuned_forest <- train(income ~., data = train, method = "rf",
                     trControl = rfcontrol, 
                     tuneGrid = data.frame(mtry = seq(2, 14, 2)),
                     nodesize = best_node,
                     ntree = best_ntree)
plot(tuned_forest)
tuned_forest$finalModel
best_mtry <- tune_forest$bestTune$mtry

fit_forest <- train(income ~., data = train, method = "rf",
                     tuneGrid = data.frame(mtry = best_mtry),
                     nodesize = best_node,
                     ntree = best_ntree)
fit_forest$finalModel
varImp(fit_forest, scale = FALSE)

pred_forest <- predict(fit_forest, test)
gauge_forest <- confusionMatrix(pred_forest, test$income, positive = ">50K")
print(gauge_forest)
F_meas(pred_forest, reference = test$income)

############TRY
set.seed(19, sample.kind = "Rounding")
try_train <- sample(1:dim(train)[1], 3500, replace = FALSE)
try_train <- train[try_train]
try_cv <- createFolds(1:3500, k = 5, returnTrain = TRUE)
trycontrol <- trainControl(method = "cv", index = try_cv)
trytune_forest <- train(income ~., data = try_train, method = "rf",
                    trControl = trycontrol, 
                    tuneGrid = data.frame(mtry = seq(2, 14, 2)))

plot(trytune_forest)
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

#SVM

