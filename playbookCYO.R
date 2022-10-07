library(tidyverse)
library(data.table)
library(caret)
library(GGally)
library(e1071)

data <- fread("adult.csv")

#data exploration
str(data)
sum(is.na(data))

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

num_fp <- cbind(num_fp, income = factor(data$income))

trellis.par.set("fontsize", list(text = 8.5))
featurePlot(x = num_fp[, 1:6], y = num_fp$income, plot = "box", 
            scales = list(y = list(relation = "free"),
                          x = list(rot = 90)),
            layout = c(3, 2), auto.key = list(columns = 2))
featurePlot(x = num_fp[, 1:6], y = num_fp$income, plot = "pairs", 
            auto.key = list(columns = 2))

##character features
char_cols <- names(data)[-which(names(data) %in% cols)][-9]

###check the missing ?
sum(data == "?") / (dim(data)[1] * dim(data)[2])
mis_locate <- data[, lapply(.SD, function(i) sum(i == "?")), 
                   .SDcols = char_cols]
mis_locate / dim(data)[1] * 100

char_fp <- data[, ..char_cols]
char_fp <- cbind(char_fp, income = data$income)
char_fp <- char_fp[, lapply(.SD, as.factor)] 
str(char_fp)
sapply(char_fp, levels) #not be shown in pdf, inspect any wired input

char_fp %>% setDF() %>% ggplot(aes(x = char_fp[,1:8])) + 
  geom_bar() + scale_y_log10() + 
  theme(axis.text.x = element_text(angle = 45, size = 8)) +
  facet_grid(income ~ ., scales = "free_y")

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

#methods details
#1. rpart
