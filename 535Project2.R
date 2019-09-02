##### Loading Packages ##### 
library(data.table)
library(dplyr)
library(class)
library(caret)
library(ggplot2)
library(glmnet)
library(e1071)
library(NMF)
library(penalized)
library(ncvreg)
library(nnet)
library(randomForest)
library(RSNNS)
library(neuralnet)
library(extlasso)
library(kernlab)
library(DMwR)
library(MASS)
library(ada)
library(xgboost)
testing <- fread("test_dat.csv")
training <- fread("train_dat.csv")
training_label <- read.csv("train_labels.csv", header = F)$V1
dim(training)
dim(testing)

#### ！！！！！！！ LASSO ！！！！！！#####
##### .... Linear SVM #####
##### ```````` precision:  ; fitting: 0; validation: 0.02833333####
set.seed(123)
regular1 <- cv.glmnet(x = data.matrix(training), y = training_label, 
                      family = "binomial", alpha = 1, type.measure = "class")
plot(regular1)
optlambda <- regular1$lambda.min
location <- as.vector(which(coef(regular1, s = "lambda.min")[-1, 1] != 0))
length(location)

set.seed(12345)
tune_linear <- tune.svm(x = data.matrix(training)[, location], y = as.factor(training_label), type = "C-classification", kernel = "linear")

tune_linear$best.performance
tune_linear$best.model

svm_linear <- svm(x = data.matrix(training)[, location], y = training_label, type = "C-classification", kernel = "linear", gamma = 0.01694915, cost = 1)
mean(predict(svm_linear, data.matrix(training)[, location]) != training_label)

model.1 <- predict(svm_linear, data.matrix(testing)[, location])

#### ！！！！！！！ MCP ！！！！！！#####
set.seed(12345)
cvMCP <- cv.ncvreg(X = data.matrix(training), seed = 1234, y = training_label, family = "binomial", penalty = "MCP")
plot(cvMCP)

location <- as.vector(which(coef(cvMCP, s = "lambda.min")[-1] != 0))
length(location)
##### .... Linear SVM **##### 
##### ```````` precision: 0.96388 ; fitting: 0.006666667; validation: 0.015####
set.seed(12345)
tune_linear <- tune.svm(x = data.matrix(training)[, location], y = as.factor(training_label), type = "C-classification", kernel = "linear", class.weights =  c("0" = 1, "1" = 1))

tune_linear$best.performance
tune_linear$best.model

svm_linear <- svm(x = data.matrix(training)[, location], y = training_label, type = "C-classification", kernel = "linear", gamma = 0.05, cost = 1, class.weights =  c("0" = 1, "1" = 1), probability = T)
mean(predict(svm_linear, data.matrix(training)[, location]) != training_label)

model.2 <- predict(svm_linear, data.matrix(testing)[, location])


#### .... Oversampling + Linear SVM  ####
##### ```````` precision:  ; fitting: 0.003722084; validation: 0.006203704####
set.seed(12)
sub.lab1 <- sample(which(training_label == 1), 206, replace = T)
sam.ind <- c(1:600, sub.lab1)
table(training_label[sam.ind])
dim(data.matrix(training)[sam.ind, location])

set.seed(12345)
cvMCP <- cv.ncvreg(X = data.matrix(training[sam.ind, ]), seed = 1234, y = training_label[sam.ind], family = "binomial", penalty = "MCP")
plot(cvMCP)
location <- as.vector(which(coef(cvMCP, s = "lambda.min")[-1] != 0))
length(location)


set.seed(12345)
tune_linear <- tune.svm(x = data.matrix(training[sam.ind, ])[, location], y = as.factor(training_label[sam.ind]), type = "C-classification", kernel = "linear")

tune_linear$best.performance
tune_linear$best.model

svm_linear <- svm(x = data.matrix(training[sam.ind, ])[, location], y = as.factor(training_label[sam.ind]), type = "C-classification", kernel = "linear", gamma = 0.05, cost = 1)
mean(predict(svm_linear, data.matrix(training[sam.ind, ])[, location]) != training_label[sam.ind])
mean(predict(svm_linear, data.matrix(training[sam.ind, ])[, location]) != training_label[sam.ind])

model.3 <- predict(svm_linear, data.matrix(testing)[, location])


#### ....Hierarchical Clustering (manhattan + ward.D + 3)+ Linear SVM  ####
##### ```````` precision:  ; fitting: 0.006666667; validation: 0.01333333####
d <- dist(scale(data.matrix(full.dat)[, location]), method = "manhattan")
Hierclus <- hclust(d, method = "ward.D")
plot(Hierclus)
cohort <- factor(cutree(Hierclus, k = 3))

dum <- dummyVars(~ ., data.frame(cohort))
new <- data.frame(predict(dum, data.frame(cohort)))

set.seed(12345)
tune_linear <- tune.svm(x = cbind(data.matrix(training)[, location], new[1:600, ]), y = as.factor(training_label), type = "C-classification", kernel = "linear")

tune_linear$best.performance
tune_linear$best.model

svm_linear <- svm(x = cbind(data.matrix(training)[, location], new[1:600, ]), 
                  y = training_label, type = "C-classification", 
                  kernel = "linear", gamma = 0.04347826, cost = 1)
mean(predict(svm_linear, cbind(data.matrix(training)[, location], new[1:600, ])) != training_label)

model.4 <- predict(svm_linear, cbind(data.matrix(testing)[, location], new[601:1800, ]))


### ....Oversampling + Hierarchical clustering��manhattan + ward.d + 3�� + svm ####
##### ```````` precision:  ; fitting: 0.00248139; validation: 0.01242284####
set.seed(1234)
tune_linear <- tune.svm(x = cbind(data.matrix(training)[sam.ind, location], new[sam.ind, ]), y = as.factor(training_label[sam.ind]), type = "C-classification", kernel = "linear")

tune_linear$best.performance
tune_linear$best.model

svm_linear <- svm(x = cbind(data.matrix(training)[sam.ind, location], new[sam.ind, ]), 
                  y = training_label[sam.ind], type = "C-classification", 
                  kernel = "linear", gamma = 0.04347826, cost = 1)
mean(predict(svm_linear, cbind(data.matrix(training)[sam.ind, location], new[sam.ind, ])) != training_label[sam.ind])

model.5 <- predict(svm_linear, cbind(data.matrix(testing)[, location], new[601:1800, ]))






#### Ensamble Learning: Stacking ####
#### *** PHASE 1####
#### __________MCP penalty selection####
set.seed(12345)
cvMCP <- cv.ncvreg(X = data.matrix(training), seed = 1234, y = training_label, family = "binomial", penalty = "MCP")
plot(cvMCP)

location <- as.vector(which(coef(cvMCP, s = "lambda.min")[-1] != 0))
length(location)

#### 1. LSVM ####
set.seed(1)
cv.ind <- createFolds(training_label, k = 10, list = F)


fit.1 <- c()
test.1 <- c()
for (i in 1:10) {
  train.ind <- which(cv.ind != i); test.ind <- which(cv.ind == i)
  
  set.seed(12345)
  tune_linear <- tune.svm(x = data.matrix(training)[train.ind, location], y = as.factor(training_label)[train.ind], type = "C-classification", kernel = "linear")
  optgamma <- tune_linear$best.model$gamma
  optcost <- tune_linear$best.model$cost
  
  svm_linear <- svm(x = data.matrix(training)[train.ind, location], y = training_label[train.ind], type = "C-classification", kernel = "linear", gamma = optgamma, cost = optcost)
  
  
  fit.1 <- c(fit.1, predict(svm_linear, data.matrix(training)[test.ind, location]))
  test.1 <- cbind(test.1, predict(svm_linear, data.matrix(testing)[, location]))
}


fit.11 <- fit.1 - 1
test.11 <- apply(test.1, 1, function(x) {ifelse(mean(x == 2) > 0.5, 2, 1)}) - 1

mean(fit.11 != training_label[as.vector(sapply(1:10, function(i) which(cv.ind == i)))])
0.015
####  2. Weighted LSVM ####
403/table(training_label)

fit.2 <- c()
test.2 <- c()
for (i in 1:10) {
  train.ind <- which(cv.ind != i); test.ind <- which(cv.ind == i)
  
  set.seed(12345)
  tune_linear <- tune.svm(x = data.matrix(training)[train.ind, location], y = as.factor(training_label)[train.ind], type = "C-classification", kernel = "linear", class.weights = c("0" = 1, "1" = 2))
  optgamma <- tune_linear$best.model$gamma
  optcost <- tune_linear$best.model$cost
  
  svm_linear <- svm(x = data.matrix(training)[train.ind, location], y = training_label[train.ind], type = "C-classification", kernel = "linear", gamma = optgamma, cost = optcost, class.weights = c("0" = 1, "1" = 2))
  
  
  fit.2 <- c(fit.2, predict(svm_linear, data.matrix(training)[test.ind, location]))
  test.2 <- cbind(test.2, predict(svm_linear, data.matrix(testing)[, location]))
}

fit.21 <- fit.2 - 1
test.21 <- apply(test.2, 1, function(x) {ifelse(mean(x == 2) > 0.5, 2, 1)}) - 1
mean(fit.21 != training_label[as.vector(sapply(1:10, function(i) which(cv.ind == i)))])
0.01333333

####  3. Clustering + LSVM ####
d <- dist(scale(data.matrix(full.dat)[, location]), method = "manhattan")
Hierclus <- hclust(d, method = "ward.D")
plot(Hierclus)
cohort <- factor(cutree(Hierclus, k = 3))

dum <- dummyVars(~ ., data.frame(cohort))
new <- data.frame(predict(dum, data.frame(cohort)))


fit.3 <- c()
test.3 <- c()
for (i in 1:10) {
  train.ind <- which(cv.ind != i); test.ind <- which(cv.ind == i)
  
  set.seed(12345)
  tune_linear <- tune.svm(x = cbind(data.matrix(training)[train.ind, location], new[train.ind, ]),
                          y = as.factor(training_label)[train.ind], type = "C-classification", 
                          kernel = "linear")
  
  optgamma <- tune_linear$best.model$gamma
  optcost <- tune_linear$best.model$cost
  
  svm_linear <- svm(x = cbind(data.matrix(training)[train.ind, location], new[train.ind, ]),
                    y = training_label[train.ind], type = "C-classification", 
                    kernel = "linear", gamma = optgamma, cost = optcost)
  
  
  fit.3 <- c(fit.3, predict(svm_linear, 
                            cbind(data.matrix(training)[test.ind, location], new[test.ind, ])))
  test.3 <- cbind(test.3, predict(svm_linear, cbind(data.matrix(testing)[, location], new[601:1800, ])))
}

fit.31 <- fit.3 - 1
test.31 <- apply(test.3, 1, function(x) {ifelse(mean(x == 2) > 0.5, 2, 1)}) - 1
mean(fit.31 != training_label[as.vector(sapply(1:10, function(i) which(cv.ind == i)))])
0.01666667

####  4. Clustering + Weighted LSVM ####
d <- dist(scale(data.matrix(full.dat)[, location]), method = "manhattan")
Hierclus <- hclust(d, method = "ward.D")
plot(Hierclus)
cohort <- factor(cutree(Hierclus, k = 3))

dum <- dummyVars(~ ., data.frame(cohort))
new <- data.frame(predict(dum, data.frame(cohort)))


fit.4 <- c()
test.4 <- c()
for (i in 1:10) {
  train.ind <- which(cv.ind != i); test.ind <- which(cv.ind == i)
  
  set.seed(12345)
  tune_linear <- tune.svm(x = cbind(data.matrix(training)[train.ind, location], new[train.ind, ]),
                          y = as.factor(training_label)[train.ind], type = "C-classification", 
                          kernel = "linear", class.weights = c("0" = 1, "1" = 2))
  
  optgamma <- tune_linear$best.model$gamma
  optcost <- tune_linear$best.model$cost
  
  svm_linear <- svm(x = cbind(data.matrix(training)[train.ind, location], new[train.ind, ]),
                    y = training_label[train.ind], type = "C-classification", kernel = "linear", 
                    gamma = optgamma, cost = optcost, class.weights = c("0" = 1, "1" = 2))
  
  
  fit.4 <- c(fit.4, predict(svm_linear, 
                            cbind(data.matrix(training)[test.ind, location], new[test.ind, ])))
  test.4 <- cbind(test.4, predict(svm_linear, cbind(data.matrix(testing)[, location], new[601:1800, ])))
}

fit.41 <- fit.4 - 1
test.41 <- apply(test.4, 1, function(x) {ifelse(mean(x == 2) > 0.5, 2, 1)}) - 1
mean(fit.41 != training_label[as.vector(sapply(1:10, function(i) which(cv.ind == i)))])
0.01833333
#### 5. Random Forest ####
fit.5 <- c()
test.5 <- c()
for (i in 1:10) {
  train.ind <- which(cv.ind != i); test.ind <- which(cv.ind == i)
  
  set.seed(12345)
  rf <- randomForest(x = data.matrix(training)[train.ind, location], 
                     y = factor(training_label)[train.ind])
  
  fit.5 <- c(fit.5, predict(rf, data.matrix(training)[test.ind, location]))
  test.5 <- cbind(test.5, predict(rf, data.matrix(testing)[, location]))
}

fit.51 <- fit.5 - 1
test.51 <- apply(test.5, 1, function(x) {ifelse(mean(x == 2) > 0.5, 2, 1)}) - 1
mean(fit.51 != training_label[as.vector(sapply(1:10, function(i) which(cv.ind == i)))])
0.1583333
#### 6. KNN ####
fit.6 <- c()
test.6 <- c()
for (i in 1:10) {
  train.ind <- which(cv.ind != i); test.ind <- which(cv.ind == i)
  
  set.seed(12345)
  cv.knn <- tune.knn(x = data.matrix(training)[train.ind, location], y = factor(training_label)[train.ind])
  
  optk <- cv.knn$best.model$k
  optl <- cv.knn$best.model$l
  
  fit.6 <- c(fit.6, 
             knn(train = data.matrix(training)[train.ind, location], cl = factor(training_label)[train.ind], k = optk, l = optl, test = data.matrix(training)[test.ind, location]))
  test.6 <- cbind(test.6, 
                  knn(train = data.matrix(training)[train.ind, location], cl = factor(training_label)[train.ind], k = optk, l = optl, test = data.matrix(testing)[, location]))
}

fit.61 <- fit.6 - 1
test.61 <- apply(test.6, 1, function(x) {ifelse(mean(x == 2) > 0.5, 2, 1)}) - 1
mean(fit.61 != training_label[as.vector(sapply(1:10, function(i) which(cv.ind == i)))])
0.205

#### 7. QDA ####
fit.7 <- c()
test.7 <- c()
for (i in 1:10) {
  train.ind <- which(cv.ind != i); test.ind <- which(cv.ind == i)
  
  qda.mod <- qda(x = data.matrix(training)[train.ind, location], 
                 grouping =  factor(training_label)[train.ind])
  
  fit.7 <- c(fit.7, predict(qda.mod, data.matrix(training)[test.ind, location])$class)
  test.7 <- cbind(test.7, predict(qda.mod, data.matrix(testing)[, location])$class)
}

fit.71 <- fit.7 - 1
test.71 <- apply(test.7, 1, function(x) {ifelse(mean(x == 2) > 0.5, 2, 1)}) - 1
mean(fit.71 != training_label[as.vector(sapply(1:10, function(i) which(cv.ind == i)))])
0.1016667

#### 8. BP network ####
fit.8 <- c()
test.8 <- c()
for (i in 1:10) {
  train.ind <- which(cv.ind != i); test.ind <- which(cv.ind == i)
  
  ctrl <- trainControl(method = "CV", number = 10, selectionFunction = "best")
  set.seed(1234)
  cv.nnet <- caret::train(x = data.matrix(training)[train.ind, location],
                             y = factor(training_label[train.ind]), trControl = ctrl,
                             tuneGrid = expand.grid(.layer1 = c(1:5), .layer2 = c(1:5), 
                                                    .layer3 = c(1:5)),
                             learningrate = 0.01, threshold = 0.01, stepmax = 5e5,
                             metric = "Accuracy", method = "mlpML")
  
  layers <- as.numeric(cv.nnet$bestTune)
  
  nnet.model <- mlp(x = data.matrix(training)[train.ind, location],
                    y = decodeClassLabels(training_label[train.ind]), size = layers)
  
  
  fit.8 <- c(fit.8, encodeClassLabels(predict(nnet.model, data.matrix(training)[test.ind, location])))
  test.8 <- cbind(test.8, encodeClassLabels(predict(nnet.model, data.matrix(testing)[, location])))
}

fit.81 <- fit.8 - 1
test.81 <- apply(test.8, 1, function(x) {ifelse(mean(x == 2) > 0.5, 2, 1)}) - 1
mean(fit.81 != training_label[as.vector(sapply(1:10, function(i) which(cv.ind == i)))])
0.025

####Adaboost####
set.seed(123)
ada.mod <- ada(x = data.matrix(training)[, location], y = as.factor(training_label), loss = "logistic", type = "gentle")
mean(ada.mod$fit != training_label)


ada.pre <- predict(ada.mod, data.frame(data.matrix(testing)[, location]))
write.csv(ada.pre, "test_result.csv")



#### __________ LAsso selection####
#### 9. Weighted Logistic + penalty ####
table(training_label)
wei <- ifelse(training_label[as.vector(sapply(1:10, function(i) which(cv.ind == i)))] == 0, 1, 2)


fit.9 <- c()
test.9 <- c()
for (i in 1:10) {
  train.ind <- which(cv.ind != i); test.ind <- which(cv.ind == i)
  
  wei <- ifelse(training_label[train.ind] == 0, 1, 2)
  set.seed(123)
  regular1 <- cv.glmnet(x = data.matrix(training)[train.ind, ], y = training_label[train.ind], 
                        family = "binomial", alpha = 1, type.measure = "class", weights = wei)
  
  fit.9 <- c(fit.9, predict(regular1, data.matrix(training)[test.ind,], s = "lambda.min", type = "class"))
  test.9 <- cbind(test.9, predict(regular1, data.matrix(testing), s = "lambda.min", type = "class"))
}

fit.91 <- as.numeric(fit.9)
test.91 <- apply(test.9, 1, function(x) {ifelse(mean(as.numeric(x) == 1) > 0.5, 1, 0)})
mean(fit.91 != training_label[as.vector(sapply(1:10, function(i) which(cv.ind == i)))])
0.1016667

correla <- apply(training, 2, function(j) cor(j, training_label))
location2 <- as.numeric(which(abs(correla) > 0.15))
intersect(location, location2)

#### __________ Correlation selection####
#####10. Adaboost####
fit.10 <- c()
test.10 <- c()
er <- c()
for (i in 1:10) {
  train.ind <- which(cv.ind != i); test.ind <- which(cv.ind == i)
  
  set.seed(123)
  ada.mod <- ada(x = data.matrix(training)[train.ind, location2], y = training_label[train.ind],
                 loss = "exponential", type = "discrete")
  
  fit.10 <- c(fit.10, predict(ada.mod, data.frame(data.matrix(training)[test.ind, location2])))
  test.10 <- cbind(test.10, predict(ada.mod, data.frame(data.matrix(testing)[, location2])))
  er <- c(er, mean(fit.10 - 1 != training_label[test.ind]))
}

fit.101 <- fit.10 - 1
test.101 <- apply(test.10, 1, function(x) {ifelse(mean(x == 2) > 0.5, 2, 1)}) - 1
mean(fit.101 != training_label[as.vector(sapply(1:10, function(i) which(cv.ind == i)))])
0.1433333


#### *** PHASE 2####
#### Construct meta.train & meta.test####
sx <- as.vector(sapply(1:10, function(j) which(cv.ind == j)))
meta.train.label <- training_label[sx]

meta.train <- cbind(fit.11, fit.21, fit.31, fit.41, fit.51, fit.61, fit.71, fit.81, fit.91, fit.101)
rownames(meta.train) <- as.character(1:600)
dim(meta.train)

meta.test <- cbind(test.11, test.21, test.31, test.41, test.51, test.61, test.71, test.81, test.91, test.101)
colnames(meta.test) <- colnames(meta.train)
dim(meta.test)
#### 1.SVM #####
set.seed(12)
sub.lab1 <- sample(which(meta.train.label == 1), 206, replace = T)
sam.ind <- c(1:600, sub.lab1)
table(meta.train.label[sam.ind])
head(meta.train[sam.ind, 1:6], 100)



set.seed(123)
meta.cv.svm <- tune.svm(x = data.matrix(meta.train), y = as.factor(meta.train.label), type = "C-classification", kernel = "linear")

meta.cv.svm$best.performance
meta.cv.svm$best.model
optgamma <- meta.cv.svm$best.model$gamma
optcost <- meta.cv.svm$best.model$cost

meta.mod1 <- svm(x = data.matrix(meta.train[sam.ind, ]), y = as.factor(meta.train.label)[sam.ind], type = "C-classification", kernel = "linear", gamma = optgamma, cost = optcost)
mean(predict(meta.mod1, meta.train) != meta.train.label)
meta.pre1 <- predict(meta.mod1, meta.test) 
write.csv(meta.pre1, "test_result.csv")


####2.RandomForest####
set.seed(123)
meta.mod2 <- randomForest(x = meta.train, y = as.factor(meta.train.label), ntree = 1000)
meta.mod2
dim(meta.test)
mean(predict(meta.mod2, meta.train) != meta.train.label)
meta.pre2 <- predict(meta.mod2, meta.test) 
write.csv(meta.pre2, "test_result.csv")




####3.Xgboost ####
dtrain <- xgb.DMatrix(meta.train, label = meta.train.label)
parame <- list(max_depth = 5, objective = "binary:logistic", base_score = 0.5)
set.seed(1)
cv.xgb <- xgb.cv(params = parame, data = dtrain, nrounds = 400, eta = 0.03, nfold = 10, metrics = "error")

xgb.mod <- xgb.train(params = parame, data = dtrain, nrounds = 80)
mean(ifelse(predict(xgb.mod, meta.train) > 0.5, 1, 0) != meta.train.label)

meta.pre3 <- ifelse(predict(xgb.mod, meta.test) > 0.5, 1, 0)
write.csv(meta.pre3, "test_result.csv")


#### blending####
weigh1 <- rep(1/600, 600)
set.seed(123)
meta.mod2 <- randomForest(x = meta.train, y = as.factor(meta.train.label), ntree = 1000)
meta.mod2
dim(meta.test)
err1 <- sum(ifelse(predict(meta.mod2, meta.train) != meta.train.label, 1, 0) * weigh1)
alpha1 <- log((1 - err1)/err1)






####_____plot#####
library(VennDiagram)
venn.diagram(list(setlasso, setmcp, setscad, setcorr), category = c("LASSO", "MC+", "SCAD", "CORR"), filename = "selection.tiff", fill = c("red","green","blue", "grey"))
