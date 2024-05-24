if (!require("lmtest")) install.packages("lmtest", dependencies=TRUE); library(lmtest)
if (!require("ggplot2")) install.packages("ggplot2", dependencies=TRUE); library(ggplot2)
if (!require("gridExtra")) install.packages("gridExtra", dependencies=TRUE); library(gridExtra)
if (!require("ggcorrplot")) install.packages("ggcorrplot", dependencies=TRUE); library(ggcorrplot)
if (!require("class")) install.packages("class", dependencies=TRUE); library(class)
if (!require("kknn")) install.packages("kknn", dependencies=TRUE); library(kknn)
if (!require("leaps")) install.packages("leaps", dependencies=TRUE); library(leaps)
if (!require("glmnet")) install.packages("glmnet", dependencies=TRUE); library(glmnet)
if (!require("rpart")) install.packages("rpart", dependencies=TRUE); library(rpart)
if (!require("rpart.plot")) install.packages("rpart.plot", dependencies=TRUE); library(rpart.plot)
if (!require("randomForest")) install.packages("randomForest", dependencies=TRUE); library(randomForest)
if (!require("gbm")) install.packages("gbm", dependencies=TRUE); library(gbm)
if (!require("nnet")) install.packages("nnet", dependencies=TRUE); library(nnet)
if (!require("mboost")) install.packages("mboost", dependencies=TRUE); library(mboost)
if (!require("MASS")) install.packages("MASS", dependencies=TRUE); library(MASS)
if (!require("vip")) install.packages("vip", dependencies=TRUE); library("vip")

## Data
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
data<-read.csv2("data/winequality-white.csv")
data<-as.data.frame(lapply(data, as.numeric))

set.seed(123)

variables <- names(data)[-length(data)]
dependent <- names(data)[length(data)]

# Divide data into 3 samples

order<-sample(nrow(data),nrow(data))
train<-data[order[1:(as.integer(nrow(data)/2))],]
valid<-data[order[(as.integer(nrow(data)/2)):(as.integer(nrow(data)/1.333))],]
test<-data[order[(as.integer(nrow(data)/1.333)):nrow(data)],]

# Define dependent variable and model matrix
train_x<-train[,variables]
train_y<-train[,dependent]
valid_x<-valid[,variables]
valid_y<-valid[,dependent]
test_x<-test[,variables]
test_y<-test[,dependent]

#Non-linear transformations

lntrain<-log(train_x)
colnames(lntrain)<-paste0(colnames(train_x),sep = ".ln")
train2<-train_x^2
colnames(train2)<-paste0(colnames(train_x),sep = ".2")
exttrain<-abs(colMeans(train_x)-train_x)
colnames(exttrain)<-paste0(colnames(train_x),sep = ".ext")
exttrain2<-(colMeans(train_x)-train_x)^2
colnames(exttrain2)<-paste0(colnames(train_x),sep = ".ext2")
transforms_train<-cbind(lntrain,train2,exttrain,exttrain2)
transforms_train_s<-cbind(train2,exttrain)

lnvalid<-log(valid_x)
colnames(lnvalid)<-paste0(colnames(valid_x),sep = ".ln")
valid2<-valid_x^2
colnames(valid2)<-paste0(colnames(valid_x),sep = ".2")
extvalid<-abs(colMeans(valid_x)-valid_x)
colnames(extvalid)<-paste0(colnames(valid_x),sep = ".ext")
extvalid2<-(colMeans(valid_x)-valid_x)^2
colnames(extvalid2)<-paste0(colnames(valid_x),sep = ".ext2")
transforms_valid<-cbind(lnvalid,valid2,extvalid,extvalid2)
transforms_valid_s<-cbind(valid2,extvalid)

lntest<-log(test_x)
colnames(lntest)<-paste0(colnames(test_x),sep = ".ln")
test2<-test_x^2
colnames(test2)<-paste0(colnames(test_x),sep = ".2")
exttest<-abs(colMeans(test_x)-test_x)
colnames(exttest)<-paste0(colnames(test_x),sep = ".ext")
exttest2<-(colMeans(test_x)-test_x)^2
colnames(exttest2)<-paste0(colnames(test_x),sep = ".ext2")
transforms_test<-cbind(lntest,test2,exttest,exttest2)
transforms_test_s<-cbind(test2,exttest)

# Merging data  

full_data_train<-cbind(train,transforms_train)
full_data_valid<-cbind(valid,transforms_valid)
full_data_test<-cbind(test,transforms_test)
full_x_train<-as.matrix(cbind(train_x,transforms_train))
full_x_valid<-as.matrix(cbind(valid_x,transforms_valid))
full_x_test<-as.matrix(cbind(test_x,transforms_test))

form<-paste(dependent," ~ (", paste(colnames(train_x),collapse = " + "),")^2 + ",paste(colnames(transforms_train),collapse = " + ")) 
form_s<-paste(dependent," ~ (", paste(colnames(train_x),collapse = " + "),")^2 + ",paste(colnames(transforms_train_s),collapse = " + ")) 

base <- as.formula(paste(dependent,"~ ."))


Min <- apply(train[-ncol(train)], 2, min)
Max <- apply(train[-ncol(train)], 2, max)

train_nn_x <- sweep(sweep(train[-ncol(train)], 2, Min, "-"), 2, Max - Min, "/")
train_nn <- cbind(train_nn_x,quality = train_y)
valid_nn_x <- sweep(sweep(valid[-ncol(valid)], 2, Min, "-"), 2, Max - Min, "/")
valid_nn <- cbind(valid_nn_x, quality = valid_y)
test_nn_x <- sweep(sweep(test[-ncol(test)], 2, Min, "-"), 2, Max - Min, "/")
test_nn <- cbind(test_nn_x, quality = test_y)

## Binary case

data_b <- data

data_b$quality[data$quality<6] <- 0
data_b$quality[data$quality>5] <- 1

# Divide data into 3 samples

train_b<-data_b[order[1:(as.integer(nrow(data_b)/2))],]
valid_b<-data_b[order[(as.integer(nrow(data_b)/2)):(as.integer(nrow(data_b)/1.333))],]
test_b<-data_b[order[(as.integer(nrow(data_b)/1.333)):nrow(data_b)],]

# Define dependent variable and model matrix
train_y_b<-train_b[,dependent]
valid_y_b<-valid_b[,dependent]
test_y_b<-test_b[,dependent]

full_data_train_b<-cbind(train_b,transforms_train)
full_data_valid_b<-cbind(valid_b,transforms_valid)
full_data_test_b<-cbind(test_b,transforms_test)

form_b <- as.formula(paste("as.factor(",dependent,") ~ (", paste(colnames(train_x),collapse = " + "),")^2 + ",paste(colnames(transforms_train),collapse = " + "))) 
form_s_b <- as.formula(paste("as.factor(",dependent,") ~ (", paste(colnames(train_x),collapse = " + "),")^2 + ",paste(colnames(transforms_train_s),collapse = " + ")))
base_b <- as.formula(paste("as.factor(",dependent,") ~ .",sep=""))

train_nn_b <- cbind(train_nn_x,quality = train_y_b)
valid_nn_b <- cbind(valid_nn_x, quality = valid_y_b)
test_nn_b <- cbind(test_nn_x, quality = test_y_b)

rmse<-function(real,fitted){
  sqrt(mean((real-fitted)^2))
}

decision_b<-function(fito,fitv){
  res<-c()
  for (i in seq(0.001,1,0.001)) {
    v<-fitv>i
    r<-sum(abs(v-fitv))
    res[length(res)+1]<-r
  }
  vec<-fitv>which.min(res)/1000
  return(c(boundary = which.min(res)/1000,misc = sum(abs(vec-fito))/length(fito)))
}


## Continuous models

# Base fit
base_fit<-lm(base,data=train)

# Best subset
best_subset<-leaps(train_x,train_y,nbest=1,method=c("adjr2"))
best<-which.max(best_subset$adjr2)
regressors<-best_subset$which[best,]
bss_data<-cbind(quality=train_y,train_x[,regressors])

bss_fit<-lm(base,data = bss_data)

# Knn

w_knn_model <- train.kknn(base, data = train, kmax = 30, scale = TRUE)
knn_model <- kknn(base, train = train, test = test, k = w_knn_model$best.parameters$k , scale = TRUE)

# Full fit - not included 
full_fit<-lm(as.formula(form_s), data = full_data_train)

# AIC-BIC

AIC_step<-step(full_fit)
aic_fit<-lm(formula(AIC_step),data = full_data_train)

BIC_step<-step(full_fit,k=log(nrow(train)))
bic_fit<-lm(formula(BIC_step),data = full_data_train)

# Ridge

cv.ridge <- cv.glmnet(x=full_x_train,y=train_y, alpha = 0)
bestlam_ridge <- cv.ridge$lambda.min

ridge<-glmnet(x=full_x_train,y=train_y, alpha = 0, lambda = bestlam_ridge)

# Lasso

cv.lasso <- cv.glmnet(x=full_x_train,y=train_y, alpha = 1)
bestlam_lasso <- cv.lasso$lambda.min

lasso<-glmnet(x=full_x_train,y=train_y, alpha = 1,lambda = bestlam_lasso)

#Elastic net

e_net<-function(xt,yt,xv,yv,s){
  a_range<-seq(0,1,s)
  result<-numeric(length(a_range))
  for (i in 1:length(a_range)){
    cv.fit<-cv.glmnet(x=xt,y=yt, alpha = a_range[i])
    bestlam <- cv.fit$lambda.min
    enet<-glmnet(x=xt,y=yt, alpha = a_range[i],lambda = bestlam)
    result[i]<-rmse(yv,as.numeric(predict(enet, newx = xv)))
  }
  return(list(rmse = result,best = a_range[which.min(result)]))
}

enet_opt<-e_net(full_x_train,train_y,full_x_valid,valid_y,0.01)
bestalp<-enet_opt$best
cv.enet<-cv.glmnet(x=full_x_train,y=train_y, alpha = bestalp)
bestlam_enet <- cv.enet$lambda.min

enet<-glmnet(x=full_x_train,y=train_y, alpha = bestalp,lambda = bestlam_enet)

# Tree

tree<-rpart(base, data = train, method = "anova", control = list(cp = 10^(-10)))
mintree<-which.min(tree$cptable[, "xerror"])
select <- which(tree$cptable[, "xerror"] < sum(tree$cptable[mintree, c("xerror", "xstd")]))[1]
p.tree <- prune(tree, cp = tree$cptable[select, "CP"])

## Bagging

bag_tree <- function(a,train_x,train_y,valid_x,valid_y){
  set.seed(a)
  bag <- sample(nrow(train_x),round(nrow(train_x)/3),replace = TRUE)
  sub_x <- train_x[bag,]
  sub_y <- train_y[bag]
  
  tree<-rpart(base, data = train, method = "anova", control = list(cp = 10^(-10)))
  mintree<-which.min(tree$cptable[, "xerror"])
  select <- which(tree$cptable[, "xerror"] < sum(tree$cptable[mintree, c("xerror", "xstd")]))[1]
  p.tree <- prune(tree, cp = tree$cptable[select, "CP"])
  return(max(p.tree$cptable[,"nsplit"]))
}

vec <- sapply(1:100, bag_tree, train_x = train_x,train_y = train_y,valid_x = valid_x,valid_y = valid_y)
n_most <- as.numeric(names(sort(table(vec), decreasing = TRUE)[1]))
n_avg <- round(mean(vec))

# Most frequent size

select_most <- which.min(abs(tree$cptable[,"nsplit"]-n_most))
tree_n_most <- prune(tree, cp = tree$cptable[select_most, "CP"])

# Average size

select_avg <- which.min(abs(tree$cptable[,"nsplit"]-n_avg))
tree_n_avg <- prune(tree, cp = tree$cptable[select_avg, "CP"])

# Random forest

rand_for <- randomForest(base, data = train, importance = TRUE, ntree = 1000, mtry=3)

rand_for_split <- randomForest(base, data = train, importance = TRUE, ntree = 1000, mtry=5)
rand_for_split_num_vip <- vip(rand_for_split, aesthetics=list(color="black", fill="#D3D3D3", lwd=0.2))

rand_for_rest <- randomForest(base, data = train, importance = TRUE, ntree = 1000, mtry=3, maxnodes=20)

# Boosting

gbm_model <- gbm(base, data = train, distribution = "gaussian", n.trees = 5000, shrinkage = 0.001, cv.folds = 10, interaction.depth = 2)

opt_boost <- function(m,train,valid){
  gbm_model <- gbm(base, data = train, distribution = "gaussian", n.trees = m, shrinkage = 0.001, cv.folds = 10, interaction.depth = 2)
  rmse(valid_y,predict(gbm_model,newdata = valid))
}

best_boost <- optimize(f = opt_boost, lower = 1000, upper = 10000, tol = 1, valid=valid, train=train)

gbm_model_best <- gbm(base, data = train, distribution = "gaussian", n.trees = best_boost$minimum, shrinkage = 0.001, cv.folds = 10, interaction.depth = 2)

# Neural network

lambdas <- seq(0, 0.4, by = 0.01)

MCRs <- lapply(lambdas, function(lambda) {
  fit_nn <- nnet::nnet(base, data = train_nn ,size = 10, decay = lambda, trace = 1, skip = TRUE, maxit = 1000,linout=TRUE)
  c(train = rmse(predict(fit_nn, train_nn), train_nn[,dependent]), valid = rmse(predict(fit_nn, valid_nn), valid_nn[,dependent]))
})
MCRs <- do.call("rbind", MCRs)

lambda <- lambdas[which.min(MCRs[, "valid"])]
nn_model <- nnet::nnet(base, data = train_nn, size = 10, trace = 0, decay = lambda, skip = TRUE, linout=TRUE, maxit = 1000)

# Comparison
results_num_response <- as.data.frame(rbind(rmse(test_y,predict(base_fit,newdata = test)),
                                            rmse(test_y,predict(bss_fit,newdata = test)),
                                            rmse(test_y,predict(knn_model, newdata = test)),
                                            rmse(test_y,predict(aic_fit,newdata = full_data_test)),
                                            rmse(test_y,predict(bic_fit,newdata = full_data_test)),
                                            rmse(test_y,as.numeric(predict(ridge , newx = full_x_test))),
                                            rmse(test_y,as.numeric(predict(lasso , newx = full_x_test))),
                                            rmse(test_y,as.numeric(predict(enet , newx = full_x_test))),
                                            rmse(test_y,predict(p.tree, newdata = test, type = "vector")),
                                            rmse(test_y,predict(tree_n_most, newdata = test, type = "vector")),
                                            rmse(test_y,predict(tree_n_avg, newdata = test, type = "vector")),
                                            rmse(test_y,predict(rand_for, newdata = test)),
                                            rmse(test_y,predict(rand_for_split, newdata = test)),
                                            rmse(test_y,predict(rand_for_rest, newdata = test)),
                                            rmse(test_y,predict(gbm_model, newdata = test)),
                                            rmse(test_y,predict(gbm_model_best, newdata = test)),
                                            rmse(test_nn$quality, predict(nn_model, test_nn))
))
results_num_response <- cbind(c("Linear","Best Subset Selection","K-nearest neighbor",
                                "Step AIC","Step BIC","Ridge","Lasso","Elastic net","Classification tree",
                                "Classification tree (frequency size)", "Classification tree (average size)", 
                                "Random forest (mtry=3)","Random forest (mtry=5)",
                                "Random forest (mtry=3, maxnodes=20)","Gradient Boosting",
                                "Optimal Size Gradient Boosting","Neural network"), results_num_response)
colnames(results_num_response) <- c("Model","RMSE")
#save(results_num_response, file = "results_num_response.RData")

## Categorical

# Base fit
# Logit
base_fit_logit_c <- polr(base_b, data=train, method = "logistic")
# Probit
base_fit_probit_c <- polr(base_b, data=train, method = "probit")

# Knn
bag_knn <- function(a,train_x,train_y,valid_x,valid_y){
  set.seed(a)
  bag <- sample(nrow(train_x),round(nrow(train_x)/3),replace = TRUE)
  sub_x <- train_x[bag,]
  sub_y <- train_y[bag]
  
  result<-numeric(30)
  for (i in 1:30){
    knn_model <- knn(train = sub_x, test = valid_x, cl = sub_y, k = i)
    result[i] <- mean(valid_y != knn_model)
  }
  which.min(result)
}

vec_c <- sapply(1:100, bag_knn, train_x = train_x, train_y = train_y, valid_x = valid_x, valid_y = valid_y)
k_most_c <- as.numeric(names(sort(table(vec_c), decreasing = TRUE)[1]))
k_avg_c <- round(mean(vec_c))

# Most frequent size
knn_model_most_c <- knn(train = train_x, test = test_x, cl =train_y, k = k_most_c)

# Average size
knn_model_avg_c <- knn(train = train_x, test = test_x, cl =train_y, k = k_avg_c)

# Full-fit not included
full_fit_c <- polr(form_s_b,data=full_data_train,method = "probit")

# AIC
AIC_step_c <- step(full_fit_c, direction = "forward")
aic_fit_c <- polr(formula(AIC_step_c),data = full_data_train,method = "probit")

# BIC
BIC_step_c <- step(full_fit_c,k=log(nrow(train)), direction = "forward")
bic_fit_c <- polr(formula(BIC_step_c),data = full_data_train,method = "probit")

# Ridge
cv.ridge_c <- cv.glmnet(x=full_x_train,y=train_y, alpha = 0, family="multinomial")
bestlam_ridge_c <- cv.ridge_c$lambda.min

ridge_c <- glmnet(x=full_x_train,y=train_y, alpha = 0, lambda = bestlam_ridge_c,family="multinomial")

# Tree

tree_c <- rpart(base_b, data = train, method = "class", control = list(cp = 10^(-10)))
mintree_c<-which.min(tree_c$cptable[, "xerror"])
select_c <- which(tree_c$cptable[, "xerror"] < sum(tree_c$cptable[mintree_c, c("xerror", "xstd")]))[1]

p.tree_c <- prune(tree_c, cp = tree_c$cptable[select_c, "CP"])

# Bagging

bag_tree <- function(a,train){
  set.seed(a)
  bag <- sample(nrow(train_x),round(nrow(train_x)/3),replace = TRUE)
  sub_x <- train_x[bag,]
  sub_y <- train_y[bag]
  
  tree<-rpart(base, data = train, method = "class", control = list(cp = 10^(-10)))
  mintree<-which.min(tree$cptable[, "xerror"])
  select <- which(tree$cptable[, "xerror"] < sum(tree$cptable[mintree, c("xerror", "xstd")]))[1]
  p.tree <- prune(tree, cp = tree$cptable[select, "CP"])
  return(max(p.tree$cptable[,"nsplit"]))
}

vec_tree_c <- sapply(1:100, bag_tree, train = train)
n_most_c <- as.numeric(names(sort(table(vec_tree_c), decreasing = TRUE)[1]))
n_avg_c <- round(mean(vec_tree_c))

select_tree_most_c <- which.min(abs(tree_c$cptable[,"nsplit"]-n_most_c))
tree_n_most_c <- prune(tree_c, cp = tree_c$cptable[select_tree_most_c, "CP"])

select_tree_avg_c <- which.min(abs(tree_c$cptable[,"nsplit"]-n_avg_c))
tree_n_avg_c <- prune(tree_c, cp = tree_c$cptable[select_tree_avg_c, "CP"])

# Random Forest

rand_for_c <- randomForest(base_b, data = train, importance = TRUE, ntree = 1000, mtry=3)

rand_for_split_c <- randomForest(base_b, data = train, importance = TRUE, ntree = 1000, mtry=5)
rand_for_split_cat_vip <- vip(rand_for_split_c, aesthetics=list(color="black", fill="#D3D3D3", lwd=0.2))

rand_for_rest_c <- randomForest(base_b, data = train, importance = TRUE, ntree = 1000, mtry=3, maxnodes=20)

#Neural network

MCRs_c <- lapply(lambdas, function(lambda) {
  fit_nn <- nnet::nnet(base_b, data = train_nn ,size = 10, decay = lambda, trace = 1, skip = TRUE, maxit = 1000)
  c(train = mean(predict(fit_nn, train_nn, type = "class") != train_nn[,dependent]), valid = mean(predict(fit_nn, valid_nn, type = "class") != valid_nn[,dependent]))
})
MCRs_c <- do.call("rbind", MCRs_c)

lambda_c <- lambdas[which.min(MCRs_c[, "valid"])]
nn_model_c <- nnet::nnet(base_b, data = train_nn, size = 10, trace = 0, decay = lambda_c, skip = TRUE, maxit = 1000)

# Comparison
results_cat_response <- as.data.frame(rbind(rmse(test_y,as.numeric(predict(base_fit_logit_c,newdata = test))),
                                            rmse(test_y,as.numeric(predict(base_fit_probit_c,newdata = test))),
                                            rmse(test_y,as.numeric(knn_model_most_c)),
                                            rmse(test_y,as.numeric(knn_model_avg_c)),
                                            rmse(test_y,as.numeric(predict(aic_fit_c,newdata = full_data_test))),
                                            rmse(test_y,as.numeric(predict(bic_fit_c,newdata = full_data_test))),
                                            rmse(test_y,as.numeric(predict(ridge_c, newx = full_x_test))),
                                            #                                            rmse(test_y,as.numeric(predict(enet_c, newx = full_x_test))),
                                            rmse(test_y,as.numeric(predict(p.tree_c, newdata = test, type = "vector"))),
                                            rmse(test_y,as.numeric(predict(tree_n_most_c, newdata = test, type = "vector"))),
                                            rmse(test_y,as.numeric(predict(tree_n_avg_c, newdata = test, type = "vector"))),
                                            rmse(test_y,as.numeric(predict(rand_for_c, newdata = test))),
                                            rmse(test_y,as.numeric(predict(rand_for_split_c, newdata = test))),
                                            rmse(test_y,as.numeric(predict(rand_for_rest_c, newdata = test))),
                                            rmse(test_nn$quality, as.numeric(predict(nn_model_c, test_nn)))
))
results_cat_response <- cbind(c("Logit","Probit",
                                "K-nearest neighbor (frequency size)","K-nearest neighbor (average size)",
                                "Step AIC","Step BIC","Ridge","Classification tree",
                                "Classification tree (frequency size)", "Classification tree (average size)", 
                                "Random forest (mtry=3)","Random forest (mtry=5)",
                                "Random forest (mtry=3, maxnodes=20)","Neural network"), results_cat_response)
colnames(results_cat_response) <- c("Model","RMSE")
#save(results_cat_response, file = "results_cat_response.RData")

## Binary

# Base model
base_fit_logit_b<-glm(base_b, family = binomial(link = "logit"), data=train_b)

base_fit_probit_b<-glm(base_b, family = binomial(link = "probit"), data=train_b)

# Best Subset

best_subset_b<-leaps(train_x,train_y_b,nbest=1,method=c("adjr2"))
best_b<-which.max(best_subset_b$adjr2)
regressors_b<-best_subset_b$which[best_b,]
bss_data_b<-cbind(quality=train_y_b,train_x[,regressors_b])

bss_fit_b<-glm(base_b,family = binomial(link = "probit"),data = bss_data_b)

# Knn

w_knn_model_b <- train.kknn(base_b, data = train_b, kmax = 30, scale = TRUE)
knn_model_b <- kknn(base_b, train = train_b, test = test_b, k = w_knn_model_b$best.parameters$k , scale = TRUE)

vec_b <- sapply(1:100, bag_knn, train_x = train_x, train_y = train_y_b, valid_x = valid_x, valid_y = valid_y_b)
k_most_b <- as.numeric(names(sort(table(vec_b), decreasing = TRUE)[1]))
k_avg_b <- round(mean(vec_b))

# Highest Frequency

knn_model_most_b <- knn(train = train_x, test = test_x, cl =train_y_b, k = k_most_b)

# Average size

knn_model_avg_b <- knn(train = train_x, test = test_x, cl =train_y_b, k = k_avg_b)

# Full fit - not included

full_fit_b<-glm(form_s_b,data=full_data_train_b,family = binomial(link = "probit"))

#AIC

AIC_step_b<-step(full_fit_b,direction = "forward")
aic_fit_b<-glm(formula(AIC_step_b),data = full_data_train_b,family = binomial(link = "probit"))

#BIC

BIC_step_b<-step(full_fit_b,k=log(nrow(train_b)),direction = "forward")
bic_fit_b<-glm(formula(BIC_step_b),data = full_data_train_b,family = binomial(link = "probit"))

# Ridge

cv.ridge_b <- cv.glmnet(x=full_x_train,y=train_y_b, alpha = 0, family="binomial")
bestlam_ridge_b <- cv.ridge_b$lambda.min

ridge_b<-glmnet(x=full_x_train,y=train_y_b, alpha = 0, lambda = bestlam_ridge_b,family="binomial")

# Lasso

cv.lasso_b <- cv.glmnet(x=full_x_train,y=train_y_b, alpha = 1, family="binomial")
bestlam_lasso_b <- cv.lasso_b$lambda.min

lasso_b<-glmnet(x=full_x_train,y=train_y_b, alpha = 1,lambda = bestlam_lasso_b, family="binomial")

# Elastic net

e_net_b<-function(xt,yt,xv,yv,s){
  a_range<-seq(0,1,s)
  result<-numeric(length(a_range))
  for (i in 1:length(a_range)){
    cv.fit<-cv.glmnet(x=xt,y=yt, alpha = a_range[i],family="binomial")
    bestlam <- cv.fit$lambda.min
    enet<-glmnet(x=xt,y=yt, alpha = a_range[i],lambda = bestlam,family="binomial")
    result[i]<-decision_b(yv,as.numeric(predict(enet, newx = xv)))[2]
  }
  return(list(misc = result,best = a_range[which.min(result)]))
}

enet_opt_b<-e_net_b(full_x_train,train_y_b,full_x_valid,valid_y_b,0.01)
bestalp_b<-enet_opt_b$best

cv.enet_b<-cv.glmnet(x=full_x_train,y=train_y_b, alpha = bestalp_b ,family="binomial")
bestlam_enet_b <- cv.enet_b$lambda.min
enet_b<-glmnet(x=full_x_train,y=train_y_b, alpha = bestalp_b,lambda = bestlam_enet_b,family="binomial")

# Tree

tree_b<-rpart(base_b, data = train_b, method = "class", control = list(cp = 10^(-10)))
mintree_b<-which.min(tree_b$cptable[, "xerror"])
select_tree_b <- which(tree_b$cptable[, "xerror"] < sum(tree_b$cptable[mintree_b, c("xerror", "xstd")]))[1]
p.tree_b <- prune(tree_b, cp = tree_b$cptable[select_tree_b, "CP"])

# Bagging tree

vec_b <- sapply(1:100, bag_tree, train = train_b)
n_most_b <- as.numeric(names(sort(table(vec_b), decreasing = TRUE)[1]))
n_avg_b <- round(mean(vec_b))

select_most_b <- which.min(abs(tree_b$cptable[,"nsplit"]-n_most_b))
tree_n_most_b <- prune(tree_b, cp = tree_b$cptable[select_most_b, "CP"])


select_avg_b <- which.min(abs(tree_b$cptable[,"nsplit"]-n_avg_b))
tree_n_avg_b <- prune(tree_b, cp = tree_b$cptable[select_avg_b, "CP"])

# Random Forest

rand_for_b <- randomForest(base_b, data = train_b, importance = TRUE, ntree = 1000, mtry=3)

rand_for_split_b <- randomForest(base_b, data = train_b, importance = TRUE, ntree = 1000, mtry=5)
rand_for_split_bin_vip <- vip(rand_for_split_b, aesthetics=list(color="black", fill="#D3D3D3", lwd=0.2))

rand_for_rest_b <- randomForest(base_b, data = train_b, importance = TRUE, ntree = 1000, mtry=3, maxnodes=20)

# Neural network

MCRs_b <- lapply(lambdas, function(lambda) {
  fit_nn <- nnet::nnet(base_b, data = train_nn ,size = 10, decay = lambda, trace = 1, skip = TRUE, maxit = 1000)
  c(train = mean(predict(fit_nn, train_nn_b, type = "class") != train_nn_b[,dependent]), valid = mean(predict(fit_nn, valid_nn_b, type = "class") != valid_nn_b[,dependent]))
})
MCRs_b <- do.call("rbind", MCRs_b)

lambda_b <- lambdas[which.min(MCRs_b[, "valid"])]
nn_model_b <- nnet::nnet(base_b, data = train_nn_b, size = 10, trace = 0, decay = lambda_b, skip = TRUE, maxit = 1000)

# Comparison

results_bin_response <- as.data.frame(rbind(rmse(test_y,as.numeric(predict(base_fit_logit_b,newdata = test))),
                                            rmse(test_y,as.numeric(predict(base_fit_probit_b,newdata = test))),
                                            rmse(test_y,as.numeric(predict(bss_fit_b,newdata = test))),
                                            rmse(test_y,as.numeric(predict(knn_model_b))),
                                            rmse(test_y,as.numeric(knn_model_most_b)),
                                            rmse(test_y,as.numeric(knn_model_avg_b)),
                                            rmse(test_y,as.numeric(predict(aic_fit_b,newdata = full_data_test))),
                                            rmse(test_y,as.numeric(predict(bic_fit_b,newdata = full_data_test))),
                                            rmse(test_y,as.numeric(predict(ridge_b, newx = full_x_test))),
                                            rmse(test_y,as.numeric(predict(lasso_b, newx = full_x_test))),
                                            rmse(test_y,as.numeric(predict(enet_b, newx = full_x_test))),
                                            rmse(test_y,as.numeric(predict(p.tree_b, newdata = test, type = "vector"))),
                                            rmse(test_y,as.numeric(predict(tree_n_most_b, newdata = test, type = "vector"))),
                                            rmse(test_y,as.numeric(predict(tree_n_avg_b, newdata = test, type = "vector"))),
                                            rmse(test_y,as.numeric(predict(rand_for_b, newdata = test))),
                                            rmse(test_y,as.numeric(predict(rand_for_split_b, newdata = test))),
                                            rmse(test_y,as.numeric(predict(rand_for_rest_b, newdata = test))),
                                            rmse(test_nn$quality, as.numeric(predict(nn_model_b, test_nn)))
))
results_bin_response <- cbind(c("Logit","Probit","Best Subset Selection","K-nearest neighbor",
                                "K-nearest neighbor (frequency size)","K-nearest neighbor (average size)",
                                "Step AIC","Step BIC","Ridge","Lasso","Elastic net","Classification tree",
                                "Classification tree (frequency size)", "Classification tree (average size)", 
                                "Random forest (mtry=3)","Random forest (mtry=5)",
                                "Random forest (mtry=3, maxnodes=20)","Neural network"), results_bin_response)
colnames(results_bin_response) <- c("Model","RMSE")
#save(results_bin_response, file = "results_bin_response.RData")


## Correlation of variables

## Corr plot
corr <- round(cor(data), 2)

ggcorrplot(corr, hc.order = TRUE, type = "lower",lab = TRUE,lab_size = 2,
           outline.col = "white",
           ggtheme = ggplot2::theme_gray,
           colors = c("#6D9EC1", "white", "#E46726"))

## Variables with respect to dependent (categorical)

plot_averages <- function(data, outcome, variable) {
  data <- data[!is.na(data[[variable]]), ]
  
  means<-aggregate(data[[variable]],list(data[[outcome]]),FUN=mean)$x
  lower<-min(means)*0.95
  upper<-max(means)*1.05
  
  # Create a plot using ggplot2 with y-axis limits
  ggplot(data, aes(x = .data[[outcome]], y = .data[[variable]])) +
    geom_bar(stat = "summary", fun = "mean") +
    labs(x = "Category", y = paste("Average", variable)) +
    ggtitle(paste("Average", variable, "by Category")) +
    coord_cartesian(ylim=(c(lower, upper)))
}


## Variables with respect to dependent (categorical)

plot_averages <- function(data, outcome, variable) {
  data <- data[!is.na(data[[variable]]), ]
  
  means<-aggregate(data[[variable]],list(data[[outcome]]),FUN=mean)$x
  lower<-min(means)*0.95
  upper<-max(means)*1.05
  
  # Create a plot using ggplot2 with y-axis limits
  ggplot(data, aes(x = .data[[outcome]], y = .data[[variable]])) +
    geom_bar(stat = "summary", fun = "mean") +
    labs(x = "Category", y = paste("Average", variable)) +
    ggtitle(paste("Average", variable, "by Category")) +
    coord_cartesian(ylim=(c(lower, upper)))
}

plot_list <- lapply(variables, function(var) plot_averages(data, dependent, var))
plot_list

grid.arrange(grobs = plot_list)


plot_list <- lapply(variables, function(var) plot_averages(data_b, dependent, var))
plot_list

grid.arrange(grobs = plot_list)


## Continuous RMSE

# Base Fit
rmse(test_y,predict(base_fit,newdata = test))

# Best Subset Selection
rmse(test_y,predict(bss_fit,newdata = test))

# K-nearest neighbor
rmse(test_y,predict(knn_model))

# Step AIC
rmse(test_y,predict(aic_fit,newdata = full_data_test))

# Step BIC
rmse(test_y,predict(bic_fit,newdata = full_data_test))

# Ridge
rmse(test_y,as.numeric(predict(ridge , newx = full_x_test)))

# Lasso
rmse(test_y,as.numeric(predict(lasso , newx = full_x_test)))

#Elastic net
rmse(test_y,as.numeric(predict(enet , newx = full_x_test)))

# Optimized tree

rmse(test_y,predict(p.tree, newdata = test, type = "vector"))
rmse(test_y,predict(tree_n_most, newdata = test, type = "vector"))
rmse(test_y,predict(tree_n_avg, newdata = test, type = "vector"))

# Random forest

rmse(test_y,predict(rand_for, newdata = test))
rmse(test_y,predict(rand_for_split, newdata = test))
rmse(test_y,predict(rand_for_rest, newdata = test))

# Boosting

rmse(test_y,predict(gbm_model, newdata = test))
rmse(test_y,predict(gbm_model_best, newdata = test))

# Neural network

rmse(test_y, predict(nn_model, test_nn))


## Categorical RMSE

# Base fit
mean(test_y != predict(base_fit_logit_c,newdata = test))
mean(test_y != predict(base_fit_probit_c,newdata = test))

# K-nearest neighbor
mean(test_y!=knn_model_most_c)
mean(test_y!=knn_model_avg_c)

# Step AIC
mean(test_y!=predict(aic_fit_c,newdata = full_data_test))

# Step BIC
mean(test_y!=predict(bic_fit_c,newdata = full_data_test))

# Ridge
mean(test_y!=as.numeric(predict(ridge_c , newx = full_x_test, type="class")))

# Optimized tree

mean(test_y!=predict(p.tree_c, newdata = test, type = "class"))
mean(test_y!=predict(tree_n_most_c, newdata = test, type = "class"))
mean(test_y!=predict(tree_n_avg_c, newdata = test, type = "class"))

# Random forest

mean(test_y!=predict(rand_for_c, newdata = test))
mean(test_y!=predict(rand_for_split_c, newdata = test))
mean(test_y!=predict(rand_for_rest_c, newdata = test))

# Neural network

mean(predict(nn_model_c, test_nn, type = "class") != test_y)


## Binary RMSE

# Base fit
mean(test_y_b != ifelse(predict(base_fit_logit_b,newdata = test_b,type="response")<0.5,0,1))
mean(test_y_b != ifelse(predict(base_fit_probit_b,newdata = test_b,type="response")<0.5,0,1))

# Best Subset Selection
mean(test_y_b != ifelse(predict(bss_fit_b,newdata = test_b,type="response") < 0.5,0,1))

# K-nearest neighbor
mean(test_y_b != predict(knn_model_b))

mean(test_y_b != knn_model_most_b)

mean(test_y_b != knn_model_avg_b)

# Step AIC
mean(test_y_b != ifelse(predict(aic_fit_b,newdata = full_data_test,type="response") < 0.5,0,1))

# Step BIC
mean(test_y_b != ifelse(predict(bic_fit_b,newdata = full_data_test,type="response") < 0.5,0,1))

# Ridge
mean(test_y_b != ifelse(as.numeric(predict(ridge_b , newx = full_x_test,type="response"))< 0.5,0,1))

# Lasso
mean(test_y_b != ifelse(as.numeric(predict(lasso_b , newx = full_x_test,type="response"))< 0.5,0,1))

#Elastic net
mean(test_y_b != ifelse(as.numeric(predict(enet_b , newx = full_x_test,type="response"))< 0.5,0,1))

# Optimized tree

mean(test_y_b != predict(p.tree_b, newdata = test, type = "class"))
mean(test_y_b != predict(tree_n_most_b, newdata = test, type = "class"))
mean(test_y_b != predict(tree_n_avg_b, newdata = test, type = "class"))

# Random forest

mean(test_y_b != predict(rand_for_b, newdata = test_b))
mean(test_y_b != predict(rand_for_split_b, newdata = test_b))
mean(test_y_b != predict(rand_for_rest_b, newdata = test_b))

# Neural network

mean(test_y_b != predict(nn_model_b, test_nn_b, type = "class"))

# Comparison of the models

# Comparison
png(file="out/rmse_res.png",width=8, height=4, units="in", res=600, pointsize=1)
op <-par(mfcol=c(1,3), mar=c(30,130,25,10), mgp=c(22,10,0), cex.axis=8, cex.lab=10, cex.main=13, xpd=TRUE)
plot(results_num_response$RMSE, 1:nrow(results_num_response), pch=19, col="dodgerblue", axes=FALSE, 
     xlim=c(0.6,0.8), xlab="RMSE", ylab="")
axis(side=1, at=seq(0.6,0.8,0.05), labels=seq(0.6,0.8,0.05), lwd=0.3)
axis(side=2, at=1:nrow(results_num_response), labels=results_num_response$Model, las=2, lwd=0.3, cex=5, tck=1, lty=2)
points(results_num_response$RMSE, 1:nrow(results_num_response),pch=19, col="dodgerblue", cex=12)
legend("top",legend = c("Numerical response variable"), pch=19, box.lwd=0, cex=13, ncol=2,
       inset=c(0,-0.08), col=c("dodgerblue"), bg="transparent")
box()

plot(results_cat_response$RMSE, 1:nrow(results_cat_response), pch=19, col="forestgreen", axes=FALSE, 
     xlim=c(2,6.5), xlab="RMSE", ylab="")
axis(side=1, at=seq(2,6.5,0.05), labels=seq(2,6.5,0.05), lwd=0.3)
axis(side=2, at=1:nrow(results_cat_response), labels=results_cat_response$Model, las=2, lwd=0.3, cex=5, tck=1, lty=2)
points(results_cat_response$RMSE, 1:nrow(results_cat_response),pch=19, col="forestgreen", cex=12)
legend("top",legend = c("Categorical response variable"), pch=19, box.lwd=0, cex=13, ncol=2,
       inset=c(0,-0.08), col=c("forestgreen"), bg="transparent")
box()

plot(results_bin_response$RMSE, 1:nrow(results_bin_response), pch=19, col="darkorange", axes=FALSE, 
     xlim=c(4,5.5), xlab="RMSE", ylab="")
axis(side=1, at=seq(4,5.5,0.05), labels=seq(4,5.5,0.05), lwd=0.3)
axis(side=2, at=1:nrow(results_bin_response), labels=results_bin_response$Model, las=2, lwd=0.3, cex=5, tck=1, lty=2)
points(results_bin_response$RMSE, 1:nrow(results_bin_response),pch=19, col="darkorange", cex=12)
legend("top",legend = c("Binary response variable"), pch=19, box.lwd=0, cex=13, ncol=2,
       inset=c(0,-0.08), col=c("darkorange"), bg="transparent")
box()
par(op)
dev.off()

#VIP Random Forest

png(file="out/varimp_rand_forests.png",width=8, height=4, units="in", res=600)
grid.arrange(
  (rand_for_split_num_vip + 
     ggtitle("Numerical response") + theme_bw(base_size=10) + 
     theme(panel.border = element_blank(), axis.text=element_text(colour="black"),
           panel.grid.minor=element_blank(), axis.line=element_line(color="black"))),
  (rand_for_split_cat_vip + 
     ggtitle("Categorical response") + theme_bw(base_size=10) + 
     theme(panel.border = element_blank(), axis.text=element_text(colour="black"),
           panel.grid.minor=element_blank(), axis.line=element_line(color="black"))),
  (rand_for_split_bin_vip + 
     ggtitle("Binary response") + theme_bw(base_size=10) + 
     theme(panel.border = element_blank(), axis.text=element_text(colour="black"),
           panel.grid.minor=element_blank(), axis.line=element_line(color="black"))),
  ncol=3, widths=c(1,1,1))
dev.off()

