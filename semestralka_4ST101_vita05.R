#Semestrální práce 4ST101
library(caret)
library(dplyr)
library(ggplot2)
library(caTools)
library(ROCR)
library(VIM)
library(Rcpp)
library(pROC)
library(ROSE)
library(performanceEstimation)
library(e1071)
library(ipred)
library(randomForest)
library(datasets)
library(rlang)
setwd("C:/Users/vitko/Documents/rHomework")
data <-  read.csv2(file = 'stroke_data.csv',
                   header = T, 
                   sep = ",", 
                   quote = "\"",
                   dec = ".",
                   fill = T,
                   comment.char = "",
                   encoding = "unknown")
data
summary(data)
  #Factorization of categorical vars
data <- mutate_at(data, vars("gender", "hypertension", "heart_disease", 
                             "ever_married", "work_type", "Residence_type", 
                             "smoking_status", "stroke"), as.factor)
#transform bmi to numeric
data$bmi <- as.double(data$bmi)
str(data)

#check for NaNs
colSums(!is.na(data)) == nrow(data)

#Data pre-processing
remove_outliers <- function(x, na.rm = TRUE, ...) {
  qnt <- quantile(x, probs=c(.25, .75), na.rm = na.rm, ...)
  H <- 1.5 * IQR(x, na.rm = na.rm)
  y <- x
  y[x < (qnt[1] - H)] <- NA
  y[x > (qnt[2] + H)] <- NA
  y
}

#initial distribution of values
#most of tracked patients are over 30 y.o, a steady decrease from 50 y.o
ggplot(data, aes(x=age)) + geom_density() +
  ggtitle("Distribution of age") + xlab("Age")
#leptokurtic asymmetric,  most patients have normal glucose level
ggplot(data, aes(x=avg_glucose_level)) + geom_density() + 
  ggtitle("Distribution of average glucose level") + xlab("Average glucose level")
#distr close to normal
ggplot(subset(data, !is.na(bmi)), aes(x=bmi)) + geom_density() + 
  ggtitle("Distribution of bmi")
#2 groups of patients
glucose_level_by_bmi <- ggplot(data, aes(x=bmi, y=avg_glucose_level, colour=age)) +
  geom_jitter(alpha=1/2, na.rm=T) + ggtitle("Glucose level by bmi") + ylab("Average glucose level")
glucose_level_by_bmi
#Remove outliers from numeric vars
#drop outliers from age
ggplot(data, aes(x=age)) + geom_boxplot() + xlab("Age")
quantile(data$age, probs = c(.25, .75), na.rm = T)
ggplot(data, aes(x=remove_outliers(data$age))) + geom_boxplot() + xlab("Age without outliers")
#drop outliers from bmi
ggplot(data, aes(x=bmi)) + geom_boxplot()
quantile(data$bmi, probs = c(.25, .75), na.rm = T)
ggplot(data, aes(x=remove_outliers(data$bmi))) + geom_boxplot() + xlab("BMI without outliers")
#drop outliers from avg_glucose_level
ggplot(data, aes(x=avg_glucose_level)) + geom_boxplot() + xlab("Average glucose level")
quantile(data$avg_glucose_level, probs = c(.25, .75), na.rm = T)
ggplot(data, aes(x=remove_outliers(data$avg_glucose_level))) + geom_boxplot() + xlab("Average glucose levels without outliers")
#append filtered values
data$age <- remove_outliers(data$age)
data$bmi <- remove_outliers(data$bmi)
data$avg_glucose_level <-remove_outliers(data$avg_glucose_level)

# drop 1 record with unique category "other" from var "Other"
data <- subset(data, gender!="Other")
data
# Splitting dataset to train-test
split <- sample.split(data, SplitRatio = 0.75)
split

train_reg <- subset(data, split == "TRUE")
test_reg <- subset(data, split == "FALSE")

#standartising the data - mean = 0 , sd = 1
train_reg[3] <- as.data.frame(scale(train_reg[3]))
train_reg[9:10] <- as.data.frame(scale(train_reg[9:10]))
train_reg

test_reg[3] <- as.data.frame(scale(test_reg[3]))
test_reg[9:10] <- as.data.frame(scale(test_reg[9:10]))
test_reg
#impute missing values with nearest instance
train_reg <- kNN(train_reg, k = 5, imp_var = F)
train_reg
test_reg <- kNN(test_reg, k = 5, imp_var = F)
test_reg
#Imbalanced data
ggplot(data, aes(stroke)) + geom_bar()
summary(data$stroke)
#Binary logistic regression
glm.fit <- glm(stroke ~ age + bmi + avg_glucose_level, 
                      data = train_reg, 
                      family = "binomial")
glm.fit

summary(glm.fit) #very big intercept, age coef. is very significant
# Predict test data based on model
glm.probs <- predict(glm.fit,
                     newdata = test_reg,
                     type = "response")
glm.probs
has_stroke <- test_reg$stroke
has_stroke
mean(glm.probs)
# 50% threshold for decision
glm.pred <- ifelse(glm.probs > 0.5, 1 ,0)
glm.pred #our model performed badly due to unbalanced data
table(glm.pred, has_stroke) 
mean(glm.pred == has_stroke)#but the prediction rate is high
#our model classifies all observations as non-stroke cases and is mostly accurate
auc(data$stroke, data$age)#we can see that the age of the patient is good predictor
auc(data$stroke, data$bmi) 
auc(data$stroke, data$avg_glucose_level) #bmi and avg_glucose_level parameters
#are random classifiers, their predictive power is close to zero

# Evaluating model accuracy
# using confusion matrix
table(test_reg$stroke, glm.pred)
missing_class_error <- mean(glm.pred != test_reg$stroke)
missing_class_error
print(paste("Accuracy =", 1 - missing_class_error))


# ROC-AUC Curve
ROCPred <- prediction(glm.pred, test_reg$stroke)
ROCPer <- performance(ROCPred, measure = "tpr", 
                      x.measure = "fpr")

auc <- performance(ROCPred, measure = "auc")
auc <- auc@y.values[[1]]
auc
# Plotting curve
plot(auc)
plot(ROCPer, colorize = TRUE, 
     print.cutoffs.at = seq(0.1, by = 0.1), 
     main = "ROC CURVE")
abline(a = 0, b = 1)

auc <- round(auc, 4)
legend(.6, .4, auc_downsampled, title = "AUC", cex = 1) #change

#Undersampling
downsampled_data = NULL
for (c in unique(data$stroke)) {
  tmp<-ovun.sample(stroke ~ ., data = test_reg, method = "under", p = 0.5, seed = 54)$data
  downsampled_data<-rbind(downsampled_data, tmp)
}
downsampled_data
table(downsampled_data$stroke)

#Train logistic regression with undersampled data
glm.fit_downsampled <- glm(stroke ~ age + bmi + avg_glucose_level, 
               data = downsampled_data, 
               family = "binomial") #delete bmi a gluc
glm.fit_downsampled

summary(glm.fit_downsampled) #very big intercept, age coef. is very significant
# Predict test data based on model
glm.probs_downsampled <- predict(glm.fit_downsampled,
                     newdata = test_reg,
                     type = "response")
glm.probs_downsampled
has_stroke
mean(glm.probs_downsampled)
# 50% threshold for decision
glm.pred_downsampled <- ifelse(glm.probs_downsampled > 0.5, 1 ,0)
glm.pred_downsampled #our model performed badly due to unbalanced data
table(glm.pred_downsampled, has_stroke) 
mean(glm.pred_downsampled == has_stroke)

# Evaluating model accuracy
# using confusion matrix
table(test_reg$stroke, glm.pred_downsampled)
missing_class_error_downsampled <- mean(glm.pred_downsampled != test_reg$stroke)
missing_class_error_downsampled
print(paste("Accuracy =", 1 - missing_class_error_downsampled))


# ROC-AUC Curve
ROCPred_downsampled <- prediction(glm.pred_downsampled, test_reg$stroke)
ROCPer_downsampled <- performance(ROCPred_downsampled, measure = "tpr", 
                      x.measure = "fpr")

auc_downsampled <- performance(ROCPred_downsampled, measure = "auc")
auc_downsampled <- auc_downsampled@y.values[[1]]
auc_downsampled
# Plotting curve
plot(ROCPer_downsampled)
plot(ROCPer_downsampled, colorize = TRUE, 
     print.cutoffs.at = seq(0.1, by = 0.1), 
     main = "ROC CURVE on downsampled data")
abline(a = 0, b = 1)

auc_downsampled <- round(auc, 4)
legend(.6, .4, auc_downsampled, title = "AUC", cex = 1)

#Oversampling
upsampled_data = NULL
for (c in unique(data$stroke)) {
  tmp<-ovun.sample(stroke ~ ., data = test_reg, method = "over", p = 0.5, seed = 5)$data
  upsampled_data<-rbind(upsampled_data, tmp)
}

upsampled_data
table(upsampled_data$stroke)
#Train logistic regression with oversampled data

glm.fit_upsampled <- glm(stroke ~ age + bmi + avg_glucose_level, 
                        data = upsampled_data, 
                        family = "binomial")
glm.fit_upsampled

summary(glm.fit_upsampled)
# Predict test data based on model
glm.probs_upsampled <- predict(glm.fit_upsampled,
                              newdata = test_reg,
                              type = "response")
glm.probs_upsampled
has_stroke
mean(glm.probs_upsampled)
# 50% threshold for decision
glm.pred_upsampled <- ifelse(glm.probs_upsampled > 0.5, 1 ,0)
glm.pred_upsampled
table(glm.pred_upsampled, has_stroke) 
mean(glm.pred_upsampled == has_stroke)

# Evaluating model accuracy
# using confusion matrix
table(test_reg$stroke, glm.pred_upsampled)
missing_class_error_upsampled <- mean(glm.pred_upsampled != test_reg$stroke)
missing_class_error_upsampled
print(paste("Accuracy =", 1 - missing_class_error_upsampled))


# ROC-AUC Curve
ROCPred_upsampled <- prediction(glm.pred_upsampled, test_reg$stroke)
ROCPer_upsampled <- performance(ROCPred_upsampled, measure = "tpr", 
                               x.measure = "fpr")

auc_upsampled<- performance(ROCPred_upsampled, measure = "auc")
auc_upsampled <- auc_upsampled@y.values[[1]]
auc_upsampled
# Plotting curve
plot(ROCPer_upsampled)
plot(ROCPer_upsampled, colorize = TRUE, 
     print.cutoffs.at = seq(0.1, by = 0.1), 
     main = "ROC CURVE on upsampled data")
abline(a = 0, b = 1)

auc_upsampled<- round(auc, 4)
legend(.6, .4, auc_upsampled, title = "AUC", cex = 1)

#resampling using the SMOTE technique

balanced_data <- smote(stroke ~ .,data = train_reg,  perc.over = 5,perc.under=2)
balanced_data
table(balanced_data$stroke)
#Train logistic regression with SMOTE data

glm.fit_balanced <- glm(stroke ~ age + bmi + avg_glucose_level, 
                         data = balanced_data, 
                         family = "binomial")
glm.fit_balanced

summary(glm.fit_balanced)
# Predict test data based on model
glm.probs_balanced <- predict(glm.fit_balanced,
                               newdata = test_reg,
                               type = "response")
glm.probs_balanced
has_stroke
mean(glm.probs_balanced)
# 50% threshold for decision
glm.pred_balanced <- ifelse(glm.probs_balanced > 0.5, 1 ,0)
glm.pred_balanced
table(glm.pred_balanced, has_stroke) 
mean(glm.pred_balanced == has_stroke)

# Evaluating model accuracy
# using confusion matrix
table(test_reg$stroke, glm.pred_balanced)
missing_class_error_balanced <- mean(glm.pred_balanced != test_reg$stroke)
missing_class_error_balanced
print(paste("Accuracy =", 1 - missing_class_error_balanced))


# ROC-AUC Curve
ROCPred_balanced <- prediction(glm.pred_balanced, test_reg$stroke)
ROCPer_balanced <- performance(ROCPred_balanced, measure = "tpr", 
                                x.measure = "fpr")

auc_balanced <- performance(ROCPred_balanced, measure = "auc")
auc_balanced <- auc_balanced@y.values[[1]]
auc_balanced
# Plotting curve
plot(ROCPer_balanced)
plot(ROCPer_balanced, colorize = TRUE, 
     print.cutoffs.at = seq(0.1, by = 0.1), 
     main = "ROC CURVE on SMOTE-balanced data")
abline(a = 0, b = 1)

auc_balanced<- round(auc, 4)
legend(.6, .4, auc_balanced, title = "AUC", cex = 1)


balanced_data_test <- balanced_data[-1]
#Fitting SVM to the Training set

classifier <- svm(formula = stroke ~ .,
                 data = balanced_data_test,
                 type = 'C-classification',
                 kernel = 'linear')
classifier
# Predicting the Test set results


y_pred = predict(classifier, newdata = test_reg)
y_pred
mean(y_pred == has_stroke)

# Evaluating model accuracy
# using confusion matrix
table(test_reg$stroke, y_pred)
missing_class_error_SVM <- mean(y_pred != test_reg$stroke)
missing_class_error_SVM
print(paste("Accuracy =", 1 - missing_class_error_SVM))

# Plotting the training data set results

roc_svm_test <- roc(response = test_reg$stroke, predictor = as.numeric(y_pred))
plot.new()
plot(roc_svm_test, add = TRUE, col = "red",  print.auc=TRUE,  print.auc.x = 0.5, print.auc.y = 0.3)
legend(0.3, 0.2, legend = c("test-svm"), lty = c(1), col = c("blue"))
plot.new()
ROCPred_SVM <- prediction(as.numeric(y_pred), test_reg$stroke)
ROCPer_SVM <- performance(ROCPred_SVM, measure = "tpr", 
                               x.measure = "fpr")

auc_SVM <- performance(ROCPred_SVM, measure = "auc")
auc_SVM <- auc_SVM@y.values[[1]]
auc_SVM
# Plotting curve
plot(ROCPer_SVM)
plot(ROCPer_SVM, colorize = TRUE, 
     print.cutoffs.at = seq(0.1, by = 0.1), 
     main = "ROC CURVE SVM")
abline(a = 0, b = 1)

auc_SVM<- round(auc, 4)
legend(.6, .4, auc_SVM, title = "AUC", cex = 1)

#Balanced Bagging Classifier
fit <- bagging(stroke~., data = balanced_data_test, coob = T, nbagg = 100)
print(fit)
pred_bagging <- predict(fit, newdata = test_reg)
result <- data.frame(original = test_reg$stroke, predicted = pred_bagging)
print(result)
#model accuracy
mean(pred_bagging == has_stroke)
#the probability of a stroke for each patient
print(predict(fit, test_reg, type = "prob"))
# Random Forest with class weighting
#computing the percentage of data in the df to be used as weights in the tree
y = balanced_data_test$stroke
wn = sum(y==0)/length(y)
wy = 1
wn
wy

# Select mtry value with minimum out of bag(OOB) error.
# We calculate it using the following formula: floor(sqrt(ncol(mydata) - 1))
mtry <- tuneRF(balanced_data[-12],balanced_data$stroke, ntreeTry=1000,
               stepFactor=1.5,improve=0.01, trace=TRUE, plot=TRUE)
best.m <- mtry[mtry[, 2] == min(mtry[, 2]), 1]
print(mtry)
print(best.m)

rf <- randomForest(
  stroke ~ .,
  data = balanced_data_test,
  mtry=best.m,
  ntree = 1000,
  classwt = c(wn, wy),
  importance = TRUE
)
rf

rf_predict_train <- predict(rf, train_reg)

rf_predict_train
table(train_reg$stroke, rf_predict_train)
missing_class_error_rf <- mean(rf_predict_train!= train_reg$stroke)
missing_class_error_rf
print(paste("Accuracy =", 1 - missing_class_error_rf))
#we get great accuracy on independent data samples
importance(rf)
#the most prominent variable in the prediction is by far age
varImpPlot(rf)
  
