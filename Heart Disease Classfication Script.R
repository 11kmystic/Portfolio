library(caret)
library(corrplot)
library(dplyr)
library(e1071)
library(kernlab)
library(ggplot2)
library(pROC)
library(randomForest)
library(tidyverse)
set.seed(169)

#Loading in Data
df <- read.csv("HeartDiseaseTrain-Test.csv")
summary(df)

#View unique values for categorical variables
sexes <- unique(df$sex)
blood_sugar <- unique(df$fasting_blood_sugar)
ang <- unique(df$exercise_induced_angina)
slope <-unique(df$slope)
vessel <-unique(df$vessels_colored_by_flourosopy)
thal <-unique(df$thalassemia)
sexes
blood_sugar
ang
slope
vessel
thal


#Allows for interactions
ves <- as.list(df$vessels_colored_by_flourosopy)

#New vector to replace old values
new_ves <- c()
for (i in ves) {
  if (i == "Two"){
    new_ves <- c(new_ves, 2)
  }
  else if (i == "Zero") {
    new_ves <- c(new_ves, 0)
  }
  else if (i == "One") {
    new_ves <- c(new_ves, 1)
  }
  else if (i == "Three") {
    new_ves <- c(new_ves, 3)
  }
  else {
    new_ves <- c(new_ves, 4) #assuming "Four" is the last unique value
  }
}

df$vessels_colored_by_flourosopy <- new_ves

# Changes 0 to Negative, 1 to Positive, and factors characters for easier use
data <- df %>% mutate(target = if_else( target ==1, "Positive", "Negative")) %>%
  mutate_if(is.character, as.factor)

# Data Visualization

ggplot(data, aes(x=target, fill=target))+
  geom_bar()+
  xlab("Heart Disease")+
  ylab("count")+
  ggtitle("Presence & Absence of Heart Disease")+
  theme(plot.title = element_text(hjust = 0.5)) +
  scale_fill_discrete(name= 'Heart Disease', labels =c("Absence", "Presence"))

## Variable Investigation

data %>%
  group_by( age) %>%
  count() %>%
  ggplot()+
  geom_col(aes(x = age, y = n), fill = 'orange')+
  ggtitle("Age Analysis")+
  xlab("Age")+
  ylab("Count")+
  theme(plot.title = element_text(hjust = 0.5))

sd(df$age)
mean(df$age)

data %>%
  ggplot(aes(x=sex, y= resting_blood_pressure))+
  geom_boxplot(fill ='purple')+
  xlab('sex')+
  ylab('Resting Blood Presure')+
  ggtitle("Angina Instances vs Resting Blood Pressure") +
  theme(plot.title = element_text(hjust = 0.5)) +
  facet_grid(~chest_pain_type)

data %>%
  ggplot(aes(x=sex, y= cholestoral))+
  geom_boxplot(fill ='green')+
  xlab('sex')+
  ylab('Cholesterol')+
  ggtitle("Cholesterol vs Chest Pain") +
  theme(plot.title = element_text(hjust = 0.5)) +
  facet_grid(~chest_pain_type)

#Correlation Plot
num_variables <- c("age","resting_blood_pressure","cholestoral","Max_heart_rate","oldpeak")

corr <- cor(data[, num_variables])

corrplot(corr, method = 'square', type = 'upper')

## Train and Test Data

partition<- createDataPartition(data$target, p =0.7, list = FALSE)
train_data <- data[partition,]
test_data <- data[-partition,]

# Logistic Regression

AUC <- list()
accuracy <- list()

log_reg_model <- train (target ~ ., data=train_data, method = 'glm', # For distrubution other than normal
                        family = 'binomial')
log_reg_prediction <- predict(log_reg_model, test_data)
log_reg_prediction_prob <- predict(log_reg_model, test_data, type='prob')[2]
log_reg_confusion <- confusionMatrix(log_reg_prediction, test_data[,"target"])

#ROC Curve
AUC$logReg <- roc(as.numeric(test_data$target),as.numeric(as.matrix((log_reg_prediction_prob))))$auc
accuracy$logReg <- as.numeric(log_reg_confusion$overall['Accuracy'])  #found names with str

## Support Vector Machine

fitControl <- trainControl(method = "repeatedcv",
                           number = 10,
                           repeats = 10,
                           ## Estimate class probabilities
                           classProbs = TRUE,
                           ## Evaluate performance using 
                           ## the following function
                           summaryFunction = twoClassSummary)

svm <- train(target ~ ., data = train_data,
             method = "svmRadial",
             trControl = fitControl,  
             preProcess = c("center", "scale"), #correspond to chosen method
             tuneLength = 8,
             metric = "ROC")

svmPrediction <- predict(svm, test_data)
svmPredictionprob <- predict(svm, test_data, type='prob')[2]
svmConfMat <- confusionMatrix(svmPrediction, test_data[,"target"])
#ROC Curve
AUC$svm <- roc(as.numeric(test_data$target),as.numeric(as.matrix((svmPredictionprob))))$auc
accuracy$svm <- as.numeric(svmConfMat$overall['Accuracy'])

## Random Forest

RFModel <- randomForest(target ~ .,
                        data=train_data, 
                        importance=TRUE, 
                        ntree=200)
#varImpPlot(RFModel)
RFPrediction <- predict(RFModel, test_data)
RFPredictionprob = predict(RFModel,test_data,type="prob")[, 2]

RFConfMat <- confusionMatrix(RFPrediction, test_data[,"target"])

AUC$RF <- roc(as.numeric(test_data$target),as.numeric(as.matrix((RFPredictionprob))))$auc
accuracy$RF <- as.numeric(RFConfMat$overall['Accuracy'])  

## Final Comparison

row.names <- names(accuracy)
col.names <- c("AUC", "Accuracy")
results_table <- cbind(as.data.frame(matrix(c(AUC,accuracy),nrow = 3, ncol = 2,
                                            dimnames = list(row.names, col.names))))
results_table

log_reg_confusion
svmConfMat
RFConfMat