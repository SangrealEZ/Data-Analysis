# Bank Case
# This case is based around a real world dataset about telemarketing calls made by a Portuguese bank. You can find more information about this dataset here: https://archive.ics.uci.edu/ml/datasets/bank+marketing
# The bank is interested in a predictive model because it will allow them to call the right customers at the right times. From an analytics perspective, the primary distinguishing feature in this case is that solving a predictive problem is directly useful to the firm.

# load data
bankData = read.csv('D:/Dropbox/Teaching Lectures/Bank Case.csv',stringsAsFactors=TRUE)
# The str function lets you see a nice summary of the data.
str(bankData)

# The dependent variable we are interested in is y, which indicates whether the customer signed up for the new deposit account.
bankData$y = bankData$y == 'yes'

# Duration’ refers to how long the call was. The variable should be deleted from the analysis, as the firm does not know how long the call will be before they make it
bankData$duration = NULL

####################################################################################
# start from some basic descriptive analysis to get a feel of the dataset
####################################################################################
# take a look at correlation
library("corrplot")
corrplot(cor(model.matrix(~.,data=bankData)))
# Some obvious correlations are apparent in the correlation plot: 
# Age is negatively correlated with marriage, but positively correlated with income. This gives some confidence in the integrity dataset.

# use a linear model to get a feel of attribute importance
summary(lm(y~.,data=bankData))
anova(lm(y~.,data=bankData))
# It turns out that seasonality is potentially very important here.
# Furthermore, retired, single people are more likely to say yes. 
# However, housing and loan seem unimportant. 

# use Mars to see if there are any non-linear relationship
library(earth)
earth1 = earth(y~.,data=trainingData)
plotmo(earth1)
# We see that age has a non-linear relationship, and that retired individuals are more likely to say yes:

library('leaps')
basicSubset = regsubsets(y~.,data=bankData)
basicSummary = summary(basicSubset)
# looking at AIC, BIC
bestAIC = which.min(basicSummary$cp)
bestBIC = which.min(basicSummary$bic)
coef(basicSubset,bestBIC)
# We learn that retired individuals and students are more likely to say yes
# March and September are the best times to call

# Alternatively, use Lasso to shrink model
library('glmnet')
lassoFit = glmnet(model.matrix(~.,data=bankData[,-11]),bankData$y,alpha=1)
plot(lassoFit,xvar='lambda',sub='The X-Axis is the penalty parameters (logged)')

# In lasso, the coefficients depend on your choice of lambda. 
# Higher lambda implies a higher penalty, which means the coefficients are smaller. 
# This plot shows how the coefficients change with lambda. As lambda gets larger, more coefficients are forced to be 0. If lambda is close to 0, you get linear regression estimates.
# In an explanatory analysis, we might want a high lambda so we can focus on the most important coefficients. 
# For example, in this case lambda = .01 is a relatively high penalty term, which results in a model that is relatively easy to interpret:
predict(lassoFit,s = .01, type = 'coefficients')

# we could also use Lasso for prediction
cvLassoFit = cv.glmnet(model.matrix(~.,data=bankData[,-11]),bankData$y,alpha=1)
predict(cvLassoFit,s = cvLassoFit$lambda.min, type = 'coefficients')

validationMat = model.matrix(~.,data=validationData[,-11])

cvLassoFit = cv.glmnet(model.matrix(~.,data=trainingData[,-11]),trainingData$y,alpha=1)
mean((validationData$y - predict(cvLassoFit,newx= validationMat, type = 'response',s=cvLassoFit$lambda.min)^2))
# 0.08695193

cvRidgeFit = cv.glmnet(model.matrix(~.,data=trainingData[,-11]),trainingData$y,alpha=0)
mean((validationData$y - predict(cvRidgeFit,newx= validationMat, type = 'response',s=cvRidgeFit$lambda.min)^2))
# 0.08687274

cvElasticFit = cv.glmnet(model.matrix(~.,data=trainingData[,-11]),trainingData$y,alpha=.5)
mean((validationData$y - predict(cvElasticFit,newx= validationMat, type = 'response',s=cvElasticFit$lambda.min)^2))
# 0.08678545

trainingMat2 = model.matrix(~.^2,data=trainingData[,-c(6,7,11)])
validationMat2 = model.matrix(~.^2,data=validationData[,-c(6,7,11)])

cvLassoFit2 = cv.glmnet(trainingMat2,trainingData$y,alpha=1)
mean((validationData$y - predict(cvLassoFit2,newx= validationMat2, type = 'response',s=cvLassoFit2$lambda.min)^2))
# 0.08536662

cvRidgeFit2 = cv.glmnet(trainingMat2,trainingData$y,alpha=0)
mean((validationData$y - predict(cvRidgeFit2,newx= validationMat2, type = 'response',s=cvRidgeFit2$lambda.min)^2))
# 0.08294781

cvElasticFit2 = cv.glmnet(trainingMat2,trainingData$y,alpha=.5)
mean((validationData$y - predict(cvElasticFit2,newx= validationMat2, type = 'response',s=cvElasticFit2$lambda.min)^2))
# 0.08530279

####################################################################################
# Predictive Modeling and Tuning
####################################################################################

# define traing and validataion dataset
set.seed(1)
isTraining = runif(nrow(bankData))<.8
trainingData = subset(bankData,isTraining)
validationData = subset(bankData,!isTraining)

# Start from Linear 

lm1 = lm(y~age+factor(month),data=trainingData)
lm2 = lm(y~poly(age,3)+factor(month),data=trainingData)
lm3 = lm(y~.,data=trainingData)
lm4 = lm(y~.^2,data=trainingData)

mean((predict(lm1,validationData) - validationData$y)^2)
# 0.09076575
mean((predict(lm2,validationData) - validationData$y)^2)
# 0.08963436
mean((predict(lm3,validationData) - validationData$y)^2)
# 0.0881642
mean((predict(lm4,validationData) - validationData$y)^2)
# 0.08765032
anova(lm3)

# Housing/Loan seem unimportant.  Lets delete those. 
lm5 = lm(y~.^2,data=trainingData[,-c(6,7)])
mean((predict(lm5,validationData) - validationData$y)^2)
# 0.08724407
# That worked.  Let's keep that up with a more manual formula
anova(lm5)

lm6 = lm(y~age+job+marital+education+default+contact+factor(month)+factor(day_of_week)+age*job+age*marital+age*education+job*default+job*month+marital*month+education*month+default*contact+default*month*contact*month+month*day_of_week,data=trainingData)
mean((predict(lm6,validationData) - validationData$y)^2)
# 0.08648047

# Using regression statistics helped to train a better predictive model. This represents a somewhat trained lm model. We can start to tune an earth model as well.

library('earth')
earth1 = earth(y~.,data=trainingData)
earth2 = earth(y~.,data=trainingData,degree=2)
earth3 = earth(y~age+job+marital+education+default+contact+factor(month)+factor(day_of_week)+age*job+age*marital+age*education+job*default+job*month+marital*month+education*month+default*contact+default*month*contact*month+month*day_of_week,data=trainingData)
#Earth 2 is the best of these

mean((predict(earth1,validationData) - validationData$y)^2)
## [1] 0.08834947
mean((predict(earth2,validationData) - validationData$y)^2)
## [1] 0.08739309
mean((predict(earth3,validationData) - validationData$y)^2)
## [1] 0.08752177
earth4 = earth(y~.,data=trainingData,degree=2,thres=0)
earth5 = earth(y~.,data=trainingData,degree=2,thres=0.01)
earth6 = earth(y~.,data=trainingData,degree=2,thres=0.1)

mean((predict(earth4,validationData) - validationData$y)^2)
## [1] 0.08661586
mean((predict(earth5,validationData) - validationData$y)^2)
## [1] 0.08954148
mean((predict(earth6,validationData) - validationData$y)^2)
## [1] 0.09787158
#Earth 4 is the best of these

####################################################################################
# Neuralnet and Deep Learning
####################################################################################
# Install the package

library('neuralnet')
bankDataMat = model.matrix(~.,data=bankData)

#Rename some of the columns so the method works correctly
colnames(bankDataMat)[3] <- "jobbluecollar"
colnames(bankDataMat)[8] <- "jobselfemployed"

#Split into training/validatoin data 
trainingMat = bankDataMat[isTraining,]
validationMat = bankDataMat[!isTraining,]


#Generate a correct formula
col_list <- paste(c(colnames(validationMat[,-c(1,44)])),collapse="+")
col_list <- paste(c("yTRUE~",col_list),collapse="")
f <- formula(col_list)


#This fit a simple neural network with 3 units in the hidden layer.  
basicSingleLayerNNet <- neuralnet(f, data=trainingMat,
                            algorithm = "rprop+",
                            hidden=c(3),
                            threshold=0.1,
                            stepmax = 1e+06)

#Get predictions for the validation data (this is super finicky)
output <- compute(basicSingleLayerNNet, validationMat[,-c(1,44)],rep=1)
#Calculate out of sample performance
mean((validationMat[,44] - output$net.result)^2)
## [1] 0.08773072346
This next section fits a deep learning neural network. Notice that ‘hidden’ is now a vector that equal (3,3). This means that a neural network with two hidden layers, each with 3 nodes, will be fit. The second model has 10 units in the first layer, and 3 units in the second.

#This will take a lot of time to compute.
deepLearningNnet1 <- neuralnet(f, data=trainingMat,
                            algorithm = "rprop+",
                            hidden=c(3,3),
                            threshold=0.1,
                            stepmax = 1e+06)
output <- compute(deepLearningNnet1, validationMat[,-c(1,44)],rep=1)
mean((validationMat[,44] - output$net.result)^2)
## [1] 0.08652831484
plot(deepLearningNnet1)


deepLearningNnet2 <- neuralnet(f, data=trainingMat,
                             algorithm = "rprop+",
                             hidden=c(10,3),
                             threshold=0.1,
                             stepmax = 1e+06)
output <- compute(deepLearningNnet2, validationMat[,-c(1,44)],rep=1)
mean((validationMat[,44] - output$net.result)^2)
## [1] 0.09246228593
plot(deepLearningNnet2)
