#Load library
library(keras)

#Input data
mydata<-dataset_boston_housing()
c(c(x_train,y_train),c(x_test,y_test)) %<-% mydata
str(x_train)

summary(x_train)
summary(y_train)

#V4 is a binary variable, and variables have quite different ranges.
#The price for a house is between 50k and 500k.

#Normalize the data
x_train<-scale(x_train)
x_test<-scale(x_test)


#Build the model
model<-keras_model_sequential() %>%
  layer_dense(units = 64,activation = 'relu',input_shape =c(ncol(x_train))) %>%
  layer_dense(units = 64,activation = 'relu') %>%
  layer_dense(units = 1) %>%
  compile(optimizer = 'rmsprop',
          loss = 'mse',
          metrics = 'mae') # Mean absolute Error
#Use K-fold cross-validation to reduce the variance of the validation score.
k<-5
set.seed(123)
indices<-sample(1:nrow(x_train))
folds<-cut(indices,breaks = k,labels = F)
