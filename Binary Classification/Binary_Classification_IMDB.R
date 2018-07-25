
#Load library
library(keras)

#Load dataset IMDB
#This dataset is a set of 50,000 movie reviews. They're split to two parts for
#training and testing. The words of veviews have been pre-processed and turned
#into integers(Rank of frequency for each word in the whole dataset). 
#The goal here is to predict whether a reivew is positive or negative.
imdb<-dataset_imdb(num_words=10000) #Only keep top 10000 more frequently occurring words.
train_data <- imdb$train$x
train_labels <- imdb$train$y
test_data <- imdb$test$x
test_labels <- imdb$test$y

#Overview
str(train_data)
str(train_labels)

#Data pre-processing
#One hot encode for 100000 classes
to_binary<-function(data,dim=10000) {
  results<-matrix(0,nrow=length(data),ncol=dim)
  for (i in 1:length(data))
    results[i,data[[i]]]<-1
  results
}

x_train<-to_binary(train_data)
x_test<-to_binary(test_data)
y_train <- as.numeric(train_labels)
y_test <- as.numeric(test_labels)
#Bulid the model
model <- keras_model_sequential() %>%
  layer_dense(units = 16, activation = "relu", input_shape = c(10000)) %>%
  layer_dense(units = 16, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")
#What activation function should we use? See more details here.https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6

#Compile the model
model %>% compile(
  optimizer = 'rmsprop',
  loss='binary_crossentropy',
  metrics='accuracy'
)

#Review the summary of the model
model

history<-model %>% fit(
  x_train,
  y_train,
  epochs = 4,
  batch_size = 512,
  validation_split = 0.2
)

plot(history)

model %>% evaluate(x_test,y_test)

#Reached an accuracy of 0.88308