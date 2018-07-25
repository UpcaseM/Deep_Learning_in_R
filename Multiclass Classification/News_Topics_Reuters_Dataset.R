
#Load dataset
library(keras)

#Input data for top 10000 frequently occuring words
reuters<-dataset_reuters(num_words = 10000)
train_data<-reuters$train$x
train_labels<-reuters$train$y
test_data<-reuters$test$x
test_labels<-reuters$test$y

#Data overview
str(train_data)
str(test_data)

#Decode the words to see what is the contents
word_ind<-dataset_reuters_word_index()
words<-names(word_ind)
names(words)<-word_ind
paste(sapply(train_data[[1]],function(ind) {
  word<-if(ind>=3) words[[as.character(ind-3)]]
  if (!is.null(word)) word else '?'
}),collapse = ' ')

to_cat <- function(data, dim = 10000) {
  results <- matrix(0, nrow = length(data), ncol = dim)
  for (i in 1:length(data))
    results[i, data[[i]]] <- 1
  results
}

x_train<-to_cat(train_data)
x_test<-to_cat(test_data)

y_train<-to_categorical(train_labels)
y_test<-to_categorical(test_labels)

model <- keras_model_sequential() %>%
  layer_dense(units = 128, activation = "relu", input_shape = c(10000)) %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 46, activation = "softmax")

model %>% compile(
  optimizer = "rmsprop",
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)

history <- model %>% fit(
  x_train,y_train,
  epochs = 8,
  batch_size = 256,
  validation_split = 0.2
)

#Evaluate the model
model %>% evaluate(x_test,y_test)

test_val<-sample(test_labels)
length(which(test_val==test_labels))/length(test_labels)
#The accuracy reached by a purely random classifier is closer to 
#18%, but our model reaches 79% which is pretty good.
