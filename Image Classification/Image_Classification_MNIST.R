#Load library
library(keras)

#Load dataset MNIST from keras
mnist<-dataset_mnist()
train_images<-mnist$train$x
train_labels<-mnist$train$y
test_images<-mnist$test$x
test_labels<-mnist$test$y

#Training data overview
str(train_images)
str(train_labels)

#Preparing the data
train_images<-array_reshape(train_images,c(60000,28 * 28))
train_images<-train_images/255

test_images<-array_reshape(test_images,c(10000,28 * 28))
test_images<-test_images/255

#One hot encode y
train_labels<-to_categorical(train_labels)
test_labels<-to_categorical(test_labels)


#Build the model
model_m<-keras_model_sequential() %>%
  layer_dense(units=600,activation = 'relu',input_shape = c(28 * 28)) %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units=100,activation = 'relu') %>%
  layer_dropout(rate = 0.4) %>%
  layer_dense(units=10,activation = 'softmax')

#Compile the model
model_m %>%
  compile(optimizer = 'adam',
          loss='categorical_crossentropy',
          metrics = 'accuracy')

#Train model
model_m %>%
  fit(train_images,train_labels,
      epochs = 20,
      batch_size=16)

#Evaluate the model
model_m %>% evaluate(test_images,test_labels)
model_m %>%
  predict_classes(test_images[1:20,])
plot(as.raster(mnist$train$x[105,,],max = 255))
