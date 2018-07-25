library(caret)
library(e1071)
library(keras)
library(tidyverse)

#sample dataset is from https://github.com/haensel-ams/recruitment_challenge/tree/master/ML_201703

#set directory and load data
datapath<-"../Deep_Learning_in_R/Multiclass Classification"
setwd(datapath)

mydata<-read_csv("../Multiclass Classification/sample.csv",col_names = FALSE)

#check data types
tibble(types=sapply(mydata, class)) %>%
  group_by(types) %>%
  summarize(num=n())


#convert the last column to numeric 
mydata<-mydata %>%
  mutate(X296=factor(X296))
levels(mydata$X296)
mydata<-mydata %>%
  mutate(X296=as.integer(X296)-1) 

#removal of constant columns
drop<-names(mydata[,sapply(mydata, function(v) ifelse(is.numeric(v),var(v, na.rm=TRUE)==0,FALSE))])
drop
mydata<- mydata %>%
  select(-one_of(drop))
#find out all non-binary features.
non_binary<-mydata[,sapply(sapply(mydata,unique),function(v) ifelse(length(v)!=2,T,F))]
non_binary
#check skewness and drop some features
sapply(non_binary,skewness)
highSkew<-c('X5','X24','X37')
ggplot(non_binary)+
  geom_freqpoly(aes(x=X5))

ggplot(non_binary)+
  geom_freqpoly(aes(x=X24))

ggplot(non_binary)+
  geom_freqpoly(aes(x=X37))


#convert columns with high skewness to categorical data
lapply(mydata[,highSkew],unique)
mydata<-mydata %>%
  mutate(X5=if_else(X5==10,7,as.numeric(X5)), #as.numeric(X5) refer to question https://github.com/jllipatz/USSR/issues/1
         X24= case_when(
           X24<0 ~0,
           T ~ 1),
         X37=case_when(
           X37<0 ~ 0,
           X37==0 ~ 1,
           X37<2000 & X37>0 ~ 2,
           X37>=2000 & X37<1400000 ~ 3,
           X37>=1400000 ~ 4,
           T ~ 5)) %>%
  mutate(X5=as.factor(X5),
         X24=as.factor(X24),
         X37=as.factor(X37))
dmy <- dummyVars(" ~ .", data = mydata[,highSkew])
trsf <-predict(dmy, newdata = mydata[,highSkew])
ncol(trsf)
head(trsf)

mydata<- mydata %>%
  select(-one_of(highSkew))

#check missing value
mydata[,sapply(mydata, function(x) sum(is.na(x))>0)]

#change to matrix
mydata<-data.matrix(mydata)
trsf<-data.matrix(trsf)
#dimnames(mydata)<-NULL

#normalize non_binary features only.
norm_fea<-c('X4','X44','X65','X295')
mydata[,norm_fea]=normalize(mydata[,norm_fea])
mydata<-cbind(mydata[,1:281],trsf,mydata[,282])
total_col=ncol(mydata)

#data partition
set.seed(123)
ind<-sample(nrow(mydata),size=floor(0.75*nrow(mydata)))
train<-mydata[ind,1:total_col-1]
test<-mydata[-ind,1:total_col-1]
train_target<-mydata[ind,total_col]
test_target<-mydata[-ind,total_col]

#one hot encoding
trainlabels<-to_categorical(train_target)
testlabels<-to_categorical(test_target)
testlabels


#create sequential model
model1<-keras_model_sequential()
model1 %>%
  layer_dense(units = 200,activation='relu',input_shape = c(total_col-1))%>%
  layer_dropout(rate=0.6) %>%
  layer_dense(units = 100,activation='relu')%>%
  layer_dropout(rate=0.5) %>%
  layer_dense(units = 30,activation='relu')%>%
  layer_dropout(rate=0.2) %>%
  layer_dense(units = 10,activation='relu')%>%
  layer_dropout(rate=0.1) %>%
  layer_dense(units=5,activation = 'softmax')
summary(model1)


#compile
model1 %>%
  compile(loss='categorical_crossentropy',
          optimizer = 'adam',
          metrics = 'accuracy')

# fit model
hist<- model1 %>%
  fit(train,
      trainlabels,
      epoch=25,
      batch_size = 128,
      validation_split = 0.2)

#evaluate model with test data
model1 %>%
  evaluate(test,testlabels)

#prediction and confusion matrix- test data
prob<-model1 %>%
  predict_proba(test)
pred<-model1 %>%
  predict_classes(test)

table1<-table(predicted=pred,actual=test_target)
table1
