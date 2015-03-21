library(car)
setwd("E:/Kaggle_Titanic")
passengers<-read.table("train.csv",TRUE,",")
names(passengers)

plot(passengers$sex, passengers$survived)
# head(passengers)