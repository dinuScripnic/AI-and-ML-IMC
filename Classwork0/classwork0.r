#Create a dataframe with 1000 observations and 2 columns

#The first column should be a random number between 1 and 10
#The second column should be class A,B,C A = 80% B = 15% C = 5%
#Apply  90/10 train/test split

df <- data.frame(value = sample(1:10, 1000, replace = TRUE), class = sample(c("A", "B", "C"), 1000, replace = TRUE, prob = c(0.8, 0.15, 0.05)))

#Create a train/test split  

#function to split the data into train and test with 90/10 split
splitting = function(df){
  train <- df[sample(1:nrow(df), 0.9*nrow(df)), ]
  test <- df[-sample(1:nrow(df), 0.9*nrow(df)), ]
  #return the train and test data frame
  return(list(train, test))
}

#stratify
res = splitting(df)
res

library(caret, help, pos = 2, lib.loc = NULL)
#stratify
set.seed(123)
stratified_split <- createDataPartition(df$class, p = 0.9, list = FALSE, times = 1)
train <- df[stratified_split, ]
test <- df[-stratified_split, ]

train
test

#summary
table(df)
table(df$value)
