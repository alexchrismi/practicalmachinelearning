Practical Machine Learning Course Project
Alexander Smit
12/23/2020
Introduction
In this assignment, we use data from accelerometers on the belt, forearm, arm, and dumbbell of 6 participants that performed barbell lifts correctly and incorrectly in 5 different ways to predict how the exercise was done. First, we set-up the working space. We then explain how we built the model, how we used cross-validation, and the expected out-of-sample error of our final model. Throughout the report, we justify the different choices made in our analysis. Have fun reading, and thanks for reviewing this assignment!

Installing packages, setting-up functions and reading the data
First, we loaded the necessary packages for our analysis, set-up a function that allows us to plot multiple graphs at the same time and read the data.

## Loading required package: lattice
## Loading required package: ggplot2
## Loading required package: foreach
## Loading required package: iterators
## Loading required package: parallel
Building the model
Before we built our model, we conducted exploratory data analysis to get an idea of the data set. Based on this exploratory step, we decided to reduce the data by (1) excluding irrelevant variables and (2) reduce the set of remaining variables.

Ad. 1: Excluding irrelevant variables
We got an idea of the different variables and the data structure by looking at the testing data set (without using it, mind you) and reading the explanation of the data set that can be found here.

Regarding the testing data set, running a summary()-command already indicates which columns are useless because they all contain NA-values (to benefit this document’s readability, we leave this command out of this report). These NA-values are explained in the explanation of the data set as well: “Each IMU has x, y, and z values + Euler angles (roll, pitch and yaw). For each time window (1s of data), there are several statistics calculations, like Kurtosis, Variance, etc.”. We verified this for one of the time windows by filtering the 2nd time window (this corresponds to activities performed by Eurico), getting the only non-NA value for the avg_roll_belt variable and compare this value with the average of all roll_belt observations for this time window:

dta.tmp <- training[ training$num_window == 2, ]
unique(dta.tmp$avg_roll_belt)[2]
## [1] -27.4
mean(dta.tmp$roll_belt)
## [1] -27.43333
remove(dta.tmp)
We also observed that some of the variables in the data set do not meaningfully predict how a subject does an exercise (the classe-score). These variables are the index (V1), the subject’s name (user_name) and the different time indicators (raw_timestamp_part_1, raw_timestamp_part_2, cvtd_timestamp, new_window and num_window). Although such time indicators in a data set could reflect the presence of time series, this is not the case with this data: inspection of these time indicators using the sort(unique(training$cvtd_timestamp)) command reveals that all data was collected in a rather short time span (i.e. between 28/11/2011 & 05/12/2011). So, said variables are removed from both training and test set as well.

sort(unique(training$cvtd_timestamp))
##  [1] "02/12/2011 13:32" "02/12/2011 13:33" "02/12/2011 13:34" "02/12/2011 13:35"
##  [5] "02/12/2011 14:56" "02/12/2011 14:57" "02/12/2011 14:58" "02/12/2011 14:59"
##  [9] "05/12/2011 11:23" "05/12/2011 11:24" "05/12/2011 11:25" "05/12/2011 14:22"
## [13] "05/12/2011 14:23" "05/12/2011 14:24" "28/11/2011 14:13" "28/11/2011 14:14"
## [17] "28/11/2011 14:15" "30/11/2011 17:10" "30/11/2011 17:11" "30/11/2011 17:12"
After ascertaining that the variables are listed in the same order in both data frames by running the colnames(testing) == colnames(training) command (except for the target variable, which differs), we select the potential useful columns.

unique(colnames(training[, -c(160)]) == colnames(testing[, -c(160)]))
## [1] TRUE
training <- training[ c(8:11, 37:49, 60:68, 84:86, 102, 113:124, 140, 151:160) ]
testing <- testing[ c(8:11, 37:49, 60:68, 84:86, 102, 113:124, 140, 151:160) ]
Ad. 2: Data reduction through principal component analysis
The remaining training dataset still has quite some predictors (52 in total). Given the nature of the measurements (e.g. acceleration in the x, y and z-direction when performing a certain action, which can be expected to be strongly correlated), it only seems logical to try reduce this data further. We do this using principal component analysis. To see if the data is suitable for this, we first plot all predictors (because the grid.arrange function seems to be a bit unstable with a large number of graphs, we commented this code out)

# plt <- lapply(colnames(training), plt.mul, data = training)
# do.call("grid.arrange", c(plt[ 1:16], nrow = 4, ncol = 4))
# do.call("grid.arrange", c(plt[17:32], nrow = 4, ncol = 4))
# do.call("grid.arrange", c(plt[33:48], nrow = 4, ncol = 4))
# do.call("grid.arrange", c(plt[49:52], nrow = 4, ncol = 4))
From this data inspection step, we deduced that there are no outliers. This also makes sense if you think about it: unless the researchers invited the Hulk or someone the like, it is not very likely that extremely large values than the mean will occur in this type of human exercise data. As an additional exploration step, we run the preProcess-function to conduct the pca:

preProcess(training[, -53], method = "pca") # the 53th column contains the target variable, which is left out
## Created from 19622 samples and 52 variables
## 
## Pre-processing:
##   - centered (52)
##   - ignored (0)
##   - principal component signal extraction (52)
##   - scaled (52)
## 
## PCA needed 25 components to capture 95 percent of the variance
So, this reduction step yields 25 components that capture 95% of the original data set variance, which is quite a bit. We, therefore, continue working with the reduced data set. This is great, as it also speeds up our subsequent modelling efforts. Following the advice found here, we add the preprocess command to the train-function below and do not do this upfront.

Modelling strategy
It is easy to get lost in the many parameters one can set in the train function (i.e. the trainControl-argument). We, therefore, decided to take the code that was provided during the lecture as a starting point and optimize this code for this specific data.

Regarding our modelling strategy, our preferred strategy would be one of stacking (i.e. estimating 3 different models, stacking the resulting predictions together and use the resulting data set to predict the outcomes of the testing data). Unfortunately, this is not possible due to the lack of the outcome variable in the testing data. We, therefore, use the random forest, which, according to the lecture, is one of the most accurate classification methods (together with boosting). An important consideration of working with large data sets is modelling speed. We took two measures to speed-up model estimation. The first one is changing the default bootstrapping for cross-validation to repeated k-fold cross validation. Not only are the lecture slides a bit ambivalent as to the use of bootstrapping (see the corresponding slides), repeated k-fold cross-validation is also advised to use with random forests (see, e.g. https://bit.ly/3ph4jqS) as it reduces both bias and variance (another topic that seems to be highly debated, but that is for a different assignment).

cvc <- trainControl(method = "repeatedcv", # this stands for repeated k-fold cross validation 
                    number = 10, # the training data is randomly divided in 10 parts. Each of these parts is testing data for the model trained on the other 9. The error terms are averaged
                    repeats = 5, # the above process is repeated 5 times, and again the 5 error terms are averaged
                    allowParallel = TRUE) # parallel computing is allowed
The number of repeats might have been a bit too high with the benefit of hindsight, but the most important thing is that this code works and leads to an outcome reasonably fast. We also allow for parallel computing using the command below. This allows the model to be estimated using most of the processor cores simultaneously.

cluster <- makeCluster(detectCores() - 1)
registerDoParallel(cluster)
Fitting the model and expected out-of-sample error
Now we fit the model (and have a coffee, as the train-command below takes about 20 minutes. It probably could be shorter but it works like a charm and I live my life slowly anyway):

mod.rf <- train(classe ~., 
                data = training, 
                method = "rf", 
                verbose = FALSE,
                trControl = cvc,
                preProcess = c("pca"))
mod.rf
## Random Forest 
## 
## 19622 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## Pre-processing: principal component signal extraction (52), centered
##  (52), scaled (52) 
## Resampling: Cross-Validated (10 fold, repeated 5 times) 
## Summary of sample sizes: 17659, 17658, 17660, 17658, 17661, 17660, ... 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa    
##    2    0.9821934  0.9774725
##   27    0.9731012  0.9659757
##   52    0.9733255  0.9662596
## 
## Accuracy was used to select the optimal model using the largest value.
## The final value used for the model was mtry = 2.
# De-registering the parallel processing cluster
stopCluster(cluster)
registerDoSEQ()
We can assess the quality of our model by looking at the confusion matrix (using the training data, as the testing data does not contain the classe variable. The confusion matrix using the training data is based on a comparison of the modelled data to the held-out folds):

confusionMatrix.train(mod.rf)
## Cross-Validated (10 fold, repeated 5 times) Confusion Matrix 
## 
## (entries are percentual average cell counts across resamples)
##  
##           Reference
## Prediction    A    B    C    D    E
##          A 28.3  0.3  0.0  0.0  0.0
##          B  0.0 18.9  0.2  0.0  0.1
##          C  0.1  0.2 17.1  0.6  0.1
##          D  0.0  0.0  0.1 15.8  0.1
##          E  0.0  0.0  0.0  0.0 18.2
##                             
##  Accuracy (average) : 0.9822
The model’s reported accuracy is very high (close to 100%), which means that the model suits the training data well (in other words: the off-diagonal percentages are very low).We can also expect that this model is a little tuned to the noise in this training data, which implies that it will perform less well for the test data. Hence, the out-of-sample error for this test data will be larger. We did manage to get a 100% score for the prediction quiz, though. Although this does not say anything about our model’s accuracy (it probably is because the answers in the quiz are based on the predicted value, not on the observed values), it at least suggests this is the type of model the instructors envisioned when making this assignment. Cool stuff! :)