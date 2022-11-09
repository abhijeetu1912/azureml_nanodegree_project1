# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Useful Resources
- [ScriptRunConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.scriptrunconfig?view=azure-ml-py)
- [Configure and submit training runs](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-set-up-training-targets)
- [HyperDriveConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.hyperdriveconfig?view=azure-ml-py)
- [How to tune hyperparamters](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters)


## Summary
The dataset used in this project is related with the direct marketing campaign of a Portugese banking institution. This data contains 39,251 rows and 21 columns including dependent variable. It contains both numerical and categorical features. The goal of the project is to predict wheter client will subscrib to the term deposit or not. Dependent variable contains 3,692 instances of 'yes' and remaining instances of 'no'. This data contains client's information like age, job, marital status, education, contact etc.

In this project, we tuned logistic regression with HyperDrive and also used AutoML to find the best performing model. VotingEnsemble from AutoML gave the best performance with accuracy of 0.9178 compared to 0.9118 of tuned logistic regression from HyperDrive.

## Scikit-learn Pipeline
We have used the logistic regression model from scikit-learn and tuned it using HyperDrive. Pipeline for this task consists of below steps: 

1. Collect data from prescribed url.
2. Clean & prepare data - processing date, one-hot encoding of categorical columns etc.
3. Split the data into train & test set at 75:25 ratio with stratified sampling on dependent variable 'y'.
4. Create hyperparameter Sampler of hyperparameters 'C' & 'max_iter.
5. Create policy for early stopping using bandit approach.
6. Train model on train data and fine tune it by HyperDrive using hyperparameter sampler and bandit policy.
7. Evaluate model on test data- accuracy.
8. Save model in '.pkl' format using joblib package.

### Benefits of Hyperparameter Sampling
I have used RandomParameterSampling on two hyperparameters called C and max_iter. C controls the level of regularization in model and max_iter decides how long a model should be trained. Lower value of C means higher regularization. Unlike grid parameter sampling, random paramater sampling doesn't look at every combination of hyperparameter values. Instead, it just selects random combination of hyperparameters and gives almost similar performance in lesser run time.

### Benefits of Early Stopping Policy**
I have used BanditPolicy for early stopping. It terminates the run when it doesn't fall within the slack factor of the evaluation metric with respect to the best performing model at an interval. It helps in saving compute when subsequent runs are not giving better performance.

## AutoML
AutoML fits variety of machine learning models at different hyperparameter values and find the best performing model based on primary performance metric. In my AutoML setup, experiment run time was set to be 30 minutes to make sure experiement doesn't goes beyond 30 mins in run time. AutoML tries different data transformation approaches and hyperparameters like max depth, max leaves, sampling rate, number of estimnatores, regularization, solvers etc. based on type of models, whether they are linear or tree based or anything else. In this case various models like lightgbm, xgboost, logistic regression etc. was implemented. VotingEnsemble was the best performing model with accuracy of 0.91778. It was followed by models like StackEnsemble, xgboost, lightgbm etc.

## Pipeline comparison
**Compare the two models and their performance. What are the differences in accuracy? In architecture? If there was a difference, why do you think there was one?**

## Future work
Some improvements can be done in HyperDrive training by using more types of regularization including elastic net method. As data set is imbalanced, class weights can be used and multiple performance metrics like precision, recall, f1 score, auc etc. can be used for better evaluation of models in both HyperDrive and AutoML. Model explanaibility approaches can also be tried to explain the result of models.

