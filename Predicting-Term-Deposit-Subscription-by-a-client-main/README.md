# Predicting Term Deposit Subscription by a client
This code provides a set of functions to compare the performance of different machine learning models on a given dataset. The code uses popular classifiers such as Logistic Regression, K-Nearest Neighbors, Decision Tree Classifier, Multinomial Naive Bayes, Support Vector Classifier, Random Forest Classifier, Gradient Boosting Classifier, and MLP Classifier to compare their accuracy and ROC-AUC scores.

The code also implements a Stacking Classifier, which is a meta-estimator that combines multiple machine learning models via a meta learner.

The functions in this code can perform text vectorization, one-hot encoding of categorical features, and log transformation of skewed numerical features.

Functions
sort_by_value(dictx)
This function takes a dictionary as input and returns a sorted version of that dictionary by values.

data_to_df(x)
This function takes an input dataset and returns a pandas dataframe.

tex2vec(X)
This function performs text vectorization on a given dataset. It uses the TfidfVectorizer from sklearn.

get_models()
This function returns a dictionary of machine learning models that are used to compare the performance of different models.

evaluate_model(model, X, y, rstate)
This function evaluates a given model on a given dataset. It uses Repeated Stratified K-Fold cross-validation to evaluate the model.

get_stacking(models)
This function returns a stacking classifier that combines multiple machine learning models via a meta learner.

prepare_data(X,Y)
This function prepares the


