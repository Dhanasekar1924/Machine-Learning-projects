# -*- coding: utf-8 -*-
"""
Created on Sun Jul 11 12:50:27 2021

@author: chandru
"""

import mlcp.pipeline as pl
import mlcp.classifires as cl
import mlcp.regressors as rg
import numpy as np
from datetime import datetime as dt
import math
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")
#execution controls
classification=1; #if regression = 0, classification = 1
read=1
primary_analysis=1 #dev only
observed_corrections=0
analyze_missing_values=1
treat_missing_values=0
define_variables=1
analyze_stats=1; #dev only
analyzed_corrections=0;
oversample=0; #dev only
feature_engineering=0
gaussian_transform=0
polynomial_transform=0
skew_corrections=0
scaling=0
encoding=0
matrix_corrections=0
reduce_dim=0
compare=0; #dev only
cross_validate=0; #dev only
grid_search=0; #dev only
train_classification=1
train_regression=0


if read==1:
    filepath = "data/KNN/Mobile_data.csv"
    y_name = 'price_range'
    df = pl.read_data(filepath)
    
#    df = df[['battery_power','px_height','ram','px_width','price_range']]


if primary_analysis==1:
    #consider: unwanted features, numerical conversions (year to no. years), 
    #wrong dtypes, missing values
    df_h = df.head()
    with open("project8_dtype_analysis.txt", "w") as f:
        for c in df_h:
            line1 = df_h[c]
            line2 = df[c].nunique()
            line3 = df[c].isnull().sum()
            f.write(str(line1) + "\n" + "Unique: " + str(line2) + 
                    ", missing: " + str(line3)
            + "\n\n" + "-----------------"+"\n")
    if classification == 0:
        plt.boxplot(df[y_name]); plt.show()

visual=0
if visual==1:
    ds=[];i=0
    s1='price_range'
    for c in sorted(df[y_name].unique()):
        df_name =  "df"+str(c)
        df_name = df[df[y_name]==c]
        ds.append([])
        for s in df.columns:
            s2 = s
            ds[i].append(list(df_name[s2]))
            ds[i].append(list(df_name[s1]))
        i=i+1
        
    ds = np.array(ds); print(ds.shape)
    
    i=0
    for j in range(0,len(ds[0]),2):
        print(df.columns[i]); i=i+1
        plt.scatter(ds[0][j],ds[0][j+1],c='blue')
        plt.scatter(ds[1][j],ds[1][j+1],c='green')
        plt.scatter(ds[2][j],ds[2][j+1],c='yellow')
        plt.scatter(ds[3][j],ds[3][j+1],c='red')
        plt.show()


if observed_corrections==1:
#    df['price_range'] = df['price_range'].astype(str)
    pass
    
    
if analyze_missing_values==1:
    drop_th = 0.4
    print(df.shape)
    df = pl.missing_value_analysis(df, drop_th)
    print(df.shape)
    before = len(df); df_copy_drop = df.dropna(); after = len(df_copy_drop); 
    print("dropped %--->", round(1-(after/before),2)*100,"%")
    num_df = df.select_dtypes(exclude=['O'])
    pl.correlations(num_df, th=0.5)
    

if treat_missing_values==1:
    pass


if define_variables==1:
    y = df[y_name]
    x = df.drop([y_name],axis=1); #print(x.info())
    n_dim = x.shape[1]
    print(x.shape)



if analyze_stats==1:
   #find important features and remove correlated features based on low-variance or High-skew
    pl.correlations(x, th=0.5)
    scores = pl.feature_analysis(x,y); print(scores); print("")
#    if classification == 1:
#        ranks = pl.feature_selection(x,y); print(ranks); print("")
    print(x.skew())
    
    
if analyzed_corrections==1:
    pass
    

if oversample==1:
    #for only imbalanced data
    x,y = pl.oversampling(x,y)
    print(x.shape); print(y.value_counts())
    

def fscore(a,b):
    return (a*b)/(a+b)

def combine(a,b):
    return a.astype(str)+b.astype(str)


if feature_engineering==1:
   #subjective and optional - True enables the execution
   print("Initial feature:");print(x.head(1));print("")
   
   x['nf1'] = fscore(x.fc,x.pc)
   x['nf2'] = combine(x.four_g,x.three_g)
   
   x=x.drop(['fc','pc','three_g','four_g'],axis=1)
   print(x.head(1));print("")


         
if polynomial_transform==1:
   degree=6
   x = pl.polynomial_features(x,degree)
   print("polynomial features:")
   print(x.head(1)); print("")


if gaussian_transform==1:
   n_dim=5
   x = pl.gaussian_features(x,y,n_dim)
   print("Gaussian features:")
   print(x.head(1)); print(x.shape,y.shape);print("")
   

if skew_corrections==1:
    x = pl.skew_correction(x)


if scaling==1:
    x_num, x_cat = pl.split_num_cat(x)
    
    if False:
        x_num, fm = pl.max_normalization(x_num)
    if False:
        x_num = pl.minmax_normalization(x_num)
    if False:
        x_num = pl.Standardization(x_num.values)
         
    x = pl.join_num_cat(x_num,x_cat)


if encoding==1:
    x_num, x_cat = pl.split_num_cat(x)
    
    if True:
        x_cat = pl.label_encode(x_cat)
    if False:
        x_cat = pl.onehot_encode(x_cat)
    
    if False:
         x,y,mmd = pl.auto_transform_data(x,y); #best choice if dtypes are fixed
    
    x = pl.join_num_cat(x_num,x_cat)     
         

if matrix_corrections==1:   
    print("before matrix correction:", x.shape)
    x,y = pl.matrix_correction(x,y); 
    print("after matrix correction:", x.shape)  
    

if reduce_dim==1:
    x = pl.reduce_dimensions(x, 2); #print(x.shape)
    print("transformed x:")
    print(x.shape); print("")
    

if compare==1:
    model_meta_data, Newx = pl.compare_models(x, y)
    best_model_id = model_meta_data['best_model'][0]
    best_model = cl.get_models()[best_model_id]; print(best_model)
    
    
if cross_validate==1:
    best_model = cl.KNeighborsClassifier()
    pl.kfold_cross_validate(best_model, x, y,111)


if grid_search==1:
    #grids
    dtc_param_grid = {"criterion":["gini", "entropy"],
                      "class_weight":[{0:1,1:1}],
                      "max_depth":[2,4,6,8,10],
                      "min_samples_leaf":[1,2,3,4,5],
                      "min_samples_split":[2,3,4,5,6],
                      "random_state":[21,111]
                      }
    
    log_param_grid = {"penalty":['l1','l2','elasticnet'],
                      "C":[0.1,0.5,1,2,5,10],
                      "class_weight":[{0:1,1:1}],
                      "solver":['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                      "max_iter":[100,150,200,300],
                      "random_state":[21,111]
                      }
    
    param_grid = log_param_grid
    best_param_mn,odel = pl.select_best_parameters(best_model,param_grid, x, y, 111)
 
#x_train , x_test,y_train,y_test = train_test_split(x,y,test_size=0.4,random_state=111)
k_range = range(1,26)
scores= []
for k in k_range:
    knn = cl.KNeighborsClassifier(n_neighbors=k)
    knn.fit(x,y)
    y_pred = knn.predict(x)
    scores.append((y))
print(scores)
if train_classification==1:
    best_param_model = cl.KNeighborsClassifier

    pl.clf_train_test(best_param_model,x,y,111,"knn")
  
    
if train_regression==1:
   model = rg.GradientBoostingRegressor()
   pl.reg_train_test(model,x,y,111,"GBR1")
    