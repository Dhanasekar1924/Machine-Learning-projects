import mlcp.pipeline as pl
import mlcp.classifires as cl
import mlcp.regressors as rg
import numpy as np
import pandas as pd
from datetime import datetime as dt
import math
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean


import warnings
warnings.filterwarnings("ignore")
#execution controls
classification=1; #if regression = 0, classification = 1
read=1
primary_analysis=0 #dev only
visual_analysis=0
observed_corrections=1
feature_engineering=0
analyze_missing_values=1
treat_missing_values=1
define_variables=1
analyze_stats=0; #dev only
analyzed_corrections=1;
gaussian_transform=0
polynomial_transform=0
skew_corrections=0
scaling=0 #do for continuous numerical values, don't for binary/ordinal/one-hot
encoding=1
matrix_corrections=0
oversample=1; #dev only
reduce_dim=0
compare=1; #dev only
cross_validate=0; #dev only
grid_search=0; #dev only
train_classification=0
train_regression=0



if read==1:
    filepath = "data/LOG_5/h1n1_vaccine_prediction.csv"
    y_name = 'h1n1_vaccine'
    dtype_file = "project5_dtype_analysis.txt"
    df = pl.read_data(filepath)
   
    else:
        print("y skew--->", df[y_name].skew())
        

if primary_analysis==1:
    #consider: unwanted features, numerical conversions (year to no. years), 
    #wrong dtypes, missing values, categorical to ordinal numbers
    df_h = df.head()
    with open(dtype_file, "w") as f:
        for c in df_h:
            line1 = df_h[c]
            line2 = df[c].nunique()
            line3 = df[c].isnull().sum()
            f.write(str(line1) + "\n" + "Unique: " + str(line2) + 
                    ", missing: " + str(line3)
            + "\n\n" + "-----------------"+"\n")
    if classification == 0:
        plt.boxplot(df[y_name]); plt.show()



if visual_analysis==1:
    pl.visualize_y_vs_x(df,y_name)



if observed_corrections==1:
    df = df.drop(['unique_id'],axis=1)
    pass
    
    

if feature_engineering==1:
   #subjective and optional - True enables the execution
   print("Initial feature:");print(df.head(1));print("")

   pos = ['h1n1_awareness','bought_face_mask','wash_hands_frequently',
          'avoid_large_gatherings','reduced_outside_home_cont','avoid_touch_face',
          'dr_recc_h1n1_vacc','is_health_worker']
   neg = ['h1n1_worry','antiviral_medication','contact_avoidance','dr_recc_seasonal_vacc',
          'chronic_medic_condition','sick_from_h1n1_vacc','sick_from_seas_vacc',]
   neu = ['cont_child_undr_6_mnths','is_h1n1_vacc_effective','is_h1n1_risky',
          'is_seas_vacc_effective', 'is_seas_risky','no_of_adults', 'no_of_children']
   
   df['pos'] = np.array([0 for i in range(df.shape[0])])
   for c in pos: df['pos'] = df['pos'] + df[c]
   
   df['neg'] = np.array([0 for i in range(df.shape[0])])
   for c in neg: df['neg'] = df['neg'] + df[c]
   
   df['neu'] = np.array([0 for i in range(df.shape[0])])
   for c in neu: df['neu'] = df['neu'] + df[c]
       
   df = df.drop(pos+neg+neu, axis=1)
   print("")
   print("Feature Engineering---->:"); print(df.head())
   pl.visualize_y_vs_x(df,y_name)
   
   
    
if analyze_missing_values==1:
    drop_th = 0.4
    print(df.shape)
    df = pl.missing_value_analysis(df, drop_th)
    print(df.shape)
    before = len(df); df_copy_drop = df.dropna(); after = len(df_copy_drop); 
    print("dropped %--->", round(1-(after/before),2)*100,"%")
    num_df = df.select_dtypes(exclude=['O'])
    

if treat_missing_values==1:
    for c in df.columns:
        c_mode = df[c].value_counts().index[0]
        df[c] = df[c].fillna(c_mode)
    pass


if define_variables==1:
    y = df[y_name]
    x = df.drop([y_name],axis=1)
    n_dim = x.shape[1]
    print(x.shape)


if analyze_stats==1:
   #find important features and remove correlated features based on low-variance or High-skew
    cors = pl.correlations(x, th=0.7)
    with open(dtype_file, "a") as f:
        f.write("\n\n\n"+str(cors))
    scores = pl.feature_analysis(x,y); print(scores); print("")
#    if classification == 1:
#        ranks = pl.feature_selection(x,y); print(ranks); print("")
    for c in x.columns:
        sd, minv, maxv = pl.bias_analysis(x,c)
        print(c, " = ", sd)
    print("")   
    print("skew in feature:")
    print(x.skew())
    
    
if analyzed_corrections==1:
#    x = x.drop(['antiviral_medication','bought_face_mask', 'cont_child_undr_6_mnths'
#                ],axis=1)
    pass
   
         
if polynomial_transform==1:
   degree=3
   x = pl.polynomial_features(x,degree)
   print("polynomial features:")
   print(x.head(1)); print("")


if gaussian_transform==1:
   n_dim=3
   x = pl.gaussian_features(x,y,n_dim)
   print("Gaussian features:")
   print(x.head(1)); print(x.shape,y.shape);print("")
   

if skew_corrections==1:
    x = pl.skew_correction(x)
 


if scaling==1:
    selective_scaling=0
    
    x_num, x_cat = pl.split_num_cat(x)
    
    if selective_scaling == 1:
        selective_features=[]
        selective_x_num = x_num[selective_features]
        x_num = x_num.drop(selective_features, axis=1)
    else:
        selective_x_num = x_num
    
    if False:
        selective_x_num, fm = pl.max_normalization(selective_x_num); #0-1
    if True:
        selective_x_num = pl.minmax_normalization(selective_x_num) ; #0-1
    if False:
        selective_x_num = pl.Standardization(selective_x_num); #-1 to 1
    
    print("")
    print("after scaling - categorical-->", x_cat.info())
    print("after scaling - numerical-->", x_num.shape)
    
    if selective_scaling == 1:
        x_num = pl.join_num_cat(x_num, selective_x_num)
    else:
        x_num = selective_x_num
        
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
    print("after encoding--->", x.shape)
      
    

if matrix_corrections==1:   
    x = pl.matrix_correction(x)
      
    
    
if oversample==1:
    #for only imbalanced data
    x,y = pl.oversampling(x,y)
    print(x.shape); print(y.value_counts())
    
    

if reduce_dim==1:
    x = pl.reduce_dimensions(x, 30); #print(x.shape)
    x = pd.DataFrame(x)
    print("transformed x:")
    print(x.shape); print("")
    

if compare==1:
    #compare models on sample
    n_samples = 500
    df_temp = pd.concat((x,y),axis=1)
    df_sample = pl.stratified_sample(df_temp, y_name, n_samples)
    print("stratified sample:"); print(df_sample[y_name].value_counts())
    y_sample = df_sample[y_name]
    x_sample = df_sample.drop([y_name],axis=1)
    model_meta_data = pl.compare_models(x_sample, y_sample, 111)
    
    
    
if cross_validate==1:
    #deciding the random state
    best_model = cl.GradientBoostingClassifier()
    pl.kfold_cross_validate(best_model, x, y,100)


if grid_search==1:
    #grids
    dtc_param_grid = {"criterion":["gini", "entropy"],
                      "class_weight":[{0:1,1:1.5}],
                      "max_depth":[2,4,6,8,10],
                      "min_samples_leaf":[1,2,3,4,5],
                      "min_samples_split":[2,3,4,5,6],
                      "random_state":[21,111]
                      }
    
    log_param_grid = {"penalty":['l1','l2','elasticnet'],
                      "C":[0.1,0.5,1,2,5,10],
                      "class_weight":[{0:1,1:1}],
                      "solver":['liblinear', 'sag', 'saga'],
                      "max_iter":[100,150,200,300],
                      "random_state":[100,111]
                      }
    
    param_grid = dtc_param_grid
    model = cl.RandomForestClassifier()
    best_param_model = pl.select_best_parameters(model, param_grid, x, y, 111)



if train_classification==1:
    print(x.shape)
    remove_features=[]
#    remove_features=[5, 32, 22, 27, 46, 41, 48, 8, 30, 37, 39, 49]; #ohe
#    remove_features=[23, 3, 26, 21, 18]; #label
    
    if remove_features!=[]:
        x = x.drop(remove_features, axis=1)
    
    best_param_model = cl.GradientBoostingClassifier(random_state=100)
    trained_model = pl.clf_train_test(best_param_model,x,y,111,"GBC1",pred_th=0.7)
    
    if True:
        #recycle the models with most important features
        fi = trained_model.feature_importances_; print("fi count-->", len(fi))
        fi_dict={}
        for i in range(len(x.columns)):
            fi_dict[x.columns[i]] = fi[i]
        fi_dict = pl.sort_by_value(fi_dict)
        print([k for k,v in fi_dict])
    
    
    
if train_regression==1:
   model = rg.GradientBoostingRegressor()
   pl.reg_train_test(model,x,y,111,"DTR1")
    