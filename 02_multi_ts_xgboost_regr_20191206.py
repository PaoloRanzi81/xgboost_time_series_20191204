"""
TITLE: "Gradient boosting classifier grid search "
AUTHOR: Paolo Ranzi 
PYTHON VERSION: 3.6.7

DESCRIPTION:
    When setting analysis = 'grid_search' it will run a grid search on specific 
hyperparameters' ranges (please select the hyperparamenters and their parameters
to be tried). Once the best hyperparameters have been found, use analysis = 'bootstrapping'. 
Wiht 'bootstrapping' a 30 runs will be run in order to see how the cross-validation
oscillates. Than the median of the 30 runs' scores will be takes. Such a median is 
the real cross-validation score.  
    The script is parallelized by using 'joblib' Python library. Please set 'RELEASE' to 
your local system specifics if you would like to use the script by a single-core mode.
By default the script works by a multi-core/multi-threaded fashion. 
    Further, please change the following sections according to your individidual input preferences:
    - 'SETTING PATHS AND KEYWORDS'; 
    - 'PARAMETERS TO BE SET!!!'
    
"""


###############################################################################
## IMPORTING LIBRARIES
# import required Python libraries
import platform
import os
import numpy as np
import pandas as pd
from joblib import Parallel
from joblib import delayed
from multiprocessing import cpu_count
import matplotlib.pyplot as plt
#import argparse
import time
import datetime
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_error
from sklearn.inspection import permutation_importance
import xgboost as xgb
from sklearn.preprocessing import PowerTransformer
#from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
                   
###############################################################################
## SETTING PATHS AND KEYWORDS
# In order to set the correct pathways/folders, check which system are you
# using. It should be either Linux laptop (release == '5.0.0-29-generic') 
# or Linux server (release == '4.4.0-143-generic').
RELEASE = platform.release()

if RELEASE == '5.0.0-37-generic': # Linux laptop
   BASE_DIR_INPUT = ('/media/paolo/4C0F-08B1/analyses/python/tensorflow/kinlay_jonathan/raw_data')
   BASE_DIR_OUTPUT = ('/media/paolo/4C0F-08B1/analyses/python/tensorflow/kinlay_jonathan/outputs')
else:
   BASE_DIR_INPUT = ('/home/ubuntu/raw_data')
   BASE_DIR_OUTPUT = ('/home/ubuntu/outputs')


###############################################################################
## PARAMETERS TO BE SET!!!
input_file_name_1 = ('sensor_dataset.csv') 
output_file_name_1 = ('cv_results.csv')
output_file_name_2 = ('best_scores.csv')
parallel = True # whenever starting the script from the Linux bash, uncomment such a variable
horizon = 300 # the actual steps to be forecasted
# set analysis to be run
analysis = 'grid_search' 
#analysis = 'bootstrapping'
#analysis = 'features_importance'
#analysis = 'testing_prediction'
#analysis = 'plotting_prediction'

"""
###############################################################################
# set whether use parallel computing (parallel = True) or 
# single-core computing (parallel = False).
 # construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--parallel",  dest='parallel', action='store_true',
	help="# enable multi-core computation")
ap.add_argument("-no-p", "--no-parallel",  dest='parallel', action='store_false',
	help="# disable multi-core computation")
ap.add_argument("-a", "--analysis", type=str, default='bootstrapping',
	help="# type analysis's name") 
args = vars(ap.parse_args())

# grab the analysis you want to run. You have to write down the analysis name
# in the command line as done for all 'argparse' arguments.
analysis = args["analysis"] 

# grab the "parallel" statment and store it in a convenience variable
# you have to set it from the command line
parallel = args["parallel"] 
"""


# start clocking time
start_time = time.time()


###############################################################################
## LOADING
#loading the .csv files
#loading the .csv filew 
input_data_tmp_1 = pd.read_csv(os.path.sep.join([BASE_DIR_INPUT, 
                                         input_file_name_1]), header = 0) 

#test on small sample
#input_data_tmp_1 = input_data_tmp_1.copy()[:10]
    
    
###############################################################################
## PREPARING DATA FOR ML

# downsampling
#input_data_tmp = input_data_tmp_1.resample('D').sum()

# select only a single sensor/time-series
#input_data_tmp = input_data_tmp_1.loc[input_data_tmp_1['sensor'] == 1]
    
# no downsampling
input_data_tmp = input_data_tmp_1.copy()

# reset index
input_data_tmp.reset_index(drop = True, inplace = True)
    
# standardize values by PowerTransformer
input_data_tmp.loc[:, ['value']] = PowerTransformer(method='yeo-johnson',).fit_transform\
(input_data_tmp.loc[:, ['value']])

# create columns of lags and differences, respectively
input_data_tmp['lag_0'] = input_data_tmp.loc[:, ['value']].shift()
input_data_tmp['differences_0'] = input_data_tmp['lag_0'].diff()
input_data_tmp['lag_2'] = input_data_tmp.loc[:, ['value']].shift(2)
input_data_tmp['differences_2'] = input_data_tmp['lag_2'].diff()
input_data_tmp['lag_4'] = input_data_tmp.loc[:, ['value']].shift(4)
input_data_tmp['differences_4'] = input_data_tmp['lag_4'].diff()
#input_data_tmp['lag_6'] = input_data_tmp.loc[:, ['Quantity']].shift(6)
#input_data_tmp['differences_6'] = input_data_tmp['lag_6'].diff()
input_data_tmp = input_data_tmp.dropna()

# build a out-of-bag set
max_date = input_data_tmp.loc[:, ['gregorian']].max()

# out-of-bag length
obb_length = max_date - horizon

# out-of-bag index
oob_index = input_data_tmp.loc[input_data_tmp['gregorian'] == int(obb_length)].index[0]

# split in train vs test sets 
X_train, X_test_oob = input_data_tmp.iloc[:oob_index, :], input_data_tmp.iloc[oob_index: , :]

# build target training set
y_train = X_train.pop('value')
y_test_oob = X_test_oob.pop('value')

# reset index out-of-bag sets
X_test_oob.reset_index(drop = True, inplace = True)
y_test_oob.reset_index(drop = True, inplace = True)


"""
from sklearn.model_selection import TimeSeriesSplit
X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4]])
y = np.array([1, 2, 3, 4, 5, 6])
tscv = TimeSeriesSplit()
print(tscv)

for train_index, test_index in tscv.split(X):
   print("TRAIN:", train_index, "TEST:", test_index)
   X_train, X_test = X[train_index], X[test_index]
   y_train, y_test = y[train_index], y[test_index]
"""

##############################################################################
## GRADIENT BOOSTING GRID SEARCH (GOAL: find the range of optimal hyperparameters)
if analysis == 'grid_search': 
    # configure bootstrap
    n_iterations = 1
    stats = []
    best_score_df = pd.DataFrame()
    best_parameters_df = pd.DataFrame()
    #best_score = []
    best_parameters = []
    results = pd.DataFrame()
    
    
    for i in range(n_iterations):
        
        # set grid search's parameters
        model = xgb.XGBRegressor()
        param_grid = {'max_depth': [12], # 'max_depth' useful for controlling overfitting
                      #'max_depth': np.linspace(2, 12, 5, dtype = int), 
                      'learning_rate': [0.01], # so called `eta` value
                      #'learning_rate': np.linspace(0.01, 0.5, 5),
                      'n_estimators': [750],
                      #'n_estimators': np.linspace(250, 2500, 10, dtype = int), #number of trees, change it to 1000 for better results
                      'verbosity': [0],
                      'objective': ['reg:squarederror'],
                      #'objective': ['reg:squarederror', 'rank:pairwise', 'rank:ndcg','rank:map'], # or 'reg:squarederror', 
                      # 'rank:pairwise', 'rank:ndcg','rank:map'. 
                      'booster': ['gbtree'], 
                      #'booster': ['gbtree', 'dart', 'gblinear'], # or 'dart', 'gblinear'
                      'tree_method':['auto'], # or 'auto', 'exact', 'approx', 'hist', 'gpu_hist'
                      'n_jobs':[1],
                      #'n_jobs':[int(round((cpu_count() - 1), 0))], #when use hyperthread, xgboost may become slower
                      'gamma' : [0], # 'gamma' useful for controlling overfitting
                      #'gamma' : np.linspace(0, 0.9, 6),
                      'min_child_weight': [1], # 'min_child_weight' useful for controlling overfitting. 
                      #'min_child_weight': np.linspace(1, 6, 6, dtype = int),
                      # Smaller values (e.g. 1, 2) for 'min_child_weight' are more appropriate for highly imbalanced data-set
                      'max_delta_step': [0],
                      #'max_delta_step': [0], # 'max_delta_step' improved predicting the right probability. Set to 
                      # a finite number (say 1). It helps when class is extremely imbalanced
                      'subsample': [0.9], # 'subsample' useful for making training robust to noise; 
                      #'subsample': np.linspace(0.5, 1, 6),
                      #'colsample_bytree': [0.9], # 'colsample_bytree' useful for making training robust to noise; 
                      'colsample_bytree': np.linspace(0.5, 1, 6),
                      #'colsample_bylevel': [0.6], # 'colsample_bytree' useful for making training robust to noise; 
                      'colsample_bylevel': np.linspace(0.5, 1, 6),
                      #'colsample_bynode': [0.7], # 'colsample_bytree' useful for making training robust to noise; 
                      'colsample_bynode': np.linspace(0.5, 1, 6),
                      #'reg_alpha': [0.05], # 'colsample_bytree' useful for making training robust to noise; 
                      #'reg_alpha': np.linspace(0.005, 0.5, 5),
                      #'reg_lambda': [0.1], # 'colsample_bytree' useful for making training robust to noise; 
                      #'reg_lambda': np.linspace(0.005, 0.5, 5),
                      'scale_pos_weight': [1], # 'scale_pos_weight' it re-balance data-set according to
                      # positive vs negative weights ratio. '1' is used for high class imbalance data-set. Changing its value
                      # it is useful for AUC evaluation only.
                      #'scale_pos_weight': np.linspace(0.5, 1, 6),
                      'base_score': [0],
                      #'missing': ['None'],
                      'num_parallel_tree': [1],
                      #'num_parallel_tree': np.linspace(1, 10, 9, dtype = int),
                      'importance_type':['gain'] 
                      #'importance_type':['gain', 'weight', 'cover', 
                      #                   'total_gain', 'total_cover'] # or 'weight', 'cover', 'total_gain' or 'total_cover'; 
                      }   

        # set cross-validation  
        cv = TimeSeriesSplit(max_train_size = None, n_splits = 20)

        # grid search
        if RELEASE == '5.0.0-37-generic': # Linux laptop
            # single-core computation
        	n_jobs = 1
            
        else:
            # use multi-cores if available    
            n_jobs = int(round((cpu_count() - 1), 0))
            
        # initialize grid search
        grid = GridSearchCV(estimator = model, param_grid = param_grid,
                                cv = cv, n_jobs = n_jobs, 
                                refit = 'mean_absolute_error')
        
        # fit model
        grid_result = grid.fit(X_train, y_train) 

        # collect and save all CV results for plotting
        results_tmp = pd.DataFrame.from_dict(grid_result.cv_results_)
        results = results.append(results_tmp, ignore_index = True) 
    
        # collect best_scores   
        best_score_series = pd.Series(np.round(grid_result.best_score_, 3))
        best_score_df = best_score_df.append(best_score_series, 
                                             ignore_index = True, sort = False)
        best_parameters.append(grid_result.best_params_)
    
        # print
        print('Number iterations %.0f' % (i))


##############################################################################
## GRADIENT BOOSTING WITH BOOTSTRAPPING (GOAL: getting a robust cross-validation score)
if analysis == 'bootstrapping':
    # configure bootstrap
    n_iterations = 30
    best_score_df = pd.DataFrame()
        
    # set grid search's parameters
    model = xgb.XGBClassifier()
    param_grid = {'nthread':[int(round((cpu_count() - 1), 0))], #when use hyperthread, xgboost may become slower
                  #'nthread':[1], # when testing on local machine
                  'booster': ['gbtree'],
                  'objective':['rank:map'],
                  'learning_rate': [0.1325], 
                  'gamma' : [0.5], 
                  'max_depth': [4], 
                  'min_child_weight': [6], 
                  'scale_pos_weight': [1], 
                  'silent': [0],
                  'max_delta_step': [0], 
                  'subsample': [0.9], 
                  'colsample_bytree': [0.9], 
                  'n_estimators': [340], 
                  'num_class' : [4]
                  #'eval_metric'
                  }

    # function for computing grid search    
    def grid_search(model, param_grid, X, y):    
        
        # grid search
        if RELEASE == '5.0.0-37-generic': # Linux laptop
            # set cross-validation  
            cv = StratifiedShuffleSplit(n_splits = 3, test_size = 0.3, 
                                    random_state = None)

            # split data-set with 'stratify' option
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size = 0.3, random_state=None,
                shuffle = True, stratify = y)
                
            # set scoring method for cross-validation
            scoring_LOSS = make_scorer(log_loss, greater_is_better = False,
                                     needs_proba = True,
                                     labels = sorted(np.unique(y)))
        
            # dictionary with a set of scoring method
            scoring = {#'f1_macro': 'f1_macro', 'f1_micro': 'f1_micro', 
                    'log_loss': scoring_LOSS, 
                    'matthews_corrcoef': make_scorer(matthews_corrcoef)}
                
            # single-core computation
            grid = GridSearchCV(estimator = model, param_grid = param_grid,
                                iid = True, cv = cv, n_jobs = 1, scoring = scoring,
                                refit='matthews_corrcoef')
        
        else:
            # set cross-validation  
            # WARNING: for parellelizing code use as many n_splits as cpus 
            # set cross-validation  
            cv = StratifiedShuffleSplit(n_splits = int(round((cpu_count() - 1), 0)), test_size = 0.3, 
                                    random_state = None)

            # split data-set with 'stratify' option
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size = 0.3, random_state=None,
                shuffle = True, stratify = y)
                
            # set scoring method for cross-validation
            scoring_LOSS = make_scorer(log_loss, greater_is_better = False,
                                     needs_proba = True,
                                     labels = sorted(np.unique(y)))
        
            # dictionary with a set of scoring method
            scoring = {#'f1_macro': 'f1_macro', 'f1_micro': 'f1_micro', 
                    'log_loss': scoring_LOSS, 
                    'matthews_corrcoef': make_scorer(matthews_corrcoef)}
                
            # single-core computation
            grid = GridSearchCV(estimator = model, param_grid = param_grid,
                                iid = True, cv = cv, n_jobs = int(round((cpu_count() - 1), 0)), 
                                scoring = scoring, refit='matthews_corrcoef')
   
        # fit model
        grid_result = grid.fit(X_train, y_train)
        
        return (grid_result)
    
        
    # running either parallel or single-core computation. 
    if parallel:
    	# execute configs in parallel
        executor = Parallel(n_jobs= int(round((cpu_count() - 1), 0)), 
                                        backend='loky')
        tasks = (delayed(grid_search)(model, param_grid, X, 
                 y) for i in range(n_iterations))
        output = executor(tasks)
                
    else:
        output = [grid_search(model, param_grid, X, 
                              y) for i in range(n_iterations)]        
 
    # append output from 'joblib' in lists and DataFrames    
    stats = []
    best_score = []
    best_parameters = []
    results = pd.DataFrame() 
    results_features = pd.DataFrame()
    
    # collect and save all CV results for plotting
    for counter in range(0,len(output)):
        
        # collect cross-validation results (e.g. multiple metrics etc.)
        results_1 = pd.DataFrame.from_dict(output[counter].cv_results_)
        results = results.append(results_1, ignore_index=True) 

        # collect best_scores    
        best_score_series = pd.Series(np.round(output[counter].best_score_, 3))
        best_score_df = best_score_df.append(best_score_series, 
                                             ignore_index = True, sort = False)
        best_parameters.append(output[counter].best_params_)

    # print
    print('Number iterations: %.0f' % (len(output)))
    
  
##############################################################################
## GRADIENT BOOSTING FINAL MODEL FEATURE'S IMPORTANCE (GOALS: 
## 1. find xgboost 'estimators_' (called also coefficients) necessary for prediction; 
## 2. compute 'permutation_importance' for features' importance;  
if analysis == 'features_importance':
    # configure bootstrap
    n_iterations = 2
       
    # set hyperparameters
    param_grid_tmp = {'max_depth': [4],   
                      'learning_rate': [0.01], 
                      'n_estimators': [3500],
                      'verbosity': [0],
                      'objective': ['reg:squarederror'],
                      'booster': ['gbtree'], 
                      'tree_method':['auto'], 
                      'n_jobs':[1],
                      #'n_jobs':[int(round((cpu_count() - 1), 0))], # NOT WORKING 
                      # for fitting
                      'gamma' : [0.5], 
                      'min_child_weight': [2], 
                      'max_delta_step': [0],
                      'subsample': [0.9],  
                      'colsample_bytree': [0.9], 
                      'colsample_bylevel': [0.6], 
                      'colsample_bynode': [0.7], 
                      'scale_pos_weight': [1], 
                      'base_score': [0],
                      #'missing': ['None'],
                      'num_parallel_tree': [1],
                      'importance_type':['gain'],
                      'nthread': [1] 
                      #'nthread':[int(round((cpu_count() - 1), 0))]# WORKING!!!
                  }
    
    # convert dictionary to DataFrame   
    param_grid = pd.DataFrame(param_grid_tmp)

    # initialize model 
    model = xgb.XGBRegressor(max_depth = param_grid.loc[:,'max_depth'][0],
                             learning_rate = param_grid.loc[:,'learning_rate'][0],
                             n_estimators = param_grid.loc[:,'n_estimators'][0],
                             verbosity = param_grid.loc[:,'verbosity'][0],
                             objective = param_grid.loc[:,'objective'][0],
                             booster = param_grid.loc[:,'booster'][0],
                             tree_method = param_grid.loc[:,'tree_method'][0],
                             gamma = param_grid.loc[:,'gamma'][0],
                             min_child_weight = param_grid.loc[:,'min_child_weight'][0],
                             max_delta_step = param_grid.loc[:,'max_delta_step'][0],
                             subsample = param_grid.loc[:,'subsample'][0],
                             colsample_bytree = param_grid.loc[:,'colsample_bytree'][0],
                             colsample_bylevel = param_grid.loc[:,'colsample_bylevel'][0],
                             colsample_bynode = param_grid.loc[:,'colsample_bynode'][0],
                             scale_pos_weight = param_grid.loc[:,'scale_pos_weight'][0],
                             num_parallel_tree = param_grid.loc[:,'num_parallel_tree'][0],
                             importance_type = param_grid.loc[:,'importance_type'][0], 
                             nthread = param_grid.loc[:,'nthread'][0],
                             )
   
    # bug in sklean permutation_importance with n_jobs > 1. We need to convert 
    # features' labels to numbers (like a Numpy matrix).
    X_n = X_train.to_numpy()
    y_n = y_train.to_numpy()
    
    # fit model. Note that train set + validation has been considered in the final model-
    #model = xgb.XGBRegressor( objective =param_grid.loc[:,'objective'][0])
    final_model = model.fit(X_n, y_n)

     
#    ###########################################################################
#    ## PERMUTATION IMPORTANCE BY ELI5 
#    # get features name
#    #from sklearn.feature_extraction import DictVectorizer
#    #vec = DictVectorizer()
#    booster = final_model.get_booster()
#    #dir(booster)
#    
#    # convert object to DataFrame
#    original_feature_names = pd.DataFrame(booster.feature_names, columns = ['feature_name'])
#    
#    ## features' importance (not permutated ==> it is wrong!)
#    #features_importance = final_model.feature_importances_
#    
#    # generate new column taking the index as input
#    original_feature_names['feature'] = original_feature_names.index
#   
#    # permutating features 
#    from eli5.sklearn import PermutationImportance
#    #perm = PermutationImportance(final_model).fit(X_test_oob, y_test_oob, n_iter = 3, n_jobs = 1, scoring)
#    perm = PermutationImportance(final_model).fit(X_test_oob, y_test_oob, n_iter = 3)
#    
#    # show results of permutation importance
#    perm_imp_weights = eli5.explain_weights_df(perm)
#    
#    # delete useless 'x'
#    perm_imp_weights['feature'] = perm_imp_weights['feature'].replace({'x': ''}, regex=True)
#    
#    # convert column from object to integer
#    perm_imp_weights[['feature']] = perm_imp_weights[['feature']].astype(int)
#
#    # merge DataFrames in order to have correct feature name sorted by feature importance
#    merged_df = pd.merge(perm_imp_weights, original_feature_names, how ='inner', 
#                     on = 'feature') 
#        
#    # drop column
#    merged_df.drop(columns = ['feature'], inplace = True)
#    
#    # move last column to first position
#    cols = list(merged_df.columns)
#    cols = [cols[-1]] + cols[:-1]
#    merged_df = merged_df[cols]
#
#    # save cleaned DataFrame as .csv file
#    merged_df.to_csv(os.path.sep.join([BASE_DIR_OUTPUT, 
#                                     output_file_name_3]), index= False) 
#    
##    # 3 ways of computing xgboost's weights
##    weights_gain = eli5.explain_weights_df(final_model, importance_type ='gain')
##    weights_weight = eli5.explain_weights_df(final_model, importance_type ='weight')
##    weights_cover = eli5.explain_weights_df(final_model, importance_type ='cover')
##    
##    # explain prediction according to feature importance (not permuted)
##    t3 = eli5.explain_prediction_df(final_model, doc = X_test_oob.iloc[0], top = 5)


    ###########################################################################
    ## PERMUTATION IMPORTANCE BY SCIKIT-LEARN 

    # scoring method
    scoring = 'neg_root_mean_squared_error'
    
    if RELEASE == '5.0.0-37-generic': # Linux laptop
        n_jobs = 2  
        n_repeats = 3
    else:
        n_jobs = int(round((cpu_count() - 1), 0))
        n_repeats = 1000 
         
    # permutation importance
    result_importance = permutation_importance(final_model, X_n, y_n, n_repeats = n_repeats,
                               n_jobs = n_jobs,  scoring = scoring)
    
    # sort
    perm_sorted_idx = result_importance.importances_mean.argsort()
    tree_importance_sorted_idx = np.argsort(final_model.feature_importances_)
    tree_indices = np.arange(0, len(final_model.feature_importances_)) + 0.5

    # plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    ax1.barh(tree_indices,
             final_model.feature_importances_[tree_importance_sorted_idx], height=0.7)
    ax1.set_yticklabels(X_train.columns)
    ax1.set_yticks(tree_indices)
    ax1.set_ylim((0, len(final_model.feature_importances_)))
    ax2.boxplot(result_importance.importances[perm_sorted_idx].T, vert=False,
                labels = X_train.columns)
    fig.tight_layout()
    plt.show()
    
 
##############################################################################
## GRADIENT BOOSTING PREDICTION FINAL MODEL (GOALS: 
## 1. check the prediction performance with the out-of-bag/test set; 
if analysis == 'testing_prediction':
       # configure bootstrap
    n_iterations = 2
       
    # set hyperparameters
    param_grid_tmp = {'max_depth': [4],   
                      'learning_rate': [0.01], 
                      'n_estimators': [3500],
                      'verbosity': [0],
                      'objective': ['reg:squarederror'],
                      'booster': ['gbtree'], 
                      'tree_method':['auto'], 
                      'n_jobs':[1],
                      #'n_jobs':[int(round((cpu_count() - 1), 0))], # NOT WORKING 
                      # for fitting
                      'gamma' : [0.5], 
                      'min_child_weight': [2], 
                      'max_delta_step': [0],
                      'subsample': [0.9],  
                      'colsample_bytree': [0.9], 
                      'colsample_bylevel': [0.6], 
                      'colsample_bynode': [0.7], 
                      'scale_pos_weight': [1], 
                      'base_score': [0],
                      #'missing': ['None'],
                      'num_parallel_tree': [1],
                      'importance_type':['gain'],
                      'nthread': [1] 
                      #'nthread':[int(round((cpu_count() - 1), 0))]# WORKING!!!
                  }
    
    # convert dictionary to DataFrame   
    param_grid = pd.DataFrame(param_grid_tmp)

    # initialize model 
    model = xgb.XGBRegressor(max_depth = param_grid.loc[:,'max_depth'][0],
                             learning_rate = param_grid.loc[:,'learning_rate'][0],
                             n_estimators = param_grid.loc[:,'n_estimators'][0],
                             verbosity = param_grid.loc[:,'verbosity'][0],
                             objective = param_grid.loc[:,'objective'][0],
                             booster = param_grid.loc[:,'booster'][0],
                             tree_method = param_grid.loc[:,'tree_method'][0],
                             gamma = param_grid.loc[:,'gamma'][0],
                             min_child_weight = param_grid.loc[:,'min_child_weight'][0],
                             max_delta_step = param_grid.loc[:,'max_delta_step'][0],
                             subsample = param_grid.loc[:,'subsample'][0],
                             colsample_bytree = param_grid.loc[:,'colsample_bytree'][0],
                             colsample_bylevel = param_grid.loc[:,'colsample_bylevel'][0],
                             colsample_bynode = param_grid.loc[:,'colsample_bynode'][0],
                             scale_pos_weight = param_grid.loc[:,'scale_pos_weight'][0],
                             num_parallel_tree = param_grid.loc[:,'num_parallel_tree'][0],
                             importance_type = param_grid.loc[:,'importance_type'][0], 
                             nthread = param_grid.loc[:,'nthread'][0],
                             )
      
    # function for fitting model    
    def fitting_function(model, X_train, y_train):
 
        # fit model. Note that train set + validation has been considered in the final model-
        final_model = model.fit(X_train, y_train)
        
        # predict class each instance belogs to
        y_predicted = final_model.predict(X_test_oob)

        # predict probability each instance belogs to a specific class
        #y_predicted_prob = final_model.predict(X_test_oob, output_margin = True)  
        
        # compute RMSE metric
        y_test_oob_reset = y_test_oob.to_numpy()
        rmse_metric = np.sqrt(mean_squared_error(y_test_oob_reset, y_predicted))
        
        return (rmse_metric)

    # running either parallel or single-core computation. 
    if parallel:
    	# execute configs in parallel
        executor = Parallel(n_jobs= int(round((cpu_count() - 1), 0)), 
                                        backend='loky')
        tasks = (delayed(fitting_function)(model, X_train, y_train) for i in range(n_iterations))
        output = executor(tasks)
                
    else:
        output = [fitting_function(model, X_train, y_train) for i in range(n_iterations)]        
 
    # append output from 'joblib' in lists and DataFrames    
    results_2 = pd.Series() 
    
    # collect and save all CV results for plotting
    for counter in range(0,len(output)):
        
        # collect cross-validation results (e.g. multiple metrics etc.)
        results_1 = pd.Series() 
        results_1 = pd.Series(output[counter])
        results_2 = results_2.append(results_1, ignore_index=True) 
        
        # convert Series to a DataFrame
        results = pd.DataFrame(results_2, columns={0: 'best_score'})
        # rename
        results.rename(columns={0: 'best_score'}, inplace = True) 

    # print
    print('Number iterations: %.0f' % (len(output)))
    
    
##############################################################################
## GRADIENT BOOSTING PREDICTION FINAL MODEL (GOALS: 
## 1. check the prediction performance with the out-of-bag/test set; 
if analysis == 'plotting_prediction':
       
    # set hyperparameters
    param_grid_tmp = {'max_depth': [4],   
                      'learning_rate': [0.01], 
                      'n_estimators': [3500],
                      'verbosity': [0],
                      'objective': ['reg:squarederror'],
                      'booster': ['gbtree'], 
                      'tree_method':['auto'], 
                      'n_jobs':[1],
                      #'n_jobs':[int(round((cpu_count() - 1), 0))], # NOT WORKING 
                      # for fitting
                      'gamma' : [0.5], 
                      'min_child_weight': [2], 
                      'max_delta_step': [0],
                      'subsample': [0.9],  
                      'colsample_bytree': [0.9], 
                      'colsample_bylevel': [0.6], 
                      'colsample_bynode': [0.7], 
                      'scale_pos_weight': [1], 
                      'base_score': [0],
                      #'missing': ['None'],
                      'num_parallel_tree': [1],
                      'importance_type':['gain'],
                      'nthread': [1] 
                      #'nthread':[int(round((cpu_count() - 1), 0))]# WORKING!!!
                  }
    
    # convert dictionary to DataFrame   
    param_grid = pd.DataFrame(param_grid_tmp)

    # initialize model 
    model = xgb.XGBRegressor(max_depth = param_grid.loc[:,'max_depth'][0],
                             learning_rate = param_grid.loc[:,'learning_rate'][0],
                             n_estimators = param_grid.loc[:,'n_estimators'][0],
                             verbosity = param_grid.loc[:,'verbosity'][0],
                             objective = param_grid.loc[:,'objective'][0],
                             booster = param_grid.loc[:,'booster'][0],
                             tree_method = param_grid.loc[:,'tree_method'][0],
                             gamma = param_grid.loc[:,'gamma'][0],
                             min_child_weight = param_grid.loc[:,'min_child_weight'][0],
                             max_delta_step = param_grid.loc[:,'max_delta_step'][0],
                             subsample = param_grid.loc[:,'subsample'][0],
                             colsample_bytree = param_grid.loc[:,'colsample_bytree'][0],
                             colsample_bylevel = param_grid.loc[:,'colsample_bylevel'][0],
                             colsample_bynode = param_grid.loc[:,'colsample_bynode'][0],
                             scale_pos_weight = param_grid.loc[:,'scale_pos_weight'][0],
                             num_parallel_tree = param_grid.loc[:,'num_parallel_tree'][0],
                             importance_type = param_grid.loc[:,'importance_type'][0], 
                             nthread = param_grid.loc[:,'nthread'][0],
                             )
    
    # fit model. Note that train set + validation has been considered in the final model-
    final_model = model.fit(X_train, y_train)
    
    # loop across time-series (i.e. sensor number)
    for sensor in pd.unique(input_data_tmp.loc[:, 'sensor']):
        # predict class each instance belogs to
        y_predicted = final_model.predict(X_test_oob.loc[X_test_oob['sensor'] == sensor])
   
        # predict probability each instance belogs to a specific class
        #y_predicted_prob = final_model.predict(X_test_oob.loc[X_test_oob['sensor'] == sensor], 
        #output_margin = True)  
    
        # select univariate time-seres (i.e. sensor number)
        y_test_oob_index = X_test_oob.loc[X_test_oob['sensor'] == sensor].index
        y_test_oob_tmp = y_test_oob.iloc[y_test_oob_index] 
         
        # convenient step
        y_test_oob_numpy = y_test_oob_tmp.to_numpy()                  
    
        # compute RMSE metric                    
        rmse_metric = np.sqrt(mean_squared_error(y_test_oob_numpy, y_predicted))
        
        # RMSE metric
        print('RMSE %.4f' % (rmse_metric))
        
        # concatenate
        dict_tmp = {'true': y_test_oob_numpy, 'predicted': y_predicted, 
                    'sensor' : sensor, 'rmse' : rmse_metric}
        comparison_tmp = pd.DataFrame(dict_tmp)
        
        # plot test vs forecast
        prediction_plot = sns.lineplot(data = comparison_tmp.loc[:, ['true', 'predicted']], 
                                       legend = 'full') 
        
        # add title                   
        prediction_plot = plt.title('sensor = %.0f\n\
                              rmse = %.3f\n' % \
                              (comparison_tmp.loc[0, 'sensor'],  
                               comparison_tmp.loc[0, 'rmse'])) 
    
        # GENERATE FIGURES 
        date = str(datetime.datetime.now())        
        fig = prediction_plot.get_figure()
        fig.savefig(os.path.sep.join([BASE_DIR_OUTPUT, date[0:10] 
        + "_" + date[11:12] + "_" + date[14:15] + date[17:22] + ".jpg"])) 
        
        # close pic in order to avoid overwriting with previous pics
        fig.clf()
        
    
###############################################################################
## SAVE BEST SCORES IN A PANDAS DATAFRAME AND PLOT THEIR BOOTSTRAPPING 
## DISTRIBUTION 

# select action according to specific process's output   
if analysis == 'grid_search' or \
   analysis == 'bootstrapping' or \
   analysis == 'testing_prediction':  
    # save cleaned DataFrame as .csv file
    results.to_csv(os.path.sep.join([BASE_DIR_OUTPUT, 
                                         output_file_name_1]), index= False) 
    
    # select action according to specific process's output   
    if analysis == 'grid_search' or \
       analysis == 'bootstrapping':  
           
        # rename
        best_score_df.rename(columns={0: 'best_score'}, inplace = True) 
        
        # concatenate results
        #best_score_df = pd.DataFrame(best_score, columns=['best_score'])
        best_parameters_df = pd.DataFrame(best_parameters)
        summary_table = pd.concat([best_score_df, best_parameters_df], axis = 1)
        
        # save cleaned DataFrame as .csv file
        summary_table.to_csv(os.path.sep.join([BASE_DIR_OUTPUT, 
                                             output_file_name_2]), index= False)
        
    elif analysis == 'testing_prediction': 
        summary_table = results.copy()
           
        # in case you want to load the .csv with the best scores
        #summary_table = pd.read_csv(os.path.sep.join([BASE_DIR_OUTPUT, 
        #                                     output_file_name_2]))    
            
    # confidence intervals
    alpha = 0.95
    p = ((1.0-alpha)/2.0) * 100
    lower = max(0.0, np.percentile(summary_table.loc[:,'best_score'], p))
    p = (alpha+((1.0-alpha)/2.0)) * 100
    upper = min(1.0, np.percentile(summary_table.loc[:,'best_score'], p))
    median = np.median(summary_table.loc[:,'best_score'])
    median_parameters = summary_table.loc[summary_table['best_score'] == (round(median, 2))]
    print('%.1f confidence interval %.4f and %.4f' % (alpha*100, lower, upper))
    print('Median %.4f' % (median))
    print('Below best score (median) and parameters ')
    print(median_parameters)

    
"""
# plot scores and save plot
date = str(datetime.datetime.now())
sns_plot = sns.distplot(best_score_df, bins = 30)
#sns_plot = sns.distplot(best_score_df, bins = (len(best_score_df)/100))
fig = sns_plot.get_figure()
fig.savefig(os.path.sep.join([BASE_DIR, date[0:10]+ "_" + date[11:16]+".jpg"])) 
""" 

# shows execution time
print( time.time() - start_time, "seconds")







