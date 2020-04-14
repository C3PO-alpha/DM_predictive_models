# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.keras import models, layers, utils, regularizers, metrics, optimizers, losses, Input
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from kerastuner.tuners import RandomSearch

import shap
import xgboost as xgb

from sklearn.model_selection import StratifiedKFold, train_test_split, RandomizedSearchCV, GridSearchCV
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import accuracy_score, auc, roc_curve, classification_report, roc_auc_score, confusion_matrix, recall_score, precision_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

from imblearn.combine import SMOTEENN, SMOTETomek

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import DBSCAN
import shap

import numba
from datetime import datetime

now = datetime.now()
f_name = str( now.strftime("%b_%d_%Y_%H%M%S") ) + '.txt'

f = open( f_name,'a')

def process_data( dataset, cat_feature_names, num_feature_names ):

    sample_weight = dataset.loc[:,['SMPLWGT']]
    labels = dataset.loc[:,['LABEL']]
    
    cat_features = dataset.loc[:,cat_feature_names]
    num_features = dataset.loc[:,num_feature_names]

    for col in cat_features:
       one_hot = pd.get_dummies( cat_features[ col ] )
       one_hot = one_hot.add_prefix( col )
       cat_features = cat_features.join( one_hot )
       cat_features = cat_features.drop( col, 1 )
       
    mms = MinMaxScaler( copy = False )
    num_features = pd.DataFrame( mms.fit_transform( num_features ), columns =num_features.columns, index = num_features.index )
   
    Y = labels
    X = pd.concat( [ cat_features, num_features ], axis = 1 )
    X = pd.concat( [ X, sample_weight ], axis = 1 )

    return X, Y 

def make_binary_labels( labels, label ):
    
    if isinstance( labels, pd.DataFrame ) | isinstance( labels, pd.Series ):
        idx = labels.index
        labels = labels.to_numpy()
    
    for i in range( len ( labels )):
        if  labels[i] not in label:
            labels[i] = 0;
        else:
            labels[i] = 1;
    
    labels = pd.DataFrame( labels, index = idx, columns = [ 'LABEL' ] )
    
    return labels
    
def run_models(x_train, x_test, y_train, y_test, sample_weight ):
    now = datetime.now()
    f_name = str( now.strftime("%b_%d_%Y_%H%M%S") ) + '.txt'

    f = open( f_name,'a')
    m = {
              'rf':RandomForestClassifier(),
              'svm' :LinearSVC(),
              'dt':DecisionTreeClassifier(),
              'reg':LogisticRegression(),
              'gb':GradientBoostingClassifier()
              }
    for model_key in m:
        f.write( '\n' + model_key + ': ' )
        print( model_key, ':' )
        model = m[model_key]
        model.fit(x_train, y_train, sample_weight = sample_weight )

        preds = model.predict(x_test)
        evaluate_performance( y_test, preds )
    f.close()
        
@numba.jit
def run_xgboost( x_train, y_train, sample_weight, x_test, y_test, test_weight, scheme ):

    model = xgb.XGBClassifier(silent=False, 
                          scale_pos_weight=1,
                          objective='binary:logistic'
                          )
    if scheme == 1:
    # scheme A
        param_grid = {
            "learning_rate": [0.1, 0.01, 0.05, 0.09],
            "colsample_bytree": [ 0.8, 1.0],
            "subsample": [0.8, 1.0],
            "max_depth": [3, 5, 8],
            "n_estimators": [200, 400, 500],
            "reg_lambda": [1, 1.5, 2],
            "gamma": [0, 0.1, 0.3, 0.5],
        }
    elif scheme == 2:
    # scheme B
        param_grid = {
            "learning_rate": [0.1, 0.01, 0.05, 0.5],
            "colsample_bytree": [0.5, 0.8, 1.0],
            "subsample": [0.6, 0.8, 1.0],
            "max_depth": [3, 6, 9, 12, 15],
            "n_estimators": [100, 200, 300, 400, 500, 800],
            "reg_lambda": [1, 1.5, 2],
            "gamma": [0, 0.1, 0.3, 0.5],
        }

    scoring = {
        'AUC': 'roc_auc',
        'Accuracy': 'accuracy'
    }
    
    # create the Kfold object
    num_folds = 10
    kfold = StratifiedKFold( n_splits = num_folds, random_state = 96, shuffle = True )
    
    # create the grid search object
    n_iter = 50
    grid = RandomizedSearchCV(
        estimator=model, 
        param_distributions=param_grid,
        cv=kfold,
        scoring=scoring,
        n_jobs=10,
        refit="AUC",
        n_iter = n_iter,
        return_train_score = True,
        verbose = True
    )
    
    # fit grid search
    optimal_model = grid.fit( X = x_train, y = y_train, sample_weight = sample_weight )
    params = optimal_model.best_params_
    print( "\nBest score: ", optimal_model.best_score_ )
    print( "\n", params )
    best_model = xgb.XGBClassifier(
        colsample_bytree = params['colsample_bytree'],
        gamma = params['gamma'],
        learning_rate = params['learning_rate'],
        max_depth = params['max_depth'],
        n_estimators = params['n_estimators'],
        reg_lambda = params['reg_lambda'],
        subsample = params['subsample']
        )
    eval_set = [(x_train, y_train), (x_test, y_test)]
    eval_metric = ["error","auc"]
    best_model.fit(
        X = x_train, 
        y = y_train, 
        sample_weight = sample_weight,
        eval_metric = eval_metric,
        sample_weight_eval_set = [ sample_weight, test_weight ],
        eval_set = eval_set, 
        early_stopping_rounds = 50,
        verbose = True )

    y_pred = best_model.predict( x_test )
    accuracy = accuracy_score( y_test.iloc[:,0], y_pred, sample_weight = test_weight.iloc[:,0] )
    print("\nAccuracy: %.3f%%" % (accuracy * 100.0))
    f.write( "\nAccuracy: " + str( accuracy * 100.0 ))
    evaluate_performance( y_test, y_pred )

    return best_model, optimal_model
    
        
def evaluate_performance( Y, pred ):
    if len(np.unique( Y )) > 2:
        average = 'weighted'
        print( classification_report( Y, pred ))
    else:
        average = 'binary'
        tn, fp, fn, tp = confusion_matrix( Y, pred ).ravel()
        print( 'Confusion Matrix:' )
        print( "tp:", tp, "\t fp:", fp )
        print( "fn:", fn, "\t tn:", tn )
        specificity = tn / ( fp + tn )
        print( 'Sensitivity:', round( recall_score( Y, pred, average = average), 4 ))
        print( 'Specificity:', round( specificity, 4 ) )
        print( 'Precision:', round( precision_score( Y, pred ), 4 ))
        fpr, tpr, th = roc_curve( Y, pred )
        print( 'ROAUC score:', round( roc_auc_score( Y, pred, average = 'weighted' ), 4 ))
        
        f.write( '\nConfusion matrix:\n' )
        f.write( str(confusion_matrix( Y, pred )) )
        f.write( '\nSpecificity:\t' + str( round( specificity, 3 ) ))
        f.write( '\nPrecision:\t' + str( round(precision_score( Y, pred ), 3 ) ))        
        f.write( '\nROAUC score:\t' + str( round( roc_auc_score( Y, pred, average = 'weighted' ), 3 )) + '\n' )

    print( 'Accuracy:', round( accuracy_score( Y, pred ), 4 ), '\n')
    f.write( 'Accuracy:\t' + str( round( accuracy_score( Y, pred ), 3 )) + '\n' )

        
def remove_label( X, Y, label ):
    if isinstance( X, np.ndarray ):
        data = np.hstack( ( X, Y ) )
        data = data[ np.logical_not( data[:,-1] == label ) ]
        return data[:,:-1], data[:,-1]
    elif isinstance( X, pd.DataFrame ):
        indexNames = Y[ Y['LABEL'] == label ].index
        X.drop( indexNames, inplace = True )
        Y.drop( indexNames, inplace = True )
        return X, Y
    else:
        return None, None
    
def process_split( X, Y, choice ):
    if choice == 1:
        label = [ 2, 3 ]
    elif choice == 2:
        X, Y = remove_label( X, Y, 3 )
        label = [ 1, 2 ]
    
    Y = make_binary_labels( Y, label )    
    print( 'Total samples: ' + str( X.shape[0] ) + '\n' )
    unique, counts = np.unique( Y, return_counts=True)
    unique = [ int( u ) for u in unique ]
    classes = dict(zip( unique, counts ))
    print( 'Classes:' + str( classes ) + '\n' )
    
    f.write( 'Total samples: ' + str( X.shape[0] ) + '\n' )
    f.write( 'Classes:' + str( classes ) + '\n' )

    X_train, X_test, Y_train, Y_test = train_test_split( X, Y, stratify = Y )    
    print( 'Train:', X_train.shape[0], '\nTest:', X_test.shape[0], '\n' ) 
    
    if choice == 1:
        smote_tomek = SMOTETomek( random_state = 1234 )
        X_resampled, Y_resampled = smote_tomek.fit_resample( X_train, Y_train.iloc[:,0] )
        X_train = X_resampled
        Y_train = Y_resampled
        print( 'Resampled: ' + str( X_resampled.shape[0] ) + '\n' )
        unique, counts = np.unique( Y_resampled, return_counts=True)
        unique = [ int( u ) for u in unique ]
        classes = dict(zip( unique, counts ))
        print( 'Classes:', classes, '\n' )
    
    sample_weight = X_train.loc[:,['SMPLWGT']]
    X_train = X_train.drop( columns = ['SMPLWGT'] )
    test_weight = X_test.loc[:,['SMPLWGT']]
    X_test = X_test.drop( columns = ['SMPLWGT'] )
    
    sample_weight.loc[(sample_weight.index < 21000), 'SMPLWGT' ] *= 2
    sample_weight /= 9                                                      # 9 waves combined
        
    return ( X_train, Y_train, sample_weight, X_test, Y_test, test_weight )
    
def process_split_run( X, Y, choice ):
    if choice == 1:
        label = [ 2, 3 ]
    elif choice == 2:
        X, Y = remove_label( X, Y, 3 )
        label = [ 1, 2 ]
    
    Y = make_binary_labels( Y, label )    
    print( 'Total samples: ' + str( X.shape[0] ) + '\n' )
    unique, counts = np.unique( Y, return_counts=True)
    unique = [ int( u ) for u in unique ]
    classes = dict(zip( unique, counts ))
    print( 'Classes:' + str( classes ) + '\n' )
    
    f.write( 'Total samples: ' + str( X.shape[0] ) + '\n' )
    f.write( 'Classes:' + str( classes ) + '\n' )

    X_train, X_test, Y_train, Y_test = train_test_split( X, Y, stratify = Y )    
    print( 'Train:', X_train.shape[0], '\nTest:', X_test.shape[0], '\n' ) 
    
    if choice == 1:
        smote_tomek = SMOTETomek( random_state = 1234 )
        X_resampled, Y_resampled = smote_enn.fit_resample( X_train, Y_train.ravel() )
        X_resampled, Y_resampled = smote_tomek.fit_resample( X_train, Y_train.iloc[:,0] )
        X_train = X_resampled
        Y_train = Y_resampled
        print( 'Resampled: ' + str( X_resampled.shape[0] ) + '\n' )
        unique, counts = np.unique( Y_resampled, return_counts=True)
        unique = [ int( u ) for u in unique ]
        classes = dict(zip( unique, counts ))
        print( 'Classes:', classes, '\n' )
        f.write( 'Resampled: ' + str( X_resampled.shape[0] ) + '\n' )
        f.write( 'Classes:' + str( classes ) + '\n' )
    
    sample_weight = X_train.loc[:,['SMPLWGT']]
    X_train = X_train.drop( columns = ['SMPLWGT'] )
    test_weight = X_test.loc[:,['SMPLWGT']]
    X_test = X_test.drop( columns = ['SMPLWGT'] )
    
    np.save( "X_train" + str( choice ) + ".npy", X_train )
    np.save( "Y_train" + str( choice ) + ".npy", Y_train )
    np.save( "sample_weight" + str( choice ) + ".npy", sample_weight )
    np.save( "X_test" + str( choice ) + ".npy", X_test )
    np.save( "Y_test" + str( choice ) + ".npy", Y_test )
    np.save( "test_wgt" + str( choice ) + ".npy", test_weight )    

    best_model, optimal_model = run_xgboost( X_train, Y_train, sample_weight, X_test, Y_test, test_weight, choice )
     
    feature_importance = best_model.feature_importances_
    evaluations = best_model.evals_result_
    
    imp = zip( X_train.columns, feature_importance )
    for ( feature, imp_score ) in imp:
        print( feature, ":", imp_score )
        
    print( "Average AUC:", np.average( np.array( evaluations['validation_1']['auc'] )) )
    
    return best_model, optimal_model

def xgboost_select( X_train, Y_train, sample_weight, X_test, Y_test, test_weight, choice ):
    best_model, optimal_model = run_xgboost( X_train, Y_train, sample_weight, X_test, Y_test, test_weight, choice )
     
    feature_importance = best_model.feature_importances_
    evaluations = best_model.evals_result_
    
    imp = zip( X_train.columns, feature_importance )
    for ( feature, imp_score ) in imp:
        print( feature, ":", imp_score )
        
    print( "Average AUC:", np.average( np.array( evaluations['validation_1']['auc'] )) )
    
    return best_model, optimal_model

def plot_fig( data, exp, shap_values, choice ):
    
    # Shap summary plot
    shap.summary_plot( shap_values, data, feature_names = data.columns, show = False )
    shap.summary_plot( shap_values, data, feature_names = data.columns, plot_type = 'bar' )
    
    # Shap interaction plot
    shap_interaction_values = exp.shap_interaction_values(data)    
    tmp = np.abs(shap_interaction_values).sum(0)
    for i in range(tmp.shape[0]):
        tmp[i,i] = 0
    inds = np.argsort(-tmp.sum(0))[:50]
    tmp2 = tmp[inds,:][:,inds]
    plt.figure(figsize=(12,12))
    plt.imshow(tmp2)
    plt.yticks(range(tmp2.shape[0]), data.columns[inds], rotation=50.4, horizontalalignment="right")
    plt.xticks(range(tmp2.shape[0]), data.columns[inds], rotation=50.4, horizontalalignment="left")
    plt.gca().xaxis.tick_top()
    plt.show()
    
    # Shap dependence plot
    for col in data.columns:
        shap.dependence_plot( col, shap_values, data, data.columns, show = False )
    
    shap.initjs()
    shap.force_plot( exp.expected_value, shap_values[0], data[0].iloc[0,:] )
    
    shap.decision_plot( exp.expected_value, shap_values[0], data[0].iloc[0,:] )

if __name__ == '__main__':
    
    print('\nStarting.....\n') 
    filename = 'raw_data.csv'
    f.write( "\n--------------------------------------------------------------------------" )
    f.write( '\nFilename:' + filename + '\n' )
    
    data = pd.read_csv( filename, index_col = "SEQN" )
    
    datasets = {}
    datasets['data9918_14'] = data.drop( columns = [ "DPLVL", "SLPDUR", "ALCOHOL", "LDL" ]).copy()
    datasets['data9916_15'] = data.drop( columns = [ "DPLVL", "SLPDUR", "ALCOHOL" ]).copy()
    datasets['data0518_16'] = data.drop( columns = [ "ALCOHOL", "LDL" ]).copy()
    datasets['1'] = data.drop( columns = [ "DPLVL", "SLPDUR", "ALCOHOL", "LDL", "HDL" ]).copy()
    datasets['2'] = data.drop( columns = [ "DPLVL", "SLPDUR", "LDL", "HDL" ]).copy()
    datasets['3'] = data.drop( columns = [ "ALCOHOL", "LDL", "HDL" ]).copy()
    
    datasets['data9918_14'] = datasets['data9918_14'].dropna( axis = 0, how = 'any' )
    datasets['data9916_15'] = datasets['data9916_15'].dropna( axis = 0, how = 'any' )
    datasets['data0518_16'] = datasets['data0518_16'].dropna( axis = 0, how = 'any' )
    datasets['1'] = datasets['1'].dropna( axis = 0, how = 'any' )
    datasets['2'] = datasets['2'].dropna( axis = 0, how = 'any' )
    datasets['3'] = datasets['3'].dropna( axis = 0, how = 'any' )
    
    data_A = {}
    data_A['data9918_14'] = process_data( datasets['data9918_14'].copy(), ['MARITAL', 'GENDER', 'RACE', 'EDULVL', 'FAMDHIST', 'SMOKING'], ['AGE', 'FPIR', 'BMI', 'WAIST', 'MVPA', 'SYST', 'DIAS', 'HDL'] )
    data_A['data9916_15'] = process_data( datasets['data9916_15'].copy(), ['MARITAL', 'GENDER', 'RACE', 'EDULVL', 'FAMDHIST', 'SMOKING'], ['AGE', 'FPIR', 'BMI', 'WAIST', 'MVPA', 'SYST', 'DIAS', 'HDL', 'LDL'] )
    data_A['data0518_16'] = process_data( datasets['data0518_16'].copy(), ['MARITAL', 'GENDER', 'RACE', 'EDULVL', 'FAMDHIST', 'SMOKING'], ['AGE', 'FPIR', 'BMI', 'WAIST', 'MVPA', 'DPLVL', 'SLPDUR', 'SYST', 'DIAS', 'HDL'] )
    data_A['1'] = process_data( datasets['1'].copy(), ['MARITAL', 'GENDER', 'RACE', 'EDULVL', 'FAMDHIST', 'SMOKING' ], ['AGE', 'FPIR', 'BMI', 'WAIST', 'MVPA', 'SYST', 'DIAS' ] )
    data_A['2'] = process_data( datasets['2'].copy(), ['MARITAL', 'GENDER', 'RACE', 'EDULVL', 'FAMDHIST', 'SMOKING', 'ALCOHOL'], ['AGE', 'FPIR', 'BMI', 'WAIST', 'MVPA', 'SYST', 'DIAS' ] )
    data_A['3'] = process_data( datasets['3'].copy(), ['MARITAL', 'GENDER', 'RACE', 'EDULVL', 'FAMDHIST', 'SMOKING'], ['AGE', 'FPIR', 'BMI', 'WAIST', 'MVPA', 'SYST', 'DIAS', 'DPLVL', 'SLPDUR' ] )
    
    best_model_A = {}
    optimal_model_A = {}
    for key in data_A.keys():
        f.write( "\n\nData: " + key + "\n" )
        dt = process_split( data_A[key][0].copy(), data_A[key][1].copy(), 1 )
        best_model_A[key], optimal_model_A[key] = xgboost_select( dt[0], dt[1], dt[2], dt[3], dt[4], dt[5], 1 )
    np.save( "best_model_A3.npy", best_model_A )
    np.save( "optimal_model_A3.npy", optimal_model_A )
    
    key = '2'                                                               # key should be specific to the model
    data_1 = process_split( data_A[key][0], data_A[key][1], 1 )

    model_1 = best_model_A[key]
    exp_1 = shap.TreeExplainer( model_1 )
    shap_values_1 = exp_1.shap_values( dt[0], dt[1] )
    np.save( "shap_1.npy", shap_values_1 )  
    plot_fig( dt[0], exp_1, shap_values_1, 1 )
    
    data_B = {}
    data_B['data9918_14'] = process_data( datasets['data9918_14'].copy(), ['MARITAL', 'GENDER', 'RACE', 'EDULVL', 'FAMDHIST', 'SMOKING'], ['AGE', 'FPIR', 'BMI', 'WAIST', 'MVPA', 'SYST', 'DIAS', 'HDL'] )
    data_B['data9916_15'] = process_data( datasets['data9916_15'].copy(), ['MARITAL', 'GENDER', 'RACE', 'EDULVL', 'FAMDHIST', 'SMOKING'], ['AGE', 'FPIR', 'BMI', 'WAIST', 'MVPA', 'SYST', 'DIAS', 'HDL', 'LDL'] )
    data_B['data0518_16'] = process_data( datasets['data0518_16'].copy(), ['MARITAL', 'GENDER', 'RACE', 'EDULVL', 'FAMDHIST', 'SMOKING'], ['AGE', 'FPIR', 'BMI', 'WAIST', 'MVPA', 'DPLVL', 'SLPDUR', 'SYST', 'DIAS', 'HDL'] )
    data_B['1'] = process_data( datasets['1'].copy(), ['MARITAL', 'GENDER', 'RACE', 'EDULVL', 'FAMDHIST', 'SMOKING'], ['AGE', 'FPIR', 'BMI', 'WAIST', 'MVPA', 'SYST', 'DIAS' ] )
    data_B['2'] = process_data( datasets['2'].copy(), ['MARITAL', 'GENDER', 'RACE', 'EDULVL', 'FAMDHIST', 'SMOKING', 'ALCOHOL'], ['AGE', 'FPIR', 'BMI', 'WAIST', 'MVPA', 'SYST', 'DIAS' ] )
    data_B['3'] = process_data( datasets['3'].copy(), ['MARITAL', 'GENDER', 'RACE', 'EDULVL', 'FAMDHIST', 'SMOKING'], ['AGE', 'FPIR', 'BMI', 'WAIST', 'MVPA', 'SYST', 'DIAS', 'DPLVL', 'SLPDUR' ] )
        
    best_model_B = {}
    optimal_model_B = {}
    for key in data_B.keys():
        f.write( "\n\nData: " + key + "\n" )
        best_model_B[key], optimal_model_B[key] = process_split_run( data_B[key][0].copy(), data_B[key][1].copy(), 2 )
    np.save( "best_model_B3.npy", best_model_B )
    np.save( "optimal_model_B3.npy", optimal_model_B )
    
    m = np.ndarray((2,3), dtype = object)
    i = 0
    for key in optimal_model_A.keys():
        m[0,i] = optimal_model_A[key]
        m[1,i] = optimal_model_B[key]
        i += 1
    for i in range( 1 ):
        print( "Best accuracy: ", m[0,i].cv_results_['mean_test_Accuracy'][m[0,i].best_index_] )
        print( "std accuracy: ", m[0,i].cv_results_['std_test_Accuracy'][m[0,i].best_index_] )
        print( "Best AUC: ", m[0,i].cv_results_['mean_test_AUC'][m[0,i].best_index_] )
        print( "std AUC: ", m[0,i].cv_results_['std_test_AUC'][m[0,i].best_index_] )
        
    for i in range( 1 ):
        print( "Best accuracy: ", m[1,i].cv_results_['mean_test_Accuracy'][m[1,i].best_index_] )
        print( "Best AUC: ", m[1,i].cv_results_['mean_test_AUC'][m[1,i].best_index_] )
  
    key = '2'                                                               # key should be specific to the model
    data_2 = process_split( data_B[key][0], data_B[key][1], 2 )
        
    model_2 = np.load( "best_model_B3.npy", allow_pickle = True ).item()
    exp_2 = shap.TreeExplainer( model_2[key] )
    shap_values_2 = exp_2.shap_values( data_2[0], data_2[1] )
    np.save( "shap_2.npy", shap_values_2 ) 
    plot_fig( data_2[0], exp_2, shap_values_2, 2 )
    
    f.close()


