# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 16:31:22 2019

@author: Dip
"""


import tensorflow as tf
from tensorflow.keras import models, layers, utils, regularizers, metrics, optimizers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import xgboost as xgb

from sklearn.model_selection import StratifiedKFold, train_test_split, RandomizedSearchCV
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import accuracy_score, auc, roc_curve, classification_report, roc_auc_score, confusion_matrix, recall_score, precision_score
from sklearn.preprocessing import LabelEncoder

from imblearn.combine import SMOTEENN, SMOTETomek

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier


from datetime import datetime

now = datetime.now()
f_name = str( now.strftime("%b_%d_%Y_%H%M%S") ) + '.txt'

f = open( f_name,'a')

def process_data( filename ):
    column_names = [
            'AGE',
            'MARITAL',
            'GENDER',
            'RACE',
            'FPIR',
            'EDULVL',
            'BMI',
            'WAIST',
            'FAMDHIST',
            'ALCOHOL',
            'MVPA',
            'SYST',
            'DIAS',
            #'SLPDUR',
            #'DPLVL',
            'SMOKE',
            'SMPLWGT',
            'LABEL'
    ]

    # Import training dataset
    dataset = pd.read_csv( filename, names=column_names, header = 0 )

    sample_weight = dataset[['SMPLWGT']].to_numpy()
    labels = dataset[['LABEL']].to_numpy()
    
    cat_features = dataset[['MARITAL','GENDER','RACE','EDULVL','FAMDHIST','ALCOHOL','SMOKE']]
    num_features = dataset[['AGE','FPIR','BMI','WAIST','MVPA','SYST','DIAS']].to_numpy()
    #num_features = dataset[['AGE','FPIR','BMI','WAIST','MVPA','SYST','DIAS','SLPDUR','DPLVL']].to_numpy()
    
    for col in cat_features:
       one_hot = pd.get_dummies( cat_features[ col ] )
       one_hot = one_hot.add_prefix( col )
       cat_features = cat_features.join( one_hot )
       cat_features = cat_features.drop( col, 1 )
    
    #cat_features = embed( cat_features )
   
    X = np.concatenate( [ cat_features, num_features ], axis = 1 )
    X = np.concatenate( [ X, sample_weight ], axis = 1 )
    Y = labels

    return X, Y 


def create_model( num_input, hidden_layer, num_classes ):

    input_layer = tf.keras.Input( shape = ( num_input, ) )
    hidden = layers.Dense( hidden_layer[0], 
                          kernel_regularizer=regularizers.l2(0.1),
                          activity_regularizer=regularizers.l1(0.1) )( input_layer )
    hidden = layers.ELU( alpha = 1 )( hidden )
    hidden = layers.Dropout( 0.1 )( hidden )
    hidden = layers.Dense( hidden_layer[1],
                          kernel_regularizer=regularizers.l2(0.1),
                          activity_regularizer=regularizers.l1(0.1))( hidden )
    hidden = layers.ELU( alpha = 1 )( hidden )
    hidden = layers.Dropout( 0.1 )( hidden )
    hidden = layers.Dense(  hidden_layer[2],
                          kernel_regularizer=regularizers.l2(0.1),
                          activity_regularizer=regularizers.l1(0.1))( hidden )
    hidden = layers.ELU( alpha = 1 )( hidden )  
    hidden = layers.Dropout( 0.1 )( hidden )
    if num_classes > 2:
        output_layer = layers.Dense( num_classes , activation = "softmax" )( hidden )
    else:
        output_layer = layers.Dense( 1 , activation = "sigmoid" )( hidden )

    model = models.Model( input_layer, output_layer )
    if num_classes > 2:
        loss_func = 'sparse_categorical_crossentropy'
    else:
        loss_func = 'binary_crossentropy'
    
    opt = optimizers.Adam(learning_rate=0.005, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(
        optimizer = opt,loss = loss_func,
        metrics = [ 'accuracy' ])
    
    return model


def create_wdmodel( n_cat_input, n_num_input, hidden_layer, num_classes ):
    
    input_layer1 = tf.keras.Input( shape = ( n_num_input, ) )
    hidden = layers.Dense( hidden_layer[0], 
                          kernel_regularizer=regularizers.l2(0.1),
                          activity_regularizer=regularizers.l1(0.1) )( input_layer1 )
    hidden = layers.ReLU(max_value=None, negative_slope=0.0, threshold=0.0)( hidden )
    hidden = layers.Dropout( 0.1 )( hidden )
    hidden = layers.Dense( hidden_layer[1],
                          kernel_regularizer=regularizers.l2(0.1),
                          activity_regularizer=regularizers.l1(0.1))( hidden )
    hidden = layers.ReLU(max_value=None, negative_slope=0.0, threshold=0.0)( hidden )
    hidden = layers.Dropout( 0.1 )( hidden )
    hidden = layers.Dense(  hidden_layer[2],
                          kernel_regularizer=regularizers.l2(0.1),
                          activity_regularizer=regularizers.l1(0.1))( hidden )
    hidden = layers.ReLU(max_value=None, negative_slope=0.0, threshold=0.0)( hidden )  
    hidden = layers.Dropout( 0.1 )( hidden )
    
    if num_classes > 2:
        output_layer1 = layers.Dense( num_classes , activation = "softmax" )( hidden )
    else:
        output_layer1 = layers.Dense( 1 , activation = "sigmoid" )( hidden )
    
    input_layer2 = tf.keras.Input( shape = ( n_cat_input, ) )
    output_layer2 = layers.ELU( alpha = 1 )( input_layer2 )
    
    inputs = []
    inputs.append( input_layer1 )
    inputs.append( input_layer2 )
    
    outputs = []    
    outputs.append( output_layer1 )
    outputs.append( output_layer2 )
    
    concat = layers.Concatenate()( outputs )
    if num_classes > 2:
        model_out = layers.Dense( num_classes , activation = "softmax" )( concat )
    else:
        model_out = layers.Dense( 1 , activation = "sigmoid" )( concat )
    #model_out = layers.Dense( 2, activation = "softmax" )( concat )
    model = models.Model( inputs, model_out )
    
    if num_classes > 2:
        loss_func = 'sparse_categorical_crossentropy'
    else:
        loss_func = 'binary_crossentropy'
    
    opt = optimizers.Adam(learning_rate=0.005, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(
        optimizer = opt,loss = loss_func,
        metrics = [ 'accuracy' ])
    
    return model


def run_cross_validation( X, Y, sample_weight ):
    
    nfolds = 10 
    skf = StratifiedKFold( n_splits = nfolds, shuffle = True )
    f.write( "\nDNN:\n" )

    num_fold = 0
    models = []
    best = 0
    best_score = 100
    ns = X.shape[0]
    ni = X.shape[1]
    no = 1
    alpha = 2
    nh = ns / ( alpha * ( ni + no ))
    hidden_layer = [ int( nh // 4 ), int( nh // 2 ), int( nh // 4 ) ]
    num_input = ni
    
    for ( train, valid ) in skf.split( X, Y):
        classes = np.unique(Y)
        num_classes = len( classes )
        model = create_model( num_input, hidden_layer,  num_classes )
        X_train, X_valid = X[train,], X[valid,]
        Y_train, Y_valid = Y[train], Y[valid]
        S_train, S_valid = sample_weight[train].reshape((len(train),)), sample_weight[valid].reshape((len(valid),))
        
        num_fold += 1
        print('\nStart StratifiedKFold number {} from {}'.format(num_fold, nfolds))
               
        callbacks = [
            EarlyStopping( monitor = 'val_loss', patience = 10, verbose = 2, min_delta = 0.01),
            ReduceLROnPlateau( monitor = 'val_loss', factor=0.2, verbose = 2, patience = 5, min_lr = 0.001, min_delta = 10 )            
        ]
               
        model_fit = model.fit(
                X_train, Y_train,
                batch_size = 16,
                epochs = 100,
                sample_weight = S_train,
                shuffle = True, 
                verbose = 2, 
                validation_data = ( X_valid, Y_valid, S_valid ),
                callbacks = callbacks
                )
        
        score = np.mean( model_fit.history['acc'] )
        
        if score < best_score:
            best = model
            best_score = score
        models.append(model)
    
    print('Best accuracy:', round( best_score, 3 ) )
    f.write('Avg accuracy:' + str( round( best_score, 3 ) ) )

    info_string = 'Accuracy: ' + str(best_score) + '_folds_' + str(nfolds)
    return info_string, models, best 

    
def run_cross_validation_wd( X, Y, sample_weight ):
    
    nfolds = 5 
    skf = StratifiedKFold( n_splits = nfolds, shuffle = True )
    f.write( "\nWide and Deep:\n" )
    
    num_fold = 0
    wdmodels = []
    best = 0
    best_score = 100
    ns = X.shape[0]
    ni = X[:,:11].shape[1]
    no = 1
    alpha = 2
    nh = ns / ( alpha * ( ni + no ))
    hidden_layer = [ int( nh // 4 ), int( nh // 2 ), int( nh // 4 ) ]
    n_cat_input = ni
    n_num_input = X[:,11:].shape[1]
    
    for ( train, valid ) in skf.split( X, Y):
        classes = np.unique(Y)
        num_classes = len( classes )
        model = create_wdmodel( n_cat_input, n_num_input, hidden_layer,  num_classes )
        X_train, X_valid = X[train,], X[valid,]
        Y_train, Y_valid = Y[train], Y[valid]
        S_train, S_valid = sample_weight[train].reshape((len(train),)), sample_weight[valid].reshape((len(valid),))
        
        num_fold += 1
        print('\nStart StratifiedKFold number {} from {}'.format(num_fold, nfolds))
               
        callbacks = [
            EarlyStopping( monitor = 'val_loss', patience = 10, verbose = 0, min_delta = 0.01),
            ReduceLROnPlateau( monitor = 'val_loss', factor=0.2, verbose = 0, patience = 5, min_lr = 0.001, min_delta = 10 )            
        ]
        
        model_fit = model.fit(
                [ X_train[:,11:], X_train[:,:11]], Y_train,
                batch_size = 16,
                epochs = 50,
                sample_weight = S_train,
                shuffle = True, 
                verbose = 1, 
                validation_data = ( [ X_valid[:,11:], X_valid[:,:11]], Y_valid, S_valid ),
                callbacks = callbacks
                )
        
        score = np.mean( model_fit.history['acc'] )
        
        if score < best_score:
            best = model
            best_score = score
            
        wdmodels.append(model)
    
    print('Best accuracy:', round( best_score, 3 ) )
    f.write('Avg accuracy:' + str( round( best_score, 3 ) ) )

    info_string = 'Accuracy: ' + str(best_score) + '_folds_' + str(nfolds)
    return info_string, wdmodels, best 


def make_binary_labels( labels, label ):
    
    for i in range( len ( labels )):
        if  labels[i] not in label:
            labels[i] = 0;
        else:
            labels[i] = 1;
    
    return labels
    
def run_models(x_train, x_test, y_train, y_test, sample_weight ):

    models = {
              'rf':RandomForestClassifier(),
              #'knn':KNeighborsClassifier(),
              'svm' :LinearSVC(),
              'dt':DecisionTreeClassifier(),
              'reg':LogisticRegression(),
              'gb':GradientBoostingClassifier()
              }
    for model_key in models:
        f.write( '\n' + model_key + ': ' )
        print( model_key, ':' )
        model = models[model_key]
        model.fit(x_train, y_train, sample_weight = sample_weight )
        '''
        f.write( '\nTraining:---')
        print( 'Training:---' )
        preds = model.predict( x_train )
        evaluate_performance( y_train, preds )
        '''
        f.write( '\nTesting:---')
        #print( 'Testing:---' )
        preds = model.predict(x_test)
        evaluate_performance( y_test, preds )
        
def embed( cat_features ):
    n_samples = len( cat_features )
    cat_feat = np.empty(( n_samples, 0 ))
    features = list( cat_features )
    le = LabelEncoder()
    for feature in features:
        output_emb = int( cat_features[feature].nunique() / 2 )
        input_emb =  cat_features[feature].nunique()
        f_transformed = le.fit_transform( cat_features[feature] )
        
        model = models.Sequential()
        model.add( layers.Embedding( input_emb, output_emb, input_length = 1 ))
        model.compile( 'rmsprop', 'mse' )
        
        feat = model.predict( f_transformed ).reshape( n_samples, output_emb )
        
        cat_feat = np.hstack(( cat_feat, feat ))   
    
    return cat_feat

def run_xgboost( x_train, x_test, y_train, y_test, sample_weight ):
    '''
    if len(np.unique( y_train )) > 2:
        xgb_model = xgb.XGBClassifier(objective="multi:softprob", random_state=96)
    else:
        xgb_model = xgb.XGBClassifier(n_estimators = 50, objective="binary:logistic")
    eval_set = [(x_train, y_train), (x_test, y_test)]
    eval_metric = ["auc","error"]
    xgb_model.fit( x_train, y_train, eval_set = eval_set, eval_metric = eval_metric, sample_weight = sample_weight, verbose = False )
    xgb_model.save_model("xgb_model.h5")
    print( '\nXGBoost:' )
    f.write( '\nXGBoost:' )
    f.write( '\nTesting:---')
    #print( 'Testing:---' )    
    y_pred = xgb_model.predict( x_test )
    evaluate_performance( y_test, y_pred )
    #xgb_model.evals_result()
    #print(xgb_model.evals_result())
    #xgb_model.to_graphviz(bst, num_trees=2)
    '''
    
    # create a default XGBoost classifier
    model = xgb.XGBClassifier(
        random_state=96, 
        #tree_method = "gpu_hist",
        eval_metric=["error", "auc"]
    )
    # Create the grid search parameter grid and scoring funcitons
    param_grid = {
        "learning_rate": [0.1, 0.01, 0.05, 0.5],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "subsample": [0.6, 0.8, 1.0],
        "max_depth": [2, 3, 4, 5, 10],
        "n_estimators": [100, 200, 300, 400, 500],
        "reg_lambda": [1, 1.5, 2],
        "gamma": [0, 0.1, 0.3, 0.5],
    }
    scoring = {
        'AUC': 'roc_auc', 
        #'Accuracy': make_scorer(accuracy_score)
    }
    # create the Kfold object
    num_folds = 10
    kfold = StratifiedKFold(n_splits=num_folds, random_state=96)
    # create the grid search object
    n_iter=50
    grid = RandomizedSearchCV(
        estimator=model, 
        param_distributions=param_grid,
        cv=kfold,
        scoring=scoring,
        n_jobs=-1,
        n_iter=n_iter,
        refit="AUC",
    )
    # fit grid search
    optimal_model = grid.fit(X_train,y_train)
    params = optimal_model.best_params_
    print( optimal_model.best_score_ )
    
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
    eval_metric = ["auc","error"]
    best_model.fit( x_train, y_train, eval_set = eval_set, eval_metric = eval_metric, sample_weight = sample_weight, verbose = False )
    
    y_pred = best_model.predict( x_test )
    evaluate_performance( y_test, y_pred )
    
    return best_model
    
        
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
        f.write( '\nConfusion matrix:\n' )
        f.write( str(confusion_matrix( Y, pred )) )
        sensitivity = tp / ( tp + fn )
        specificity = tn / ( fp + tn )
        print( 'Sensitivity:', round( recall_score( Y, pred, average = average), 3 ))
        f.write( '\nSensitivity:\t' + str( round( sensitivity, 3 ) ))
        print( 'Specificity:', round( specificity, 3 ) )
        print( 'Precision:', round( precision_score( Y, pred ), 3 ))
        f.write( '\nSpecificity:\t' + str( round( specificity, 3 ) ))
        f.write( '\nPrecision:\t' + str( round(precision_score( Y, pred ), 3 ) ))
        fpr, tpr, th = roc_curve( Y, pred )
        #auc_score = auc( fpr, tpr )
        #print( 'AUC score:', round( auc_score, 3 ))
        #f.write( '\nAUC score:\t' + str( round( auc_score, 3 )) + '\n' )
        print( 'ROAUC score:', round( roc_auc_score( Y, pred, average = 'weighted' ), 3 ))
        f.write( '\nROAUC score:\t' + str( round( roc_auc_score( Y, pred, average = 'weighted' ), 3 )) + '\n' )

    print( 'Accuracy:', round( accuracy_score( Y, pred ), 3 ), '\n')
    f.write( 'Accuracy:\t' + str( round( accuracy_score( Y, pred ), 3 )) + '\n' )

        
def remove_label( X, Y, label ):
    data = np.hstack( ( X, Y ) )
    data = data[ np.logical_not( data[:,-1] == label ) ]
    return data[:,:-1], data[:,-1]

        
if __name__ == '__main__':
    
    print('\nStarting.....\n') 
    filename = 'complete_data.csv'
    #filename = 'complete_data_6_waves.csv'
    f.write( "\n--------------------------------------------------------------------------" )
    f.write( '\n\nNew run:\nFilename:' + filename + '\n' )
    X, Y = process_data( filename )
    
    #X, Y = remove_label( X, Y, 3 )
    
    label = [ 3, 2 ]
    Y = make_binary_labels( Y, label )
    
    f.write( 'Total samples: ' + str( X.shape[0] ) + '\n' )
    print( 'Total samples: ' + str( X.shape[0] ) + '\n' )
    unique, counts = np.unique( Y, return_counts=True)
    unique = [ int( u ) for u in unique ]
    classes = dict(zip( unique, counts ))
    print( 'Classes:' + str( classes ) + '\n' )
    f.write( 'Classes:' + str( classes ) + '\n' )

    X_train, X_test, Y_train, Y_test = train_test_split( X, Y, stratify = Y )    
    print( 'Train:', X_train.shape[0], '\nTest:', X_test.shape[0], '\n' )

    smote_enn = SMOTEENN( random_state = 1234 )
    smote_tomek = SMOTETomek( random_state = 1234 )
    #X_resampled, Y_resampled = smote_enn.fit_resample( X_train, Y_train.ravel() )
    X_resampled, Y_resampled = smote_tomek.fit_resample( X_train, Y_train.ravel() )
    Y_train = Y_resampled
    #'''            
    f.write( 'Resampled: ' + str( X_resampled.shape[0] ) + '\n' )
    print( 'Resampled: ' + str( X_resampled.shape[0] ) + '\n' )
    unique, counts = np.unique( Y_resampled, return_counts=True)
    unique = [ int( u ) for u in unique ]
    classes = dict(zip( unique, counts ))
    print( 'Classes:', classes, '\n' )       
    f.write( 'Classes:' + str( classes ) + '\n' )
    X_train = X_resampled
    #'''
    
    #label = [ 1, 2 ]
    #Y_train = make_binary_labels( Y_resampled, label )
    #Y_test = make_binary_labels( Y_test, label )
    classes = [ 'Normal', 'Prediabetes', 'Undiagnosed', 'Diabetes' ]
    f.write( "\nClassification: " + str( [ classes[i] for i in label ] ) + " vs others \n" )
    
    sample_weight = X_train[:,-1]
    X_train = X_train[:,:-1]
    test_wgt = X_test[:,-1]
    X_test = X_test[:,:-1]
    
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
    n_train_col = X_train.shape[1]
    n_test_col = X_test.shape[1]
    
    '''
    run_models( X_train, X_test, Y_train.reshape((n_train,)), Y_test.reshape((n_test,)), sample_weight )
    best_model = run_xgboost( X_train, X_test, Y_train.reshape((n_train,)), Y_test.reshape((n_test,)), sample_weight )
    
    feature_importance = best_model.feature_importances_
    evaluations = best_model.evals_result_
    
    column_names = [
            'MARRIED','WIDOWED','DIVORCED','SEPARATED','NEVER MARRIED','LIVING WITH PARTNER',
            'MALE','FEMALE',
            'MEXICAN AMERICAN','OTHER HISPANIC','NON-HISPANIC WHITE','NON-HISPANIC BLACK','OTHER RACE',
            'EDULVL HIGH','EDULVL MEDIUM','EDULVL LOW',
            'FAM D HIST NO','FAM D HIST YES',
            'ALC ABSTAINERS','ALC HEAVY','ALC MODERATE','ALC OCCASIONAL',
            'SMK CURRENT','SMK EX','SMK NEVER',
            'AGE','FPIR','BMI','WAIST','MVPA','SYST','DIAS','SLPDUR','DPLVL'
    ]
    
    imp = zip( column_names, feature_importance )
    for ( feature, imp_score ) in imp:
        print( feature, ":", imp_score )
        
    print( "Average AUC:", np.average( np.array( evaluations['validation_1']['auc'] )) )
    f.write( "XGBoost:\nAverage AUC: " + str(np.average( np.array( evaluations['validation_1']['auc'] ))) )
    '''
    '''
    info, model_all, best_model = run_cross_validation_wd( X_train, Y_train, sample_weight )
    #info, model_all, best_model = run_cross_validation( X_train, Y_train, sample_weight )
    
    callbacks = [
        EarlyStopping( monitor = 'val_loss', patience = 10, verbose = 2, min_delta = 0.01 ),
        ReduceLROnPlateau( monitor = 'val_loss', factor = 0.2, verbose = 2, patience = 5, min_lr = 0.001, min_delta = 10 )            
    ]
    
    print( best_model.summary() )
    
    history = best_model.fit( 
            #X_train, Y_train, 
            [ X_train[:,11:], X_train[:,:11]], Y_train, 
            sample_weight = sample_weight.reshape(( n_train, )),
            epochs = 100,
            verbose = 1,
            shuffle = True,
            validation_split = 0.2,
            callbacks = callbacks,
            )

    plt.plot(history.history['acc'], label='accuracy')
    plt.plot(history.history['val_acc'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    #savefile = 'accuracy_plot_' + str(lbl) + '.png'
    #plt.savefig( savefile )
    
    plt.clf()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss']) 
    plt.title('model train vs validation loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    #savefile = 'loss_plot_' + str(lbl) + '.png'
    #plt.savefig( savefile )
   
    Y_pred = best_model.predict( [ X_test[:,11:], X_test[:,:11]] )
    #Y_pred = best_model.predict( X_test )
     
    if len(np.unique( Y_train )) > 2:
        prob = list( Y_pred )
        pred = []
        for p in prob:
            pred.append(list(p).index( max(p)))
    else:
        pred = []
        for p in Y_pred:
            v = 1 if p > 0.5 else 0
            pred.append( v )
    evaluate_performance( Y_test, pred )
    '''
    f.close()
    
