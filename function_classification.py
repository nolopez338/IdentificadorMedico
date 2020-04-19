
################################################################
###################### SETUP ###################################
################################################################

# Import packages
from random import sample
from random import seed

from itertools import chain

import numpy as np
import pandas as pd

################################################################
###################### TEST/TRAINING SETS ######################
################################################################

# Extracts data for training keeping detection proportions
def get_proportional_subset(X, Y, n , sd = 1):
        
    # Set seed
    seed(sd)
    
    # Divide data accordingo to its class
    X_tot0 = [X[i] for i in range(len(Y)) if not Y[i]]
    X_tot1 = [X[i] for i in range(len(Y)) if Y[i]]

    # Basic information
    ones = sum(Y) # number of 1's
    zeros = len(Y) - ones # number of 0's
    prop = ones/len(Y) # proportion of 1's in the total

    # numbers of samples of each class to extract
    n1 = int(round(prop*n) + 1)
    n0 = int(n - n1)    
    
    # Indexes to extract
    idx_out1 = sample(range(ones), n1)
    idx_out0 = sample(range(zeros),n0)
    
    # Create subsets of the selected size
    X_out0 = [X_tot0[i] for i in range(zeros) if i in idx_out0]
    X_out1 = [X_tot1[i] for i in range(ones) if i in idx_out1]

    X_rest0 = [X_tot0[i] for i in range(zeros) if not i in idx_out0]
    X_rest1 = [X_tot1[i] for i in range(ones) if not i in idx_out1]
    
    # Juntar resultados
    X_out = X_out0 + X_out1
    X_rest = X_rest0 + X_rest1
    # Finalizar cambio de formato
    Y_out = pd.Series(chain.from_iterable([[0]*len(X_out0) , [1]*len(X_out1)]))
    Y_rest = pd.Series(chain.from_iterable([[0]*len(X_rest0) , [1]*len(X_rest1)]))
    
    return [X_out] , Y_out, [X_rest], Y_rest
################################################################
###################### CLASSIFICATION ##########################
################################################################

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


def create_classificator(X_train, Y_train, prm):
    # We use model for several layers
    classifier = Sequential()
    
    # Conv2D(# Filters , (filter shape)) , input_shape =(resolution, RGB)
    classifier.add(Conv2D(prm['n filters'], prm['shape filters'], input_shape = prm['input_shape'], activation = prm['f activate']))
    
    # Pooling
    classifier.add(MaxPooling2D(pool_size = prm['shape pooling'] ))
    
    # Flatten for further analysis
        # needed for further data analysis
    classifier.add(Flatten())
    
    # hidden layers
    for i in range(prm['dense layers']):
        classifier.add(Dense(units = prm['dense units'][0] , activation = prm['f activate']))
    
    # output layer (binary classification)
    classifier.add(Dense(units = 1, activation = prm['out activate']))
    
    # Compile
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    # Train
        # Batch size according to input
    if prm['batch size'] < 1 :
        batch_size = int(len(Y_train)*prm['batch size'])
    else:
        batch_size = prm['batch size']
        
    classifier.fit(X_train, Y_train, batch_size = batch_size, epochs = prm['epochs'], verbose = 1)
    
    return classifier


################################################################
###################### RESULTS #################################
################################################################
    
#############################################################
# Metrics
    
from sklearn.metrics import precision_score, recall_score
    
def metric_measures(classifier,X_train,Y_train,X_test,Y_test, y_type = 'binary'):
    
    # Training metrics ------------------------------
    Y_pred = classifier.predict(X_train)
    Y_pred = transform_prediction(Y_pred)

    # precision, and recall scores
    if y_type == 'binary':
        prec_tr = precision_score(Y_train, Y_pred , average="macro")
        rec_tr = recall_score(Y_train, Y_pred , average="macro")
        
    # accuracy and loss function
    loss_tr, acc_tr = classifier.evaluate(X_train, Y_train, verbose = 0)
    
    # Test metrics ------------------------------------
    Y_pred = classifier.predict(X_test)
    Y_pred = transform_prediction(Y_pred)

    # precision, and recall scores
    if y_type == 'binary':
        prec_test = precision_score(Y_test, Y_pred , average="macro")
        rec_test = recall_score(Y_test, Y_pred , average="macro")   
        
    # accuracy and loss function
    loss_test, acc_test = classifier.evaluate(X_test, Y_test, verbose = 0)
    
    # Changes format for saving
    metrics = ['loss','accuracy']
    training_metrics = [loss_tr, acc_tr]
    test_metrics = [loss_test, acc_test]
    
    if y_type == 'binary':
        metrics += ['precision','recall']
        training_metrics +=  [prec_tr, rec_tr]
        test_metrics += [prec_test, rec_test]
    
    training_metrics = dict(zip(metrics, training_metrics))
    test_metrics =  dict(zip(metrics, test_metrics))
    
    return test_metrics, training_metrics

def transform_prediction(Y_pred):
    
    if Y_pred.shape[1] > 1:
        Y_new = np.argmax(Y_pred, axis=1)
    else:
        Y_new = [round(float(y)) for y in Y_pred]
    
    return Y_new























