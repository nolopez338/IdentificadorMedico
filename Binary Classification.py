"""
Image Identifier
"""

################################################################
###################### SETUP ###################################
################################################################

# Install packages
import pandas as pd
import os
import sys

# Set working directory
manual_working_directory = True

# Directory where the Folder finds itself
if manual_working_directory:
    os.chdir('C:/Users/nicol/Desktop/Codigos Miscelaneos/Identificador Medico')
    #    os.chdir('C:/Users/no.lopez338/Desktop/Identificador Medico')
else:
    os.chdir(os.path.dirname(sys.argv[0]))

########################################################################
######################### PARAMETERS #######3###########################
########################################################################

# Parameters Basic --------------------------------------
prm = {}
    # adress of training Data
prm['trPath'] =   'Images'

    # Percentage to be used for the training set
    # Each class is divided according to this proportion
prm['training_percentage'] = 0.7

# Parameters for neural network -------------------------

# Hidden layers -------------------------
    # Hidden Layers activation function
prm['f activate'] = 'relu'
    # number of hidden units
prm['dense layers'] = 1
    # Units per Hidden layer
prm['dense units'] = [128]

# Convolutional layer -------------------
    # Size of filters
prm['shape filters'] = (3,3)
    # Number of filters (We use all posible filters)
prm['n filters'] = 32

# Pooling layer ------------------------
    # Pooling size
prm['shape pooling'] = (2,2)

# Output layer -------------------------
    # Activation function for output neuron
prm['out activate'] = 'sigmoid'

# Fitting procedure --------------------
    # Batch size for gradient descent training
prm['batch size'] = 0.1
    # Epochs to be used with the whole training sample set.
prm['epochs'] = 8

########################################################################
######################### Test parameters ##############################
########################################################################

prm_test = {}
prm_test['parameter_name'] = 'epochs'
prm_test['parameters'] = [7]
prm_test['n_training'] = 1



########################################################################
########################################################################
########################################################################
########################################################################
########################################################################
######################### RUN ##########################################
########################################################################
########################################################################
########################################################################
########################################################################


################################################################
###################### ESPECIAL MODULES ########################
################################################################

from load_images import load_images
from load_images import get_imagesY

from function_classification import get_proportional_subset
from function_classification import create_classificator
from function_classification import metric_measures

################################################################
###################### LOADING #################################
################################################################


# Load images from training path
images, frames = load_images(prm['trPath'])
# Saves input shape
prm['input_shape'] = images[0].shape


# Loads information summary
images_information = pd.read_csv('Labels.csv', sep =';')

# Selects objective variable and creates labels
objective_variable = 'structures'
# =============================================================================
# objective_variable = 'cystic'
# objective_variable = 'triangle'
# objective_variable = 'total'
# =============================================================================

column = [col for col in images_information.columns if objective_variable in col]

labels = images_information[column].to_numpy()

if objective_variable == 'total':
    labels = [lab[0] >= 4 for lab in labels]
else:
    labels = [lab[0] >= 2 for lab in labels]

images_information['label'] = labels

Y = get_imagesY(frames, images_information)

################################################################
###################### TESTING #################################
################################################################


df_test = pd.DataFrame({'loss': [] ,'accuracy' : [], 'precision' :[], 'recall': [], prm_test['parameter_name'] : []})
df_train = pd.DataFrame({'loss': [] ,'accuracy' : [], 'precision' :[], 'recall': [], prm_test['parameter_name'] : []})

# Loop over all parameters
for p in prm_test['parameters']:
    
    print(str(p))
    print('\n \n')
    # Loop over 3 different training sets
    
    prm[prm_test['parameter_name']] = p
    
    for n_tr in range(prm_test['n_training']):
        # Get Test and training sets from the set of images
        n_train = int(prm['training_percentage']*len(Y))
        
        X_train, Y_train, X_test, Y_test = get_proportional_subset(images, Y, n = n_train, sd = n_tr)
        
        # Create classificator
        classifier = create_classificator(X_train, Y_train, prm)
        
        # Get metric measures
        test_metrics, training_metrics = metric_measures(classifier, X_train, Y_train, X_test, Y_test, y_type = 'binary')
        
        test_metrics[prm_test['parameter_name']] = str(p)
        training_metrics[prm_test['parameter_name']] = str(p)
        
        # Save in dataframes
        df_test = pd.concat([df_test, pd.DataFrame(test_metrics, index = [0])], ignore_index=True)
        df_train = pd.concat([df_train, pd.DataFrame(training_metrics, index = [0])], ignore_index=True)
    
df_test.to_excel('df_test ' + prm_test['parameter_name'] + '.xlsx')
df_train.to_excel('df_train ' + prm_test['parameter_name']  + '.xlsx')



    
