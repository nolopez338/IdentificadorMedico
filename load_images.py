# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 11:20:28 2020
@author: nicol
"""
################################################################
###################### SETUP ###################################
################################################################

# Import packages
import os
import numpy as np

# Image load
from keras.preprocessing.image import load_img
from keras.preprocessing.image import save_img
# Convert to array
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img


################################################################
###################### LOAD DATA###### #########################
################################################################

def load_images(path = 'Images', extension = 'jpg'):
    
    images = []
        
        # Initialize
    files_trData = []

        # Gets all files with corresponding extension
    for file in os.listdir(path):
        if extension in file:
            files_trData.append(path + '\\' + file)
        
        # Loads file into a numpy array.
    for f in files_trData:
        img = load_img(f)
        img_array = img_to_array(img)
        img_array = np.array(img_array)
        
        images.append(img_array)
    
        # Gets frame id
    spl1 = 'frame_'
    spl2 = '.jpg'
    
    frames = [file.split(spl1)[1] for file in files_trData]
    frames = [file.split(spl2)[0] for file in frames]

        
    return images, frames


# Reduces the amount of images in a folder by sampling.
def reduce_image_quantity(trPath_old = 'Images/', trPath_new = 'Images reduced/', step = 3):
    # Loads images
    images = load_images(trPath = trPath_old, step = step)
    
    # Saves Images
    if True:
        os.mkdir(trPath_new.replace('/',''))
        
        idx = 1
        for img in images:
            path = trPath_new + 'frame' + str(idx) + '.jpg'
            save_img(path, img)
            idx += 1

# Cut black soaces from image
def cut_top_images(images, cut = 0.8):
    images_new = {}
    

    
    for img in images:
        
        lvl_tmp.append(img[cut:,:,:])
        
    images_new[key] = lvl_tmp
    
    return images_new

################################################################
###################### AUXILIAR ###### #####################
################################################################
            
def number_of_files(path, extension = '.jpg', return_files = False):
     # Initialize
    files = []
    
    # Gets all .jpg files
    for r, d, f in os.walk(path):
        for file in f:
            if ".jpg" in file:
                files.append(os.path.join(r, file))
    out = len(files)
    if return_files:
        out = files
    return out

def get_imagesY(frames, images_information):
    Y = []
    for i in range(len(frames)):
        
        row = images_information[images_information['score'].str.contains(frames[i])]
        
        # Saves value
        Y.append(row['label'].values[0])
        
    return Y
        
























            
