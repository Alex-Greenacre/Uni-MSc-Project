from matplotlib import pyplot as plt 
import tensorflow as tf
from tensorflow.keras import ops
from tensorflow.keras.utils import pad_sequences
from scipy import io as sci
import h5py as h5
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
####
#Title - Functions for custom loss 
#Description - Functions used to measure model loss and metrics 
#Notes - Uses tf instead of numpy for compatibility with tensors
#        Main functions convert degree + distance from origin to x and y 
####
#Converts degrees to radians  
def degreesToRadians(degree):
    #radain = 0 = d * pi/180
    return degree* np.pi/180
#Main conversion function to gain x and y from distance and degree form origin 
def polarToCartesian(distance,degree):
    # x = r cos(0)
    x= distance*tf.cos(degreesToRadians(degree))
    # y = r sin(0)
    y = distance*tf.sin(degreesToRadians(degree))
    return x,y

#Mean square error functions
def getDistance(pred_x,pred_y,real_x,real_y):
    #As data is in meters times by 100 to convert to cm 
    return (tf.square(real_x-pred_x) + tf.square(real_y-pred_y))*100
#For ECG
def getDistanceMse(pred_y,real_y):
    return tf.square(real_y-pred_y)

#Absolute Error Functions 
def getDistanceMae(pred_x,pred_y,real_x,real_y):
        return (abs(real_x-pred_x) + abs(real_y-pred_y))*100
#Mean Squared Error Functions 

#Gets the mean loss for all labels 
def distanceLoss(real_y,pred_y):
    mask = real_y[:,:,0]
    #Gets X and Y points 
    pred_x,pred_y = polarToCartesian(pred_y[:,:,0],pred_y[:,:,1])
    real_x,real_y = polarToCartesian(real_y[:,:,0],real_y[:,:,1])    
    distance = getDistance(pred_x,pred_y,real_x,real_y)
    #filtered_distance = tf.where(mask != 0,distance,-1)
    #filtered_distance = ops.mean(tf.boolean_mask(filtered_distance, tf.not_equal(filtered_distance,-1)))
    #Gets distance and returns as the error 
    return ops.mean(distance) 

#Filters out all labels with 0 getting mean only for valid labels 
def distanceLossMasked(real_y,pred_y):
    mask = real_y[:,:,0]
    pred_x,pred_y = polarToCartesian(pred_y[:,:,0],pred_y[:,:,1])
    real_x,real_y = polarToCartesian(real_y[:,:,0],real_y[:,:,1])    
    distance = getDistance(pred_x,pred_y,real_x,real_y)
    #Where mask is not 0 keep value else -1 
    filtered_distance = tf.where(mask != 0,distance,-1)
    #filter by -1 removing the mask 
    distance = tf.boolean_mask(filtered_distance, tf.not_equal(filtered_distance,-1))
    return ops.mean(distance) 

#For use in metrics rather than loss, metrics needs converting to float to allow for calculation 
def distanceMetric(real_y,pred_y):
    real_y = tf.cast(real_y,tf.float32)
    pred_y = tf.cast(pred_y,tf.float32)
    mask = real_y[:,:,0]
    #Gets X and Y points 
    pred_x,pred_y = polarToCartesian(pred_y[:,:,0],pred_y[:,:,1])
    real_x,real_y = polarToCartesian(real_y[:,:,0],real_y[:,:,1])    
    distance = getDistance(pred_x,pred_y,real_x,real_y)
    #filtered_distance = tf.where(mask != 0,distance,-1)
    #filtered_distance = tf.boolean_mask(filtered_distance, tf.not_equal(filtered_distance,-1))
    #Gets distance and returns as the error 
    return ops.mean(distance) 

#Absolute error loss functions 
def distanceLossMae(real_y,pred_y):
    #Gets X and Y points 
    pred_x,pred_y = polarToCartesian(pred_y[:,:,0],pred_y[:,:,1])
    real_x,real_y = polarToCartesian(real_y[:,:,0],real_y[:,:,1])    
    distance = getDistanceMae(pred_x,pred_y,real_x,real_y)
    return ops.mean(distance) 

#Filters out all labels with 0 getting mean only for valid labels 
def distanceLossMaeMasked(real_y,pred_y):
    mask = real_y[:,:,0]
    pred_x,pred_y = polarToCartesian(pred_y[:,:,0],pred_y[:,:,1])
    real_x,real_y = polarToCartesian(real_y[:,:,0],real_y[:,:,1])    
    distance = getDistanceMae(pred_x,pred_y,real_x,real_y)
    #Where mask is not 0 keep value else -1 
    filtered_distance = tf.where(mask != 0,distance,-1)
    #filter by -1 removing the mask 
    distance = tf.boolean_mask(filtered_distance, tf.not_equal(filtered_distance,-1))
    return ops.mean(distance) 

#Loss function for ECG model 
#Note - this does not work, data shape not compatible with mask 
def mseDistance(real_y,pred_y):
    mask = real_y
    distance = getDistanceMse(real_y,pred_y)
    filtered_distance = tf.where(mask != 0,distance,0)
    distance = tf.boolean_mask(filtered_distance, tf.not_equal(filtered_distance,-360))
    return ops.mean(distance)

