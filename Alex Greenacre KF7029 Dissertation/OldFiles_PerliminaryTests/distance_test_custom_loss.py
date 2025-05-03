from matplotlib import pyplot as plt 
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers,ops
from tensorflow.keras.utils import pad_sequences
from scipy import io as sci
import h5py as h5
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from Utils.loss_function import distanceLoss, distanceMetric
dir='./Data/'
#Experiment 1 & 3 
#All different postures and no obsticle, angles 0,-45 distance 1,3
#data_files=['m3_radar.mat','m1_radar.mat','m2_radar.mat','m30_radar.mat']
#reference_files= ['m3_reference.mat','m1_reference.mat','m2_reference.mat','m30_reference.mat']

#data_files=['m30_radar.mat']
#reference_files= ['m30_reference.mat']
#data_files=['m35_radar.mat','m1_radar.mat','m30_radar.mat','m11_radar.mat','m35-1_radar.mat','m30-1_radar.mat']
#reference_files= ['m35_reference.mat','m1_reference.mat','m30_reference.mat','m11_reference.mat','m35-1_reference.mat','m30-1_reference.mat']
data_files=['m1_radar.mat','m2_radar.mat','m3_radar.mat','m4_radar.mat','m5_radar.mat','m6_radar.mat','m7_radar.mat','m8_radar.mat','m9_radar.mat','m10_radar.mat','m11_radar.mat']
reference_files= ['m1_reference.mat','m2_reference.mat','m3_reference.mat','m4_reference.mat','m5_reference.mat','m6_reference.mat','m7_reference.mat','m8_reference.mat','m9_reference.mat','m10_reference.mat','m11_reference.mat']

data = []
labels = []
max_people = 1
#Gets data and labels from every file  
for i in range(0,len(data_files)):
    data_file = h5.File(dir+data_files[i],'r')
    reference_file = h5.File(dir+reference_files[i],'r')
    antena_data = np.array(data_file['data_radar']['tx_1'])
    #change arr structure so each time sample can have a label assigned
    antena_data = antena_data.reshape(antena_data.shape[1],antena_data.shape[2],antena_data.shape[0])
    data.extend(antena_data)
    
    #Get the labels for the amount of people in the image
    person_range = np.array([float(x) for x in reference_file['meta_data_ref']['range']])
    person_angle = np.array([float(x) for x in reference_file['meta_data_ref']['doa']])
    person_location = []
    for i in range(0,len(person_range)):
        person_location.append([person_range[i],person_angle[i]])
    if len(person_range) < max_people:
        for x in range(0,max_people - len(person_range)):
            person_location.append([-0,-0])  
    person_location = np.tile(person_location,(2000,1,1))
    labels.extend(person_location)
data = np.array(data)
#As data is complex real and imag is extracted then combined into a array as standard numbers 
real_data = data['real']
img_data = data['imag']
data = np.stack((real_data,img_data),axis=-1)
print('combined shape: ',data.shape)
#data = data['real']
#data = data.astype(str)
#data = data.astype(float)
labels = np.array(labels)
print(labels[1])

#labels = labels.reshape(-1,1)
#labels = labels.reshape(labels.shape+(1,))
print (labels.shape)
shape = data.shape 
shape += (1,)
data = data.reshape(shape)

print(data.shape)
#Data split 
X_train,X_test,y_train,y_test = train_test_split(data,labels,test_size=0.3,random_state=1)
#Model 
input = layers.Input(shape= data.shape[1:])
#mask = layers.Masking(mask_value=-360)(input)
model_layer = layers.Conv3D(filters=64,kernel_size=(5,5,2),activation='relu')(input)
model_layer = layers.BatchNormalization()(model_layer)
model_layer = layers.MaxPooling3D((2,2,1))(model_layer)
model_layer = layers.Conv3D(filters=32,kernel_size=(3,3,1),activation='relu')(model_layer)
model_layer = layers.MaxPooling3D((2,2,1))(model_layer)
model_layer = layers.Flatten()(model_layer)
model_layer = layers.Dense(units=512)(model_layer)
model_layer = layers.Dense(units=2)(model_layer)
output = layers.Reshape((1,2))(model_layer)
#Allows for two ouputs one for angle and one for distance 
#This setup will mean that loss can be tracked for both variables rather than having both in the same array  
#output = layers.Reshape((1,2))(model_layer)

#Both output layers 
model = models.Model(inputs =input,outputs=output)
#Model can take a dictionary or an array, because of this the name of each output can be stated to view the loss for both outputs whilst also handling y data as and array 
model.compile(optimizer='adam',loss=distanceLoss, metrics=['mse'])
model.summary()
print('\nTrain shape\n',X_train.shape,y_train.shape,'\nTest shapes\n',X_test.shape,y_test.shape)
#Organise the y data to be angle then distance 
results = model.fit(X_train,y_train,epochs=50,batch_size=64,validation_data=(X_test,y_test))

plt.plot(results.history['loss'],label='loss' )
plt.plot(results.history['val_loss'],label='val loss')
plt.show()
#test_file = h5.File(dir+'m30_radar.mat','r')
#test_data = np.array(test_file['data_radar']['tx_1'])
#test_data = test_data.reshape(test_data.shape[1],test_data.shape[2],test_data.shape[0])
#test_data_real = test_data['real']
#test_data_img = test_data['imag']
#test_data = np.stack((test_data_real,test_data_img),axis=-1)
#test_data = test_data.reshape((2000,137,13,2))
#print(test_data[0].shape)
#test_data = test_data[0]
#test_data = test_data.reshape(1,137,13,2)
#print(model.predict(test_data))
test_file = h5.File(dir+'m2_radar.mat','r')
test_data = np.array(test_file['data_radar']['tx_1'])
test_data = test_data.reshape(test_data.shape[1],test_data.shape[2],test_data.shape[0])
test_data_real = test_data['real']
test_data_img = test_data['imag']
test_data = np.stack((test_data_real,test_data_img),axis=-1)
#test_data = test_data.reshape((2000,137,13,2))
print(test_data[0].shape)
test_data = test_data[0]
test_data = test_data.reshape(1,137,13,2)
print(model.predict(test_data))
test_file = h5.File(dir+'m1_radar.mat','r')
test_data = np.array(test_file['data_radar']['tx_1'])
test_data = test_data.reshape(test_data.shape[1],test_data.shape[2],test_data.shape[0])
test_data_real = test_data['real']
test_data_img = test_data['imag']
test_data = np.stack((test_data_real,test_data_img),axis=-1)
#test_data = test_data.reshape((2000,137,13,2))
print(test_data[0].shape)
test_data = test_data[0]
test_data = test_data.reshape(1,137,13,2)
print(model.predict(test_data))