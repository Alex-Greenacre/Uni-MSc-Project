from matplotlib import pyplot as plt 
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from scipy import io as sci
import h5py as h5
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


dir='./Data/'
#Experiment 1 & 3 
#All different postures and no obsticle, angles 0,-45 distance 1,3
data_files=['m3_radar.mat','m1_radar.mat','m2_radar.mat']
reference_files= ['m3_reference.mat','m1_reference.mat','m2_reference.mat']
data = []
labels = []
#Gets data and labels from every file  
for i in range(0,len(data_files)):
    data_file = h5.File(dir+data_files[i],'r')
    reference_file = h5.File(dir+reference_files[i],'r')
    antena_data = np.array(data_file['data_radar']['tx_1'])
    #change arr structure so each time sample can have a label assigned
    print(antena_data.dtype)
    antena_data = antena_data.reshape(antena_data.shape[1],antena_data.shape[2],antena_data.shape[0])
    data.extend(antena_data)
    
    #Get the labels for the amount of people in the image
    person_range = np.array([float(x) for x in reference_file['meta_data_ref']['range']])
    person_angle = np.array([float(x) for x in reference_file['meta_data_ref']['doa']])
    person_location = np.concatenate((person_range,person_angle))
    #if len(labels) == 0:
    labels.extend(np.tile(person_location,(2000,1)))
    #print(np.reshape(np.array(labels).shape,(-1,2)))
    #else:
    #    labels += np.repeat(person_count,antena_data.shape[0])
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
#labels = labels.reshape(-1,1)
#labels = labels.reshape(labels.shape+(1,))
print (labels.shape)
shape = data.shape 
shape += (1,)
data = data.reshape(shape)

print(data.shape)
print(labels[0],labels[2000])
#Data split 
X_train,X_test,y_train,y_test = train_test_split(data,labels,test_size=0.3,random_state=1)
#Model 
input = layers.Input(shape= data.shape[1:])

model_layer = layers.Conv3D(filters=128,kernel_size=(5,5,2),activation='relu')(input)
model_layer = layers.BatchNormalization()(model_layer)
model_layer = layers.MaxPooling3D((2,2,1))(model_layer)
model_layer = layers
model_layer = layers.Conv3D(filters=64,kernel_size=(3,3,1),activation='relu')(model_layer)
model_layer = layers.MaxPooling3D((2,2,1))(model_layer)
model_layer = layers.Flatten()(model_layer)
model_layer = layers.Dense(units=64)(model_layer)
#Allows for two ouputs one for angle and one for distance 
#This setup will mean that loss can be tracked for both variables rather than having both in the same array  
angle_output = layers.Dense(units=1,name='angle_output')(model_layer)
distance_output = layers.Dense(units=1,name='distance_output')(model_layer)
#output = layers.Reshape((1,2))(model_layer)

#Both output layers 
model = models.Model(inputs =input,outputs=[angle_output,distance_output])
#Model can take a dictionary or an array, because of this the name of each output can be stated to view the loss for both outputs whilst also handling y data as and array 
model.compile(optimizer='adam',loss={'angle_output':'mse','distance_output':'mse'},metrics={'angle_output':'mean_squared_error','distance_output':'mean_squared_error'})
model.summary()
print('\nTrain shape\n',X_train.shape,y_train.shape,'\nTest shapes\n',X_test.shape,y_test.shape)
#Organise the y data to be angle then distance 
results = model.fit(X_train,[y_train[:,1],y_train[:,0]],epochs=100,batch_size=100,validation_data=(X_test,[y_test[:,1],y_test[:,0]]))

fig,ax = plt.subplots(3,1)
ax[0].plot(results.history['loss'],label='loss' )
ax[0].plot(results.history['val_loss'],label='val loss')
ax[1].plot(results.history['angle_output_mean_squared_error'],label='angle loss' )
ax[1].plot(results.history['val_angle_output_mean_squared_error'],label='angle val loss')
ax[2].plot(results.history['distance_output_mean_squared_error'],label='distance loss' )
ax[2].plot(results.history['val_distance_output_mean_squared_error'],label='distance val loss')

ax[0].set_title('Total_Loss')
ax[1].set_title('Angle_Loss')
ax[2].set_title('Distance_Loss')
ax[0].legend()
ax[1].legend()
ax[2].legend()

plt.subplots_adjust(wspace=0.1,hspace=0.3)
plt.show()
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
