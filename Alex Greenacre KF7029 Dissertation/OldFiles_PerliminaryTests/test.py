from matplotlib import pyplot as plt 
import tensorflow as tf
#from tf_keras import layers, models, optimizers
from tensorflow.keras import layers, models, optimizers
from scipy import io as sci
import h5py as h5
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

#Revert to keras 2 for complex dtypes 
#import os
#os.environ["TF_USE_LEGACY_KERAS"]="1"

dir='./Data/'
#Open data: 3 open (1,2,3)
#data_files=['m35_radar.mat','m1_radar.mat','m30_radar.mat']
#reference_files= ['m35_reference.mat','m1_reference.mat','m30_reference.mat']

#Mixed data: 3 open (1,2,3) & 2 wall (1,2,3) 
data_files=['m35_radar.mat','m1_radar.mat','m30_radar.mat','m11_radar.mat','m35-1_radar.mat','m30-1_radar.mat','m36_radar.mat','m2_radar.mat']
reference_files= ['m35_reference.mat','m1_reference.mat','m30_reference.mat','m11_reference.mat','m35-1_reference.mat','m30-1_reference.mat','m36_reference.mat','m2_reference.mat']

#Wall data: 3 wall (1,2,3) 
#data_files=['m11_radar.mat','m35-1_radar.mat','m30-1_radar.mat']
#reference_files= ['m11_reference.mat','m35-1_reference.mat','m30-1_reference.mat']

data = []
labels = []
label_count =0
for i in range(0,len(data_files)):
    data_file = h5.File(dir+data_files[i],'r')
    reference_file = h5.File(dir+reference_files[i],'r')
    antena_data = np.array(data_file['data_radar']['tx_1'])
    #change arr structure so each time sample can have a label assigned
    antena_data = antena_data.reshape(antena_data.shape[1],antena_data.shape[2],antena_data.shape[0])
    data.extend(antena_data)
    
    #Get the labels for the amount of people in the image
    person_ids = reference_file['meta_data_ref']['person_ID']
    person_count = person_ids.shape[0]
    #if len(labels) == 0:
    labels.extend(np.repeat(person_count-1,antena_data.shape[0]))
    label_count+=1
    #else:
    #    labels += np.repeat(person_count,antena_data.shape[0])
encoder = OneHotEncoder(sparse_output=False) 
data = np.array(data)
#data = data['real']
#data = data.astype(str)
#data = data.astype(float)
test_data_real = data['real']
test_data_img = data['imag']
data = np.stack((test_data_real,test_data_img),axis=-1)
labels = np.array(labels)
labels = labels.reshape(labels.shape+(1,))
print (labels.shape)
labels = encoder.fit_transform(labels)
shape = data.shape 
shape += (1,)
data = data.reshape(shape)

print(data.shape)
#Data split 
X_train,X_test,y_train,y_test = train_test_split(data,labels,test_size=0.3,random_state=1)

#Model 
input = layers.Input(shape= data.shape[1:])
model_layer = layers.Conv3D(filters=32,kernel_size=(5,5,2),activation='relu')(input)
model_layer = layers.BatchNormalization()(model_layer)
model_layer = layers.MaxPooling3D((2,2,1))(model_layer)
model_layer = layers.Conv3D(filters=16,kernel_size=(3,1,1),activation='relu')(model_layer)
model_layer = layers.MaxPooling3D((2,2,1))(model_layer)
model_layer = layers.Flatten()(model_layer)
model_layer = layers.Dense(units=16,activation='softmax')(model_layer)
output = layers.Dense(units=labels.shape[1],activation='softmax')(model_layer)

model = models.Model(inputs =input,outputs=output)

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()
print('\nTrain shape\n',X_train.shape,y_train.shape,'\nTest shapes\n',X_test.shape,y_test.shape)
results = model.fit(X_train,y_train,epochs=50,batch_size=100,validation_data=(X_test,y_test))

fig,ax = plt.subplots(2,1)
ax[0].plot(results.history['loss'],label='loss' )
ax[0].plot(results.history['val_loss'],label='val loss')
ax[1].plot(results.history['accuracy'],label='accuracy')
ax[1].plot(results.history['val_accuracy'],label='val accuracy')
ax[0].set_title('Loss')
ax[1].set_title('Accuracy')
ax[0].legend()
ax[1].legend()
plt.subplots_adjust(wspace=0.1,hspace=0.3)
plt.show()

print(len(data_file['data_radar']['tx_1'][:][:][0]))
print(reference_file['data_ref']['ecg'])




"""
Legacy Code 

print(data.keys())
print(data['data_radar'].keys())
print(data['data_radar']['tx_1'])
sequence = np.array(data['data_radar']['tx_1'])
print(sequence.shape)
sequence = sequence.reshape(sequence.shape[1],sequence.shape[2],sequence.shape[0])
print(sequence.shape)
print(sequence[1])
labels = h5.File(dir+'m35_reference.mat','r')
print(labels['meta_data_ref'].keys())
test = labels['meta_data_ref']
print(test['person_ID'])
"""