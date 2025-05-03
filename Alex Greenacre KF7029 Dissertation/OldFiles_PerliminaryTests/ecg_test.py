from matplotlib import pyplot as plt 
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from scipy import io as sci
import h5py as h5
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
dir='./Data/'
data_files=['m1_radar.mat','m2_radar.mat','m35_radar.mat']
reference_files= ['m1_reference.mat','m2_reference.mat','m35_reference.mat']
data = []
labels = []
label_count =0
reshaped_ecg_labels = []
longest_entry = 0
for i in range(0,len(data_files)):
    data_file = h5.File(dir+data_files[i],'r')
    reference_file = h5.File(dir+reference_files[i],'r')
    #plt.plot(reference_file['data_ref']['ecg'][0][:100])
    #plt.show()
    antena_data = np.array(data_file['data_radar']['tx_1'])
    #change arr structure so each time sample can have a label assigned
    antena_data = antena_data.reshape(antena_data.shape[1],antena_data.shape[2],antena_data.shape[0])
    data.extend(antena_data)
    
    time_axis = np.array(data_file['meta_data_radar']['time_axis_st'])
    ecg_data = np.array(reference_file['data_ref']['ecg'])
    #ecg array as 17,000+ entries wheres time axis has 200 entries with the final matching the total number of ecg entires 
    #convert ecg size into slow sample rate 
    #will add each ecg entry a smaple rate based on the time the sample rate was taken 
    
    for i in range(0,len(time_axis[0])):
        slow_sample_label = []
        for y in range(0,len(ecg_data[0])):
            if i == 0:
                if y <= time_axis[0][i]*100:
                    slow_sample_label.append(ecg_data[0][y])
            elif y <= time_axis[0][i]*100 and y> time_axis[0][i-1]*100:
                slow_sample_label.append(ecg_data[0][y])
        #gets the count of the longest sample label for padding 
        if len(slow_sample_label) >= longest_entry: 
            longest_entry = len(slow_sample_label)
        #if len(slow_sample_label)!=21: print(len(slow_sample_label))
        reshaped_ecg_labels.append(slow_sample_label)
#as ecg did not register for the first sample that is dropped from both data and labels 

reshaped_ecg_labels = np.array(np.pad(reshaped_ecg_labels[:],longest_entry))
print(reshaped_ecg_labels.shape) 
data = np.array(data)
data = data['real']
count =0
reshaped_data = []
segment = []
#Prepares training data and label 
for i in range(0,len(data)):
    if count == 50-1:
        labels.append(reshaped_ecg_labels[i])
        reshaped_data.append(segment)
        segment = []
        count = 0
    else:
        count+=1
        segment.append(data[i])
reshaped_data = np.array(reshaped_data)
reshaped_data = reshaped_data.reshape(reshaped_data.shape[0],49,-1)
labels = np.array(labels)
print(reshaped_data.shape,labels.shape)
X_train,X_test,y_train,y_test = train_test_split(reshaped_data,labels,test_size=0.3,random_state=1)

#Model 
input = layers.Input(shape= reshaped_data.shape[1:])
model_layer = layers.LSTM(units=64,return_sequences=True)(input)
model_layer = layers.LSTM(units=64)(model_layer)
model_layer = layers.Dense(units=10)(model_layer)
output = layers.Dense(units=1)(model_layer)
model = models.Model(inputs=input, outputs=output)

model.compile(loss='mse',optimizer='adam')
model.summary()
results = model.fit(X_train,y_train, epochs=250,validation_data=(X_test,y_test))

fig,ax = plt.subplots(2,1)
ax[0].plot(results.history['loss'],label='loss' )
ax[0].plot(results.history['val_loss'],label='val loss')
ax[0].set_title('Loss')
ax[1].set_title('Accuracy')
ax[0].legend()
ax[1].legend()
plt.subplots_adjust(wspace=0.1,hspace=0.3)
plt.show()