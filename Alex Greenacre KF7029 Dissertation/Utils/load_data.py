####
# Title - Load data 
# Description - Loads data for localisation and ECG scenarios 
# Notes -  loadData: loads for localisation 
#          loadEcgData: loads for ecg  
####

import numpy as np 
import h5py as h5
from sklearn.model_selection import train_test_split

def loadData (data_files,reference_files):
    dir='./Data/'
    X_train =[]
    X_test = []
    y_train = []
    y_test = []
    max_people = 5 
    #Loops through each data and label files, gets data and labels from each
    for i in range(0,len(data_files)):
        data_file = h5.File(dir+data_files[i],'r')
        reference_file = h5.File(dir+reference_files[i],'r')
        antena_data = np.array(data_file['data_radar']['tx_1'])
        #change arr structure so each time sample can have a label assigned
        antena_data = antena_data.reshape(antena_data.shape[1],antena_data.shape[2],antena_data.shape[0])
        #data.extend(antena_data)
        #Get the labels for the amount of people in the image
        person_range = np.array([float(x) for x in reference_file['meta_data_ref']['range']])
        person_angle = np.array([float(x) for x in reference_file['meta_data_ref']['doa']])
        person_location = []
        for i in range(0,len(person_range)):
            person_location.append([person_range[i],person_angle[i]])
        #Used in a multi person count model (ie scenarios with 1-5 people)
        #Fills the remaining label list with 0,0 - as distance cant be <1 this can be filtered in loss/post model prediction
        if len(person_range) < max_people:
            for x in range(0,max_people - len(person_range)):
                person_location.append([0,0]) 
        #Repeats the label for each radar entry (2000 per file)
        person_location = np.tile(person_location,(2000,1,1))
        train_data,test_data,train_labels,test_labels = train_test_split(antena_data,person_location,test_size=0.3,random_state=1)
        X_train.extend(train_data)
        X_test.extend(test_data)
        y_train.extend(train_labels)
        y_test.extend(test_labels)
        #labels.extend(person_location)
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    #As data is complex real and imag is extracted then combined into a array as standard numbers 
    train_real = X_train['real']
    train_img = X_train['imag']
    X_train=np.stack((train_real,train_img),axis=-1)
    shape = X_train.shape 
    shape += (1,)
    X_train = X_train.reshape(shape)
    test_real = X_test['real']
    test_img = X_test['imag']
    X_test=np.stack((test_real,test_img),axis=-1)
    shape = X_test.shape 
    shape += (1,)
    X_test = X_test.reshape(shape)
    
    return X_train,X_test,y_train,y_test

def loadEcgData(data_files,reference_files):
    dir='./Data/'
    data = []
    labels = []
    segmented_ecg=[]
    segmented_data =[]
    #Gets data and labels from every file  
    for i in range(0,len(data_files)):
        data_file = h5.File(dir+data_files[i],'r')
        reference_file = h5.File(dir+reference_files[i],'r')
        antena_data = np.array(data_file['data_radar']['tx_1'])
        #change arr structure so each time sample can have a label assigned
        antena_data = antena_data.reshape(antena_data.shape[1],antena_data.shape[2],antena_data.shape[0])
        data.extend(antena_data)
        #Get the labels for the amount of people in the image
        ecg = reference_file['data_ref']['ecg']
        time_axis = data_file['meta_data_radar']['time_axis_st']
    
        #print(ecg.shape)
        count =1
        block_start = 0 
        data_block=[]
        ecg_block=[]
       
        for x in range(0,len(antena_data)):
            if count ==10:
                data_block.append(data[i])
                segmented_data.append(data_block)
                segmented_ecg.append(ecg[:,int(round(time_axis[0,block_start]*100)):int(round(time_axis[0,x]*100))].tolist())
                data_block=[]
                ecg_block=[]
                count =1
                block_start = x+1
            else: 
                data_block.append(data[x])
                count +=1
    print(np.array(segmented_data).shape)
        #print(max(segmented_ecg[:][:]))
    max_length=0
        #segmented_ecg = segmented_ecg.tolist()
    for dim_one in range(0,len(segmented_data)):
        for dim_two in segmented_ecg[dim_one]:
            if len(dim_two) > max_length: 
                max_length=len(dim_two) 
                #print(max_length)
    for dim_one in range(0,len(segmented_ecg)):
        for dim_two in range(0,len(segmented_ecg[dim_one])):
            for num_of_zeros in range(0,max_length - len(segmented_ecg[dim_one][dim_two])):
                segmented_ecg[dim_one][dim_two].append(0)
    data = np.array(segmented_data)
    labels = np.array(segmented_ecg)
    #As data is complex real and imag is extracted then combined into a array as standard numbers 
    real_data = data['real']
    img_data = data['imag']
    data = np.stack((real_data,img_data),axis=-1)
    
    shape = data.shape 
    shape += (1,)
    data = data.reshape(shape)
    print(data.shape)
    
    return data,labels
  









def loadDataBackup (data_files,reference_files):
    dir='./Data/'
    data = []
    labels = []
    max_people = 5 
    #Loops through each data and label files, gets data and labels from each
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
        #Used in a multi person count model (ie scenarios with 1-5 people)
        #Fills the remaining label list with 0,0 - as distance cant be <1 this can be filtered in loss/post model prediction
        if len(person_range) < max_people:
            for x in range(0,max_people - len(person_range)):
                person_location.append([0,0]) 
        #Repeats the label for each radar entry (2000 per file)
        person_location = np.tile(person_location,(2000,1,1))
        labels.extend(person_location)
    data = np.array(data)
    labels = np.array(labels)
    #As data is complex real and imag is extracted then combined into a array as standard numbers 
    real_data = data['real']
    img_data = data['imag']
    data = np.stack((real_data,img_data),axis=-1)
    shape = data.shape 
    shape += (1,)
    data = data.reshape(shape)
    return data,labels
