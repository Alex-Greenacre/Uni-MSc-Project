import h5py as h5
import numpy as np
dir = './Data/'
data_file='m1_radar.mat'
reference_file= 'm1_reference.mat'
#Data format 
#Antenna
# slow time samples (holding fat time samples)
# fast time samples  
data = h5.File(dir+data_file,'r')
label = h5.File(dir+reference_file,'r')
print(data.keys())
print(data['data_radar']['tx_1']['real'][2][1][0])

test_complex_num = np.complex64(data['data_radar']['tx_1'])
print('\ncomplex test')
print(test_complex_num.dtype)

#time axis 
print('Time stamps')
print(data['meta_data_radar']['time_axis_st'][0][-5:])
time_axis = np.array(data['meta_data_radar']['time_axis_st'])
#ecg 
print('labels')
print(label.keys())
print(label['data_ref']['ecg'])
ecg_data = np.array(label['data_ref']['ecg'])
print(ecg_data.shape)
print(ecg_data[0][0])
reshaped_ecg_labels = []
#convert ecg size into slow sample rate 
for i in range(0,len(time_axis[0])):
    slow_sample_label = []
    for y in range(0,len(ecg_data[0])):
        if i == 0:
            if y <= time_axis[0][i]*100:
                slow_sample_label.append(ecg_data[0][y])
        elif y <= time_axis[0][i]*100 and y> time_axis[0][i-1]*100:
            slow_sample_label.append(ecg_data[0][y])
    reshaped_ecg_labels.append(np.mean(slow_sample_label))
reshaped_ecg_labels = np.array(reshaped_ecg_labels)
print(reshaped_ecg_labels.shape) 

