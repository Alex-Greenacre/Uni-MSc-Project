from Models.cnn_model import CnnModel
from Models.resnet_model import ResNetModel
from Models.vit_model import ViTModel
from Utils.data_dictionary import getDataFiles
from Utils.load_data import loadData
#Data prep 
data_files, reference_files = getDataFiles(2) 
X_train,X_test,y_train,y_test = loadData(data_files,reference_files)
#Metric 1 - Cnn Model 
model = CnnModel()
model.loadModel('CNN_Model')
cnn_results = model.getEvaluation(X_test,y_test)
#Remove last axis 
X_train = X_train.reshape(X_train.shape[:-1])
X_test = X_test.reshape(X_test.shape[:-1])

#Metric 2 - ResNet Model 
model = ResNetModel()
model.loadModel('ResNet_Model')
resnet_results = model.getEvaluation(X_test,y_test)

#Metirc 3 - ViT Model 
model = ViTModel()
model.loadModel('ViT_Model')
vit_results = model.getEvaluation(X_test,y_test)
print('Masked')
print('--Loss (mse cm)--','\nCNN ',cnn_results[0],'\nResNet ',resnet_results[0],'\nViT ',vit_results[0])
print('--mse (polar)--','\nCNN ',cnn_results[1],'\nResNet ',resnet_results[1],'\nViT ',vit_results[1])
print('--Mae (cm)--','\nCNN ',cnn_results[2],'\nResNet ',resnet_results[2],'\nViT ',vit_results[2])


#Reload Files with original shape, as random is locked in test train split this will be the same vals 
X_train,X_test,y_train,y_test = loadData(data_files,reference_files)

#Metric 1 - Cnn Model 
model = CnnModel()
model.loadModel('CNN_Model_No_Mask',False)
cnn_results = model.getEvaluation(X_test,y_test)
X_train = X_train.reshape(X_train.shape[:-1])
X_test = X_test.reshape(X_test.shape[:-1])

#Metric 2 - ResNet Model 
model = ResNetModel()
model.loadModel('ResNet_Model_No_Mask',False)
resnet_results = model.getEvaluation(X_test,y_test)

#Metirc 3 - ViT Model 
model = ViTModel()
model.loadModel('ViT_Model_No_Mask',False)
vit_results = model.getEvaluation(X_test,y_test)

print('No Mask')
print('--Loss (mse cm)--','\nCNN ',cnn_results[0],'\nResNet ',resnet_results[0],'\nViT ',vit_results[0])
print('--mse (polar)--','\nCNN ',cnn_results[1],'\nResNet ',resnet_results[1],'\nViT ',vit_results[1])
print('--Mae (cm)--','\nCNN ',cnn_results[2],'\nResNet ',resnet_results[2],'\nViT ',vit_results[2])
