from Models.cnn_model import CnnModel
from Models.resnet_model import ResNetModel
from Models.vit_model import ViTModel
from Utils.data_dictionary import getAllFiles
from Utils.load_data import loadData
from Plots.Scenario1 import saveCombinedModelResults
#Data prep 
data_files, reference_files = getAllFiles() 
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
saveCombinedModelResults(cnn=cnn_results[0],resnet=resnet_results[0],vit=vit_results[0])
print('--Loss (mse cm)--','\nCNN ',cnn_results[0],'\nResNet ',resnet_results[0],'\nViT ',vit_results[0])
print('--mse (polar)--','\nCNN ',cnn_results[1],'\nResNet ',resnet_results[1],'\nViT ',vit_results[1])
print('--Mae (cm)--','\nCNN ',cnn_results[2],'\nResNet ',resnet_results[2],'\nViT ',vit_results[2])
