from Models.cnn_model import CnnModel
from Models.resnet_model import ResNetModel
from Models.vit_model import ViTModel
from Utils.data_dictionary import getAllFiles
from Utils.load_data import loadData
#Data prep 
data_files, reference_files = getAllFiles() 
X_train,X_test,y_train,y_test = loadData(data_files,reference_files)
#Metric 1 - Cnn Model 
model = CnnModel()
model.loadModel('Cnn_Model_No_Mask',False)
cnn_results = model.getEvaluation(X_test,y_test)

#Metric 2 - ResNet Model 
X_test = X_test.reshape(X_test.shape[:-1])
model = ResNetModel()
model.loadModel('ResNet_Model_No_Mask',False)
resnet_results = model.getEvaluation(X_test,y_test)

#Metirc 3 - ViT Model 
model = ViTModel()
model.loadModel('ViT_Model_No_Mask',False)
vit_results = model.getEvaluation(X_test,y_test)

X_train,X_test,y_train,y_test = loadData(data_files,reference_files)

#Metric 1 - Cnn Model 
model = CnnModel()
model.loadModel('Cnn_Model')
cnn_results_mask = model.getEvaluation(X_test,y_test)

X_test = X_test.reshape(X_test.shape[:-1])
#Metric 2 - ResNet Model 
model = ResNetModel()
model.loadModel('ResNet_Model')
resnet_results_mask = model.getEvaluation(X_test,y_test)

#Metirc 3 - ViT Model 
model = ViTModel()
model.loadModel('ViT_Model')
vit_results_mask = model.getEvaluation(X_test,y_test)
print(
    'No Mask\n-------',
    '\ncnn:    mse-',cnn_results[0],' mae-',cnn_results[-1],
    '\nresnet: mse-',resnet_results[0],' mae-',resnet_results[-1],
    '\nvit:    mse-',vit_results[0],' mae-',vit_results[-1],
    '\nMask\n----',
    '\ncnn:    mse-',cnn_results_mask[0],' mae-',cnn_results_mask[-1],
    '\nresnet: mse-',resnet_results_mask[0],' mae-',resnet_results_mask[-1],
    '\nvit:    mse-',vit_results_mask[0],' mae-',vit_results_mask[-1],
)

