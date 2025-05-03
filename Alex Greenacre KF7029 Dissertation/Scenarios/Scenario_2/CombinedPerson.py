from Models.cnn_model import CnnModel
from Models.resnet_model import ResNetModel
from Models.vit_model import ViTModel
from Utils.data_dictionary import getDataFiles
from Utils.load_data import loadData
from Plots.Scenario2 import savePersonChange
def perPersonTest(person_count):
    #Data prep 
    data_files, reference_files = getDataFiles(person_count) 
    X_train,X_test,y_train,y_test = loadData(data_files,reference_files)
    #Metric 1 - Cnn Model 
    model = CnnModel()
    model.loadModel('CNN_Model_No_Mask',False)
    cnn_results = model.getEvaluation(X_test,y_test)
    #Remove last axis 
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
    return cnn_results,resnet_results,vit_results
cnn=[]
resnet =[]
vit = []
for i in range(1,6):
    cnn_person,resnet_person,vit_person = perPersonTest(i)
    cnn.append(cnn_person[0])
    resnet.append(resnet_person[0])
    vit.append(vit_person[0])
print('cnn: ',cnn)
print('resnet: ',resnet)
print('vit: ',vit)

savePersonChange(cnn=cnn,resnet=resnet,vit=vit)