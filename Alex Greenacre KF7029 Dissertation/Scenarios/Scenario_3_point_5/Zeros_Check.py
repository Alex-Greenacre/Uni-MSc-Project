from Models.cnn_model import CnnModel
from Models.resnet_model import ResNetModel
from Models.vit_model import ViTModel
from Utils.data_dictionary import getDataFiles
from Utils.load_data import loadData
from Plots.Scenario3Point5Plot import savePeromanceChange 
#Data prep 
def getDataPerGroup(amount_of_people):
    data_files, reference_files = getDataFiles(amount_of_people) 
    X_train,X_test,y_train,y_test = loadData(data_files,reference_files)
    #Metric 1 - Cnn Model 
    model = CnnModel()
    model.loadModel('CNN_Model')
    cnn_results = model.getEvaluation(X_test,y_test)
    #Remove last axis 
    X_test = X_test.reshape(X_test.shape[:-1])

    #Metric 2 - ResNet Model 
    model = ResNetModel()
    model.loadModel('ResNet_Model')
    resnet_results = model.getEvaluation(X_test,y_test)

    #Metirc 3 - ViT Model 
    model = ViTModel()
    model.loadModel('ViT_Model')
    vit_results = model.getEvaluation(X_test,y_test)
    cnn = [cnn_results[0]]
    vit =[vit_results[0]]
    resnet=[resnet_results[0]]

    #Reload Files with original shape, as random is locked in test train split this will be the same vals 
    X_train,X_test,y_train,y_test = loadData(data_files,reference_files)
    #Metric 1 - Cnn Model 
    model = CnnModel()
    model.loadModel('CNN_Model_No_Mask',False)
    cnn_results = model.getEvaluation(X_test,y_test)
    X_test = X_test.reshape(X_test.shape[:-1])
    #Metric 2 - ResNet Model 
    model = ResNetModel()
    model.loadModel('ResNet_Model_No_Mask',False)
    resnet_results = model.getEvaluation(X_test,y_test)
    #Metirc 3 - ViT Model 
    model = ViTModel()
    model.loadModel('ViT_Model_No_Mask',False)
    vit_results = model.getEvaluation(X_test,y_test)
    
    cnn.append(cnn_results[0])
    resnet.append(resnet_results[0])
    vit.append(vit_results[0])
    return cnn,resnet,vit
def getPerfomanceChange(array):
    return(((array[0]-array[1])/array[0])*100)
cnn = []
resnet=[]
vit =[]
for i in range(1,6):
    cnn_person,resnet_person,vit_person = getDataPerGroup(i)
    cnn.append(getPerfomanceChange(cnn_person))
    resnet.append(getPerfomanceChange(resnet_person))
    vit.append(getPerfomanceChange(vit_person))
print('vit: ',vit,'\nresnet: ',resnet,'\ncnn: ',cnn)
savePeromanceChange(cnn,resnet,vit)