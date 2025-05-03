####
# Title - ResNet Training Scenario 
# Description - Training Model For ResNet Scenarios 
####

from Utils.data_dictionary import getAllFiles
from Utils.load_data import loadData
from Models.vit_model import ViTModel
from Plots.trainingPlot import saveBothMetricResults,saveTrainingResults

file_name = 'ViT_Model'
data_files,reference_files = getAllFiles()
X_train,X_test,y_train,y_test = loadData(data_files,reference_files)
X_train = X_train.reshape(X_train.shape[:-1])
X_test = X_test.reshape(X_test.shape[:-1])

model = ViTModel()
model.createModel(X_train=X_train,X_test=X_test,y_train=y_train,y_test=y_test,file_name=file_name)
results = model.getResults()
saveTrainingResults(results,file_name)
saveBothMetricResults(results,file_name)
