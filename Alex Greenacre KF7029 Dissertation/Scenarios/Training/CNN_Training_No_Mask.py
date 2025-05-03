####
# Title - CNN Training Scenario 
# Description - Training Model For CNN Scenario with no mask (ie as a single unit) 
####

from Utils.data_dictionary import getAllFiles
from Utils.load_data import loadData
from Models.cnn_model import CnnModel
from Plots.trainingPlot import saveBothMetricResults,saveTrainingResults

file_name = 'CNN_Model_No_Mask'
data_files,reference_files = getAllFiles()
X_train,X_test,y_train,y_test = loadData(data_files,reference_files)
model = CnnModel()
model.createModel(X_train=X_train,X_test=X_test,y_train=y_train,y_test=y_test,file_name=file_name,masked=False)
results = model.getResults()
saveTrainingResults(results,file_name)
saveBothMetricResults(results,file_name)
