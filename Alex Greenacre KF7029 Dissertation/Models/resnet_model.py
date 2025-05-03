####
# Title - ResNet model 
# Description - baseline model that will be used to comapre against the ViT model 
# Note - A internet connection is reuqired to download the ResNet model from the keras libary 
####

from keras_cv.models import ResNet34Backbone
from tensorflow.keras import layers,models
from Utils.loss_function import distanceLoss,distanceMetric,distanceLossMasked,distanceLossMae,distanceLossMaeMasked
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt 

class ResNetModel: 
    def __init__(self):
        self.file_destination = 'Models/Weights/ResNet/'
        self.model = None
    def loadModel(self,file_name,masked=True):
        if masked == True:
            self.model =models.load_model(self.file_destination+file_name+'.h5',custom_objects={'distanceLossMasked':distanceLossMasked})
            self.model.compile(optimizer='adam',loss=distanceLossMasked, metrics=['mse',distanceLossMaeMasked])
        
        else:
            self.model =models.load_model(self.file_destination+file_name+'.h5',custom_objects={'distanceLoss':distanceLoss})    
            self.model.compile(optimizer='adam',loss=distanceLoss, metrics=['mse',distanceLossMae])
        
    def createModel(self,X_train,X_test,y_train,y_test,file_name='ResNet',masked=True):
        self.person_count = y_train.shape[-2]
        print(X_train.shape,self.person_count)
        #Resnet model imported from keras, setting include_top allows for custom input shape 
        input = layers.Input(shape=X_train.shape[1:])
        model_base = ResNet34Backbone(input_tensor=input,load_weights=False,include_rescaling=False)
        model_layer= model_base.output
        model_layer = layers.Flatten()(model_layer)
        model_layer = layers.Dense(units=self.person_count*2)(model_layer)
        output = layers.Reshape((self.person_count,2))(model_layer)
        self.model = models.Model(inputs=input,outputs=output)
        #If model masked 
        if masked == True:
            self.model.compile(optimizer='adam',loss=distanceLossMasked, metrics=['mse'])
        else:
            self.model.compile(optimizer='adam',loss=distanceLoss, metrics=['mse'])
        print(self.getSummary())
        print('\nTrain shape\n',X_train.shape,y_train.shape,'\nTest shapes\n',X_test.shape,y_test.shape)
        self.results = self.model.fit(X_train,y_train,epochs=150,batch_size=64,validation_data=(X_test,y_test))
        self.model.save(self.file_destination+file_name+'.h5')
    def getSummary(self):
        return self.model.summary()

    def getResults(self):
        return self.results

    def getEvaluation(self,X_test,y_test):
        return self.model.evaluate(X_test,y_test)    

    def saveTrainingResultsPlot(self,file_name):
        plt.plot(self.results.history['loss'],label='loss' )
        plt.plot(self.results.history['val_loss'],label='val loss')
        plt.savefig('../Test_Results/Training/ResNet_Training_Result_'+str(file_name))


