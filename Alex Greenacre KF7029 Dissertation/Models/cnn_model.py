####
# Title - ResNet model 
# Description - baseline model that will be used to comapre against the ViT model 
# Note - A internet connection is reuqired to download the ResNet model from the keras libary 
####
from tensorflow.keras import layers, models, optimizers,ops
from Utils.loss_function import distanceLoss,distanceLossMae,distanceLossMasked,distanceLossMaeMasked
from matplotlib import pyplot as plt 

class CnnModel:
    def __init__(self):
        self.model= []
        self.results = []
        self.file_destination = 'Models/Weights/CNN/'
    

    def loadModel(self,file_name,masked=True):
        if (masked == True):
            self.model =models.load_model(self.file_destination+file_name+'.h5',custom_objects={'distanceLossMasked':distanceLossMasked})
            #Recompile model with absolute metric 
            self.model.compile(optimizer='adam',loss=distanceLossMasked, metrics=['mse',distanceLossMaeMasked])
        else:
            self.model =models.load_model(self.file_destination+file_name+'.h5',custom_objects={'distanceLoss':distanceLoss})
            self.model.compile(optimizer='adam',loss=distanceLoss, metrics=['mse',distanceLossMae])
        
    def createModel(self,X_train,X_test,y_train,y_test,file_name, masked = True ):
        self.person_count = y_train.shape[-2] 
        
        input = layers.Input(shape= X_train.shape[1:])
        model_layer = layers.Conv3D(filters=32,kernel_size=(5,5,2),activation='relu')(input)
        model_layer = layers.Dropout(0.1)(model_layer)
        model_layer = layers.BatchNormalization()(model_layer)
        model_layer = layers.MaxPooling3D((2,2,1))(model_layer)
        model_layer = layers.Conv3D(filters=64,kernel_size=(3,3,1),activation='relu')(model_layer)
        model_layer = layers.MaxPooling3D((2,2,1))(model_layer)
        model_layer = layers.Flatten()(model_layer)
        model_layer = layers.Dense(units=512)(model_layer)
        #Creates the ouput for the model, units are based off the expected person count in the scenario 
        model_layer = layers.Dense(units=self.person_count*2)(model_layer)
        output = layers.Reshape((self.person_count,2))(model_layer)
        self.model = models.Model(inputs =input,outputs=output)
        if masked == True:
            self.model.compile(optimizer='adam',loss=distanceLossMasked, metrics=['mse'])
        else:
            self.model.compile(optimizer='adam',loss=distanceLoss, metrics=['mse'])    
        print(self.getSummary())
        #print(self.data.shape,self.labels.shape)
        #X_train,X_test,y_train,y_test = train_test_split(self.data,self.labels,test_size=0.3,random_state=1)
        print('\nTrain shape\n',X_train.shape,y_train.shape,'\nTest shapes\n',X_test.shape,y_test.shape)
        self.results = self.model.fit(X_train,y_train,epochs=150,batch_size=64,validation_data=(X_test,y_test))
        self.model.save(self.file_destination+file_name+'.h5')
    
    def getResults(self):
        return self.results
    def getSummary(self):
        return self.model.summary()
    
    def getPrediction(self,pData):
        print(pData.shape)
        #pData = pData.reshape(1,137,13,2,1)
        return self.model.predict(pData)
    def getEvaluation(self,X_test,y_test):
        return self.model.evaluate(X_test,y_test) 
    
    def saveTrainingResultPlot(self,file_name):
        plt.plot(self.results.history['loss'],label='loss' )
        plt.plot(self.results.history['val_loss'],label='val loss')
        plt.savefig('Test_Results/Training/CNN_Training_Result_'+str(file_name))


