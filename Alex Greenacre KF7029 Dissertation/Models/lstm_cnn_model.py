from tensorflow.keras import layers, models, optimizers,ops
from Utils.load_data import loadData
from Utils.loss_function import mseDistance
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt 

class LstmCnn:
    def __init__(self,pData,pLabels):
        self.model= []
        self.results = []
        self.person_count = pLabels.shape[1]
        self.data = pData
        self.labels = pLabels

    def buildModel(self):
        input = layers.Input(shape= self.data.shape[1:])
        #mask = layers.Masking(mask_value=-360)(input)
        model_layer = layers.ConvLSTM3D(filters=32,kernel_size=(2,1,2),activation='relu',return_sequences=True)(input)
        model_layer = layers.BatchNormalization()(model_layer)
        model_layer = layers.ConvLSTM3D(filters=64,kernel_size=(4,4,1),activation='relu')(model_layer)
        model_layer = layers.MaxPooling3D((2,2,1))(model_layer)
        model_layer = layers.Conv3D(filters=64,kernel_size=(2,2,1))(model_layer)
        #model_layer = layers.MaxPooling3D((2,2,1))(model_layer)
        model_layer = layers.Flatten()(model_layer)
        model_layer = layers.Dense(units=512)(model_layer)
        #Creates the ouput for the model, units are based off the expected person count in the scenario 
        model_layer = layers.Dense(units=self.person_count*self.labels.shape[-1])(model_layer)
        output = layers.Reshape((self.person_count,self.labels.shape[-1]))(model_layer)
        self.model = models.Model(inputs =input,outputs=output)
        self.model.compile(optimizer='adam',loss=mseDistance, metrics=['mse'])
    
    def trainModel(self):
        print(self.data.shape,self.labels.shape)
        X_train,X_test,y_train,y_test = train_test_split(self.data,self.labels,test_size=0.3,random_state=1)
        print('\nTrain shape\n',X_train.shape,y_train.shape,'\nTest shapes\n',X_test.shape,y_test.shape)
        self.results = self.model.fit(X_train,y_train,epochs=25,batch_size=64,validation_data=(X_test,y_test))
    def getResults(self):
        return self.results
    def getSummary(self):
        return self.model.summary()
    
    def getPrediction(self,pData):
        print(pData.shape)
        pData = pData.reshape(1,137,13,2,1)
        return self.model.predict(pData)
    def plotTrainingResults(self):
        plt.plot(self.results.history['loss'],label='loss' )
        plt.plot(self.results.history['val_loss'],label='val loss')
        plt.show()
    