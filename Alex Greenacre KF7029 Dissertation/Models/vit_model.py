####
#Title - Vit Model 
#Description - Novel model will be used to check the models efffectivness against the radar scenarios 
#Note - Model based off the ViT model example produced by Salama K
#Reference - Salama K.(2021)'Keras documentation: Image classification with Vision Transformer' 
#            Available at: https://keras.io/examples/vision/image_classification_with_vision_transformer/ 
#            (Accessed:21/08/2024)
####
from tensorflow.keras import layers,models,optimizers
from Utils.loss_function import distanceLoss,distanceLossMasked,distanceLossMae,distanceLossMaeMasked
import tensorflow as tf
from Models.Layers.patchEmbeddingLayer import PatchEmbeddingLayer
from matplotlib import pyplot as plt 
class ViTModel:
    def __init__(self):
        
        self.file_destination = 'Models/Weights/ViT/'
    def loadModel(self,file_name,masked=True):
        adam = optimizers.Adam(clipnorm=1)
        
        if masked == True:
            self.model =models.load_model(self.file_destination+file_name+'.h5',custom_objects={'distanceLossMasked':distanceLossMasked,
                                                                                                'PatchEmbeddingLayer':PatchEmbeddingLayer})
            self.model.compile(optimizer=adam,loss=distanceLossMasked, metrics=['mse',distanceLossMaeMasked])
        
        else:
            self.model =models.load_model(self.file_destination+file_name+'.h5',custom_objects={'distanceLoss':distanceLoss,
                                                                                                'PatchEmbeddingLayer':PatchEmbeddingLayer})    
            self.model.compile(optimizer=adam,loss=distanceLoss, metrics=['mse',distanceLossMae])
        
    def createModel(self,X_train,X_test,y_train,y_test,file_name='ViT', masked=True):
        self.person_count = y_train.shape[-2] 
        input = layers.Input(shape = X_train.shape[1:])
        patches = PatchEmbeddingLayer()(input)
        #ViT block 
        model_layer = layers.LayerNormalization()(patches)
        attention = layers.MultiHeadAttention(num_heads=12,key_dim=128,dropout=0.1)(model_layer,model_layer)
        model_layer = layers.Add()([attention,patches])
        mlp_layer = layers.LayerNormalization()(model_layer)
        #Creates the mlp layer to update the token  
        for x in range(0,8):
            mlp_layer = layers.Dense(units=128,activation='relu')(mlp_layer)
        #Update encoded vectors and flatten for MLP Head    
        patches = layers.Add()([mlp_layer,model_layer])
        model_layer = layers.Flatten()(patches)
        #Mlp head 
        model_layer = layers.Dense(units=1024,activation='relu')(model_layer) 
        model_layer = layers.Dropout(0.1)(model_layer)
        model_layer = layers.Dense(units=512,activation='relu')(model_layer) 

        #Converts Mlp output to degree and distance for each person 
        model_layer = layers.Dense(units=self.person_count*2)(model_layer)
        output = layers.Reshape((self.person_count,2))(model_layer)
        self.model = models.Model(inputs=input,outputs=output)
        # Clipnorm stops Gradient Explosion 
        adam = optimizers.Adam(clipnorm=1)
        if masked == True:
            self.model.compile(optimizer=adam,loss=distanceLossMasked, metrics=['mse'])
        else:
            self.model.compile(optimizer=adam,loss=distanceLoss, metrics=['mse'])
        
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
    def plotTrainingResults(self):
        plt.plot(self.results.history['loss'],label='loss' )
        plt.plot(self.results.history['val_loss'],label='val loss')
        plt.show()

    def saveTrainingResultsPlot(self,file_name):
        plt.plot(self.results.history['loss'],label='loss' )
        plt.plot(self.results.history['val_loss'],label='val loss')
        plt.savefig('../Test_Results/Training/ViT_Training_Result_'+str(file_name))

