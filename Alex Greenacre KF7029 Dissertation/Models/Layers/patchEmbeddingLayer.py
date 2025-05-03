####
# Title - Patch layer 
# Description - A layer that takes radar data and splits into patches 
#Note - Layer based off the ViT model example produced by Salama K
#Reference - Salama K.(2021)'Keras documentation: Image classification with Vision Transformer' 
#            Available at: https://keras.io/examples/vision/image_classification_with_vision_transformer/ 
#            (Accessed:21/08/2024)
####
from tensorflow.keras import layers,ops
class PatchEmbeddingLayer(layers.Layer):
    #Kwargs used for custom layer to pass in arguments in model loading  
    def __init__(self,**kwargs):
        super().__init__()
        self.freq_split = 13
        self.antenna_split = 2
        self.freq_patches = 137//self.freq_split
        self.antenna_patches = 13//self.antenna_split
        self.amount_of_patches = self.freq_patches * self.antenna_patches
        self.projection = layers.Dense(units=128)
        self.position_embedding = layers.Embedding(input_dim=self.amount_of_patches,output_dim=128) 
    def call(self,pData):
        #Splits the radar into patches of size (13,2,2)
        patches = ops.image.extract_patches(pData,size=(13,2))
        #Flattens the patches into a shape of (batch,patch,data)
        patches = ops.reshape(patches,(ops.shape(pData)[0],
                                                 self.amount_of_patches,
                                                 self.freq_split*self.antenna_split*2))
        #Assigns the order of the patches relatice to its original position (ie first entry of radar is pos 1) 
        positions = ops.expand_dims(ops.arange(0,self.amount_of_patches,1),axis=0)
        #Assigns a learnable token to the patch  
        projected_patches = self.projection(patches)
        embedded_positions = self.position_embedding(positions)
        #Combines data to format - token/vector,position
        encoded = projected_patches + embedded_positions
        return encoded