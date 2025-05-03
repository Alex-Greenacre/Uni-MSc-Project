from matplotlib import pyplot as plt 
import numpy as np

def saveCombinedObjectResults(cnn,resnet,vit):
    object_names = ['Wall','Door','Open']
    x = np.arange(3)
    plt.bar(x-0.25,cnn,width=0.25,label='cnn')
    plt.bar(x-0,resnet,width=0.25,label='resnet')
    plt.bar(x+0.25,vit,width=0.25,label='vit')
    
    plt.xlabel('Object')
    plt.ylabel('MSE (cm)')
    plt.title('Distance Error For Object Per Model')
    plt.xticks(x,object_names)
    plt.legend()
    plt.savefig('Test_Results/Scenario_4/All_Object_Perfomance')
    plt.close()
