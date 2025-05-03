from matplotlib import pyplot as plt 

def saveCombinedModelResults(cnn,resnet,vit):
    model_names = ['cnn','resnet','vit']    
    model_vals = [cnn,resnet,vit]
    plt.bar(model_names,model_vals,color='black')
    #Add text to center of bars 
    for i in range(0,len(model_vals)):
        plt.text(i,model_vals[i]/2,str(int(model_vals[i]))+' cm',ha='center',color='white')
    plt.xlabel('Model')
    plt.ylabel('MSE (cm)')
    plt.title('Distance Error For Models Per Scenario')
    plt.savefig('Test_Results/Scenario_1/All_Model_Perfomance')
    plt.close()
