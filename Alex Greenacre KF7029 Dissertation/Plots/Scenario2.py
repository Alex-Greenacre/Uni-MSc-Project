from matplotlib import pyplot as plt 

def savePersonChange(cnn,resnet,vit):
    plt.plot([x for x in range(1,6)],cnn,label='cnn' )
    plt.plot([x for x in range(1,6)],resnet,label='resnet')
    plt.plot([x for x in range(1,6)],vit,label='vit' )
    plt.xticks([x for x in range(1,6)],labels=[x for x in range(1,6)])
    plt.xlabel('People In Scenario')
    plt.ylabel('MSE (cm)')
    plt.title('Distance Error For People Count Per Scenario')
    plt.legend()
    plt.savefig('Test_Results/Scenario_2/All_Model_Perfomance_Displayed')
    plt.close()
