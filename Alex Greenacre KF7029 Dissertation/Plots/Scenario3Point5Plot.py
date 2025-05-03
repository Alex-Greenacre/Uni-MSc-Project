from matplotlib import pyplot as plt 

def savePeromanceChange(cnn,resnet,vit):
    plt.plot([x for x in range(1,6)],cnn,label='cnn' )
    plt.plot([x for x in range(1,6)],resnet,label='resnet')
    plt.plot([x for x in range(1,6)],vit,label='vit' )
    plt.xticks([x for x in range(1,6)],labels=[x for x in range(1,6)])
    plt.xlabel('People In Scenario')
    plt.ylabel('Perfomance Increase(%)')
    plt.title('Peromance Increase from Unmaksed to Masked')
    plt.legend()
    plt.savefig('Test_Results/Scenario_3_Point_5/Perfomance_Report')
    plt.close()
