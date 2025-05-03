from matplotlib import pyplot as plt 

def saveTrainingResults(results,file_name):
    plt.plot(results.history['loss'],label='loss' )
    plt.plot(results.history['val_loss'],label='val loss')
    plt.xlabel('epochs')
    plt.ylabel('mse loss (cm)')
    plt.title(file_name)
    plt.legend()
    plt.savefig('Test_Results/Training/'+str(file_name)+'_Training_Results')
    plt.close()
def saveBothMetricResults(results,file_name):
    fig,ax = plt.subplots(2)
    fig.suptitle(file_name)
    ax[0].plot(results.history['loss'],label='custom loss')
    ax[0].plot(results.history['val_loss'],label='val custom loss')
    ax[1].plot(results.history['mse'],label='mse')
    ax[1].plot(results.history['val_mse'],label='val_mse')
    ax[0].set_title('Loss')
    ax[1].set_title('Mse')
    ax[0].legend()
    ax[1].legend()

    plt.savefig('Test_Results/Training/'+str(file_name)+'_Both_Metrics_Training_Results')
    plt.close()

def plotTrainingResults(results):
        plt.plot(results.history['loss'],label='loss' )
        plt.plot(results.history['val_loss'],label='val loss')
        plt.show()
