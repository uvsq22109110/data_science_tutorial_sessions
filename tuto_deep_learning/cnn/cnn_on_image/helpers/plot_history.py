import matplotlib.pyplot as plt

def plot_history(history_dict,nb_epochs):
    
    metrics = list(history_dict.keys())
    plt.style.use("ggplot")
    fig, axs = plt.subplots(1,len(metrics)//2, figsize=(5*len(metrics)//2,5))
    fig.suptitle('Training/Test performences')
    range_epochs = range(1,nb_epochs+1)
    axis_iter = 0
    for metric in metrics[:len(metrics)//2] :
        axs[axis_iter].plot(range_epochs,history_dict[metric] , label="train_"+metric)
        axs[axis_iter].plot(range_epochs, history_dict["val_"+metric], label="test_"+metric)
        axs[axis_iter].set_xlabel("Epoch #")
        axs[axis_iter].set_xlabel(metric)
        axs[axis_iter].legend(loc="best")
        axis_iter+=1
    plt.show()