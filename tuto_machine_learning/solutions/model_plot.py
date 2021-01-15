import matplotlib.pyplot as plt 
def plot_model(x, y, w):
    
    plt.plot(x[:,1], y, "x",label=" données initiales ")
    plt.plot(x[:,1], np.dot(np.array(x),np.array(w)), "r-",label="Droite Ajustée")
    plt.legend()
    plt.title("Visualisation des des données et de la ligne ajustée ")
    plt.show()
    