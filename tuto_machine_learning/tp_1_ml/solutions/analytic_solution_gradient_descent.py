def analytical_resolution(X,Y):
    
    '''
        cette fonction utilsiera de l'algébre linéaire pour 
        résoudre en une deux ou trois ligne, la descente de gradient. 
    '''
    
    return np.around(np.dot(np.dot(np.linalg.inv(np.dot(x.T,x)),x.T),y),4)
