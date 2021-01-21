
def gradient_descent(x,y,learning_rate,nb_iterations):
    
    '''
        Cette fonction devrait retourner la liste 
        des estimateurs qui sont arrondis au 1000ème 
    '''
    
    
    length_data, nb_columns = x.shape
    weights = np.ones(nb_columns) # [0...0] with a len == Nb Columns
    
    for i in range(0, nb_iterations):
        
        # Forward pass: compute predicted y
        y_pred = np.dot(x, weights)
        
        # Compute and print loss
        loss = y_pred - y
        cost = np.sum(loss ** 2)
        if(i % 40000 == 0):
            print("After {} itérations, loss = {}".format(i, cost))

        # Backprop to compute gradients    
        gradient = np.dot(x.T, loss) / length_data
        
        # Update weights
        weights = weights - learning_rate * gradient
    
    return np.around(weights,4)

