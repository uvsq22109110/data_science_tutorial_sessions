import matplotlib.pyplot as plt 

def update_weights(X, Y, learning_rate,m=[1,1], b=1):
    
    m1_deriv = 0
    m2_deriv = 0
    b_deriv = 0
    N = len(X)
    for i in range(N):
        '''
        # fit with intercept
        # Calculate partial derivatives
        # -2x(y - (mx + b))
        m1_deriv += X[i][0] * (((m[0])*(X[i][0])+m[1]*X[i][1] + b) - Y[i])
        m2_deriv  += X[i][1] * ((m[1]*X[i][1]+m[0]*X[i][0] + b) - Y[i] )
        # -2(y - (mx + b))
        b_deriv += ((m[0]*X[i][0]+m[1]*X[i][1] + b) - Y[i] )
        '''
        # fit without intercept
        # Calculate partial derivatives
        # -2x(y - (mx))
        m1_deriv += X[i][0] * (((m[0])*(X[i][0])+m[1]*X[i][1]) - Y[i])
        m2_deriv  += X[i][1] * ((m[1]*X[i][1]+m[0]*X[i][0]) - Y[i] )
        
    # We subtract because the derivatives point in direction of steepest ascent
    m[0] -= ((m1_deriv / N) * learning_rate)
    m[1] -= ((m2_deriv / N) * learning_rate)
    #b -= ((b_deriv / N) * learning_rate)
    return [m, b]

def grad_descent_2(X, Y, learning_rate,number_iteration,m=[1,1], b=1):
    
    
    for i in range(number_iteration):
        [m, b] = update_weights(X, Y, learning_rate, m, b)
        
    return np.around(m,4),np.around(b,4)

def plot_model(x, y, w, bias=0):
    
    plt.plot(x[:,1], y, "x",label=" données initiales ")
    plt.plot(x[:,1], np.dot(np.array(x),np.array(w))+bias, "r-",label="Droite Ajustée")
    plt.legend()
    plt.title("Visualisation des des données et de la ligne ajustée ")
    plt.show()

#plot_model(x, y, m)
#plot_model(x, y, sol1)