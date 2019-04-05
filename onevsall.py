from lrcostfunction import lrcostfunction 
from gradient import gradient
import numpy as np
from scipy import optimize
def onevsall(X,y,num_labels,lamda):
    m = np.size(X,0)
    n = np.size(X,1)
    all_theta = np.zeros((num_labels,n+1))
    X = np.insert(X,0,1,axis = 1)
    all_theta = np.zeros((m,n+1));
    for c in range(0, num_labels):
        initial_theta = np.zeros(n+1);
        tmp_y = y==(c+1)
        tmp_y = tmp_y.astype(int)
        x = optimize.fmin_tnc(func=lrcostfunction, x0=initial_theta,fprime = gradient,args=(X,tmp_y.flatten(),lamda))
        #print(x[0]);
        all_theta = np.insert(all_theta,c,x[0],axis = 0)
    return all_theta

	
	