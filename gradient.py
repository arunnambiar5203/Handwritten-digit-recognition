

from sigmoid import sigmoid
import numpy as np
def gradient(theta,*args):
    #again y and theta reshaped for same reason 
    X,y,lamda = args;
    l = np.size(X,1);
    theta = np.reshape(theta,(l,1));
    m = np.size(X,0);
    y = np.reshape(y,(m,1));
    h = sigmoid( X.dot(theta) );
    
    grad = (1/m) * X.T.dot( h-y );
    grad[1:np.size(grad),] = grad[1:np.size(grad),] + (lamda/m)*theta[1:np.size(theta),] ;
   
    return grad.ravel()
