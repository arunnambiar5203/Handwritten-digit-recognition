from sigmoid import sigmoid
import numpy as np
def lrcostfunction(theta,*args):
     
    X,y,lamda= args;
    l = np.size(X,1);
    theta = np.reshape(theta,(l,1));  
	#reshaped because theta passed is of form ([1,2,3,4]) as required in fmin_tnc while implementation required ([[1],[2],[3],[4]])
    m = np.size(X,0);
    y = np.reshape(y,(m,1));
    #reshaped as y is flattened before passing in fmin_tnc
    
    h = sigmoid( X.dot(theta) ); #hypotheses
    #for implementing regularization
    tmp = theta * theta;
    tmp = tmp[1:np.size(tmp),];

    j = (-(1/m) * np.sum( y * np.log(h) + (1-y) * np.log( 1 -h) ) )+ (lamda/(2*m))*np.sum(tmp);
    
    return j