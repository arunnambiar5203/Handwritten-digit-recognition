import numpy as np
from scipy import optimize
from sigmoid import sigmoid
from onevsall import onevsall
from gradient import gradient
from lrcostfunction import lrcostfunction

#checking with a small data set given below
theta_t = np.array([[-2],[-1],[1],[2]]);
X_t = np.array([[1,0.1,0.6,1.1],[1,0.2,0.7,1.2],[1,0.3,0.8,1.3],[1,0.4,0.9,1.4],[1,0.5,1,1.5]])
y_t = np.array([[1],[0],[1],[0],[1]])
lamda_t = 3
args = (X_t,y_t,lamda_t)
print(lrcostfunction(theta_t,*args))
print(gradient(theta_t,*args))
	
	

#checkgradient using check grad function
from scipy.optimize import check_grad
from sklearn.datasets import make_classification
X_examples, Y_labels = make_classification()
X_examples = np.insert(X_examples,0,1,axis = 1)
lamda = 0.1
#print(check_grad(lrcostfunction,gradient, np.zeros(np.size(X_examples,1)), X_examples, Y_labels.flatten(),lamda))





import scipy.io as sio
mat_contents = sio.loadmat('ex4data1.mat');#training data

X = mat_contents['X'];
y = mat_contents['y'];
num_labels = 10;
lamda = 0.1;
all_theta = onevsall(X,y,num_labels,lamda);
#print(np.size(all_theta,0))
m = np.size(X,0)
#predicting value and calculating accuracy
print(all_theta);
X = np.insert(X,0,1,axis = 1)
tmp = np.zeros((m,num_labels),dtype = np.float64);
tmp = sigmoid( X.dot(all_theta.T) );
print(tmp);
p = 1 + np.argmax(tmp,axis = 1);
print(p);
print(y.flatten());
print(   np.mean( (np.equal(p,y.flatten())).astype(int) * 100 ) );
