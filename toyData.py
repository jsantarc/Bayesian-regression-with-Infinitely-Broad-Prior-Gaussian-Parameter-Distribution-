
import numpy as np
from sklearn import  preprocessing

def toyData(w,sigma,N):
    """
    This function creates 1d polynomial toy data for linear regression 
     y= w[0]+w[1]x+..w[d]x^d 
     
    Input 
    w: numpy array of paramters (w)0 is the bias     
    sigma:Standard deviation of noise 
    N:Number of samples 

    Output
    Out[0]:array representing value of axis  value of axis 
    Out[1]: Deterministic function  
    Out[3]:Transformed design matrix 
    Out[4]: Deterministic function with Additive noise     
    """    
    #Degree of polynomial 
    degree=w.size;   
    
    #generate x values 
    x=np.linspace(0, 1,N);
    
    poly=preprocessing.PolynomialFeatures(degree-1,include_bias=True)
   
    PHI=poly.fit_transform(x.reshape(N,1))  
  
    y=np.dot(PHI,w);
    
    target=y+np.random.normal(0, sigma, N);
    
    Out=[x,y,PHI, target]

    return Out
    
    
    
    
    