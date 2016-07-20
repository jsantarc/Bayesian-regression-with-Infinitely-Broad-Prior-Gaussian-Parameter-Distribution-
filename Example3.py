"""
Kernel interpolation using polynomial kernel  
"""
import matplotlib.pyplot as plt
import numpy as np
import BayesianRegGaussPrior as BRGP
import toyData 

#Parameters     
w=np.array([1,1,1]);
sigma=0.1
N=100;   
#toy training data
Out=toyData.toyData(w,sigma,N);
PHI=Out[2];
targets=Out[3];
y=Out[1]
x=Out[0];
#Train model 
Model =BRGP.BayesianRegGaussPrior('polynomial');
Model.fit(PHI,targets,[1,1,2]);
#Test toy data 
Out=toyData.toyData(w,sigma,N);
PHI=Out[2];
targets=Out[3];
y=Out[1]
x=Out[0];
#Prediction 
yhat=Model.predicte(PHI);
sigma=np.sqrt(Model.var(PHI));
#plot data 
plt.plot(x,yhat,'b')
plt.plot(x,y,'g')
plt.plot(x,targets,'ro')
plt.fill_between(x,yhat - 2 * sigma,yhat+ 2 * sigma, alpha=.10)
axes.set_ylim([-3,3])   
plt.legend(('Predicted function',' function','targets','2 xstd'))
Ti='Testing Data Predicted function with N='+str(N);
plt.title(Ti)
