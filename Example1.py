
import matplotlib.pyplot as plt
import numpy as np
import BayesianRegGaussPrior as BRGP
import toyData 
from sklearn import  linear_model,preprocessing
"""
example one
This example shows how the confidence of the prediction is
dependent on the location of your training samples  
@author: Joseph
"""
#Generate polynomial toy training data  
w=np.array([-1,-1,-10,12]);
sigma=0.1
N=100;   
Out=toyData.toyData(w,sigma,N);
PHI=Out[2];
targets=Out[3];
y=Out[1]
x=Out[0];
#train model
Parameters=[0];
Model =BRGP.BayesianRegGaussPrior();
Model.fit(PHI[x>0.5,:],targets[x>0.5],Parameters);
#Generate test data
Out=toyData.toyData(w,sigma,N);
PHI=Out[2];
y=Out[1]
# generate a prediction 
y_pred=Model.predicte(PHI);
sigma=np.sqrt(Model.var(PHI));
#plot data 
plt.figure()
axes = plt.gca()
plt.plot(x,y_pred)
plt.plot(x[x>0.5],targets[x>0.5],'ro')
plt.plot(x,y,'g')
plt.fill_between(x,y_pred - 2 * sigma,y_pred + 2 * sigma, alpha=.10)
axes.set_ylim([-3,3])   
plt.legend(('Predicted function', 'Training data',' function','2 xstd'))
Ti='confidence of the prediction and location of training samples '
plt.title(Ti)
