"""
alfa and bata parameter  
see Bishop, Christopher M. "Pattern recognition." Machine Learning 128 (2006) Chapter three 
"""

import matplotlib.pyplot as plt
import numpy as np
import BayesianRegGaussPrior as BRGP
import toyData 

    
w=np.array([1,2,-10,5]);
sigma=0.5
N=100;   

#training data
Out=toyData.toyData(w,sigma,N);
PHI=Out[2];
targets=Out[3];
y=Out[1]
x=Out[0];
#train model
Parameters=[1, 2];
Model =BRGP.BayesianRegGaussPrior();
Model.fit(PHI,targets,Parameters);

#test data
Out=toyData.toyData(w,sigma,N);
PHI=Out[2];
#targets=Out[3];
y=Out[1]
x=Out[0];
  
#predicte model
y_pred=Model.predicte(PHI);
sigma=np.sqrt(Model.var(PHI));



plt.figure()
axes = plt.gca()
plt.plot(x,y_pred)
plt.plot(x,targets,'ro')
plt.plot(x,y,'g')
plt.fill_between(x,y_pred - 2 * sigma,y_pred + 2 * sigma, alpha=.10)
axes.set_ylim([-3,3])   
plt.legend(('Predicted function', 'targets',' function','2 xstd'))
Ti='Testing Data Predicted function with N='+str(N);
plt.title(Ti)





