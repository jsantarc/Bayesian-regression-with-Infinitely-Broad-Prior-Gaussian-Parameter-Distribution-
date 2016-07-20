"""
Class BayesianRegGaussPrior: Bayesian regression with Gaussian prior Parameter distribution this code  only considers the Infinitely Broad Prior and the Equivalent kernel. If the Parameter “kind “ is left empty the default  feature vectors  is used, else the parameter  kind corresponds to the type of kernel.
Methods and Parameters 
fit: trains the model 
Parameters 
PHI: feature vectors rows correspond to observations, columns correspond to variables 
Targets: targets 
Parameters: parameters of model

If kind is a “basis” i.e a basis function model then then there are two options

1a)If there is one input the parameter corresponds to the “quadratic regularization term” used in ridge regression and Bata is 
calculated using the MAP estimate 

1b) If there are two parameters the model alfa and bata see [1]

If a kernel is used the parameters correspond to kernel parameters

predicte(self,PHI): predicts the output given feature vectors 
PHI:  feature vectors rows correspond to observations, columns correspond to variables 

var(self,PHI): predicts variance of estimate  

References: 
[1] Bishop, Christopher M. "Pattern recognition." Machine Learning 128 (2006) Chapter three
[2]http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.pairwise_kernels.html
"""

import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels

class BayesianRegGaussPrior:
    
   def __init__(self,kind=None):
       
     if kind is None:  
         kind='basis';
   
     self.kind=kind;
     self.Parameters=[];
     self.targets=[];
     self.Parameters=[]; 
     self.Sn=[];
     self.Mn=[];
     self.y_pred=[];
     self.bata=[];
     self.sigma_train=[];
     self.sigma=[];
     self.trainingdata=[];
     self.K=[];
     
   def fit(self,PHI,targets, Parameters):
  
       if (self.kind=='rbf'or self.kind=='sigmoid'or self.kind=="polynomial" or self.kind== "lin" or self.kind =="cosine"):
           
           # save target need for kernel interpellation  
           self.targets=targets;
           
           #parameters  
           self.Parameters=Parameters;
           #Kernel matrix 
           K=pairwise_kernels(PHI[:,:],PHI[:,:],metric=self.kind,filter_params= self.Parameters);
           
          #To make a prediction on sample x using n-training sample  sum with respect to xn i.e sum  k(x,xn) 
          #must equal one i.e   sum  k(x,xn)=1
           Normalization=np.power(np.tile(np.sum(K,1),(K.shape[1], 1)),-1) ;
           K=Normalization*K;
           
           # Prediction using training data 
           y_pred=np.dot(K,targets);
           
           # Prediction varance  
           self.bata=np.var(targets-y_pred);
           # Prediction variance of each sample  
           self.sigma=(self.bata)+np.diag(K)/(self.bata);
           
           self.trainingdata=PHI;
           
  
       if (self.kind=='basis'):
            self.targets=targets;
            self.Parameters;
            
            dim=PHI.shape[1];
            S0=np.identity(dim);
            Parameter=np.array(Parameters);
    
            if (Parameter.size==1):
                #zero mean ,broad prior, with one parameter with maximum likelihood estimation of prior 
                Lambda=Parameter[0];
                self.Sn=np.linalg.inv(Lambda*S0+np.dot(PHI.transpose(),PHI))
                self.Mn=np.dot(self.Sn, np.dot(PHI.transpose(),targets))
                # Prediction of training data 
                y_pred=np.dot(PHI,self.Mn)
                self.bata=np.var(targets-y_pred);
                self.sigma=(self.bata)+np.diag(np.dot(PHI,np.dot(self.Sn,PHI.transpose())));

            if (Parameter.size==2):
                 #zero mean ,broad prior, with two parameter 
                alfa=Parameter[0];
                bata=Parameter[1];
                self.Sn=np.linalg.inv(alfa*S0+bata*np.dot(PHI.transpose(),PHI))
                self.Mn=bata*np.dot(self.Sn, np.dot(PHI.transpose(),targets))
                # Prediction of training data 
                y_pred=np.dot(PHI,self.Mn)
                #Calculate noise variance on training data using MAP estimate 
                self.bata=np.var(targets-y_pred);
                # prediction variance on training data  
                self.sigma=(self.bata)+np.diag(np.dot(PHI,np.dot(self.Sn,PHI.transpose())));
            
                
   def predicte(self,PHI):
       
       if (self.kind=='basis'):
           yhat=np.dot(PHI,self.Mn);
           return yhat
               
       if (self.kind=='rbf'or self.kind=='sigmoid'or self.kind=="polynomial" or self.kind== "lin" or self.kind =="cosine"):
           
           self.K=pairwise_kernels(self.trainingdata,PHI[:,:],metric=self.kind,filter_params=self.Parameters)
           #To make a prediction on sample x using n-training sample  sum with respect to xn i.e sum  k(x,xn) 
          #must equal one i.e   sum  k(x,xn)=1
           Normalization=np.power(np.tile(np.sum(self.K,1),(self.K.shape[1], 1)),-1) ;

           self.K=Normalization*self.K;
           
           # Prediction using training data 
           yhat=np.dot(self.K,self.targets);   
           # Prediction variance of each sample  
           self.sigma=(self.bata)+np.diag(self.K)/(self.bata);
           return yhat
           
   def var(self,PHI):
       
       if (self.kind=='basis'):
           #Variance of prediction for each of the testing samples  
           sigma=(self.bata)+np.diag(np.dot(PHI,np.dot(self.Sn,PHI.transpose())));
           return sigma;
        
       if (self.kind=='rbf'or self.kind=='sigmoid'or self.kind=="polynomial" or self.kind== "lin" or self.kind =="cosine"):
           sigma=np.zeros((PHI.shape[0])); 
           #Variance of prediction for each of the testing samples  
           sigma=(self.bata)+np.diag(self.K)/(self.bata);
           return sigma;
           
   
      