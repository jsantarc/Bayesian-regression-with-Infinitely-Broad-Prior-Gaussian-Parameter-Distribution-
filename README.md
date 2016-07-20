# Bayesian-regression-with-Infinitely-Broad-Prior-Gaussian-Parameter-Distribution-
Class implements the Bayesian regression with Gaussian prior Parameter distribution this code only considers the Infinitely Broad Prior and the Equivalent kernel. 

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
"""