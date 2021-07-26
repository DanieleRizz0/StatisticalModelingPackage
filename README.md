# StatisticalModelingPackage
Package for Python with tools for producing and evaluating statistical models.
The idea to build this package came to me when I couldn't find any package that implements validation tools for statistical models in Python.
The heart of the package is in the class called "linear" and its methods.

Once you have defined a linear object that provides, regressors and target values as numpy series or pandas, it is essential to compute the model (find the coefficients and bias values).
It is possible to accomplish this task in two different ways: using the normal equation or with batch gradient descent.

As you may know, once a model has been provided, it is essential to validate it. 

The first thing to do is a hypothesis test on the regressors used to explain the objective. The test implemented in this code is the one based on the F-distribution. 
A crucial aspect to investigate is the problem of multicollinearity. With this code it is possible to calculate the VIF value that give an idea, based on the correlation between the different
explanatory variables.

To make all the analysis easier the "summary" method is defined. This function give an overview of the model and of the validation.

Much is about to come (heteroskedasticity, autocorrelation, linearity, outliers...).
