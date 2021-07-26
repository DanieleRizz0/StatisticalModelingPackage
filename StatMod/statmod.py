import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import f, norm, t
from IPython import display
from scipy import stats



class linear:
    
    def __init__(self, x, y, pred = None, w = None, state = 0, columns = None):
        
        self.columns = [i for i in range(x.shape[1])]
        self.target = ""
        
        if (str(type(x)) == "<class 'pandas.core.frame.DataFrame'>"):
            self.columns = list(x.columns)
            self.columns.insert(0, "Intercept")
            x = x.to_numpy()
        
        if (str(type(y)) == "<class 'pandas.core.frame.DataFrame'>"):
            y = y.to_numpy()
        
        
        if (x.ndim > 1):
            self.x = np.vstack((np.ones(len(x[:,0])), np.array(x).T)).T
            self.w = np.zeros(len(x[0]) + 1 )
        elif(x.ndim == 1):
            self.x = np.vstack((np.ones(len(x)), np.array(x).T)).T
            self.w = np.zeros(len(x) + 1 )
            
        self.y = np.array(y)
        self.pred = pred

        self.state = state

        
            
    ### print for linear object ###
    def pprint(self):
        print(self.x)
        print(self.y)
        print("lo stato è: ", self.w)

        
    
    
    ### plot of the model and real points ###
    def show(self, position = 1):
        if self.state == 0:
            print("not useful until you FIT a function. Try the 'fit' method on your linear object")
        else :
            h = np.linspace(max(self.x[:,position]) + 2, min(self.x[:,position]) - 2, 1000)
            plt.plot(h, self.w[0] + self.w[position] * h, color = "red")
            plt.scatter(self.x[:,position], self.y)
            
            
            
    ### inference phase ###        
    def predict(self, z):
        if self.state == 0:
            print("not useful until you FIT a function. Try the 'fit' method on your linear object")
        else :
            return np.inner(self.w, z)

        
        
#############################################################################################
################ Machine learning ###########################################################
#############################################################################################
            
    ### gradient descent ###
    def gradient(self, visual = False, nint = 10000, eta = 0.001):
            
            self.state = 1
            q = [0 for i in range(nint)]

            for i in range(nint):
                
                dZ = (self.y - np.dot(self.x, self.w)) * eta
                m = len(self.y)
                dW = 1. / m * (np.dot(dZ, self.x))
                
                self.w = self.w + dW
                
                nn = np.linalg.norm(dW)
                q[i] = nn 

                
            if(visual):
        
                for j in range(nint):
                
                    plt.plot(q)
                    
                    display.clear_output(wait=True)
                    plt.grid()
                
                    plt.ylim(0, 1.3*max(q))
                    plt.xlim(-1,j+1)
                
                    plt.show()
                    
            self.pred = []
            for j in self.x:
                self.pred.append(float(np.inner(j,self.w)))
            self.pred = np.array(self.pred)
            self.state = 1
            return self.w

        

#############################################################################################
################ statistical modeling #######################################################
#############################################################################################
    
    ## Exhaustive plot of data ##
    def panels(self):
        sns.pairplot(pd.DataFrame(self.x[:,1:]), corner=True)
    
    
    ## Estimation of model ##
    def normal(self):
        t1 = np.dot( np.transpose(self.x), self.x)
        t2 = np.dot( np.linalg.inv(t1), np.transpose(self.x) )
        self.w = np.dot( t2, self.y )
        
        self.pred = []
        for j in self.x:
            self.pred.append(float(np.inner(j,self.w)))
        
        self.pred = np.array(self.pred)
        self.state = 1
        return self.w
    
    
    ## Model Variance ##
    def mss(self):
        m = np.mean(self.y)
        n = len(self.y)
        a = 0
        for i in self.x:
            a += (m - np.inner(i,self.w))**2
        return float(a)
    
    
    ## Residual Variance ##
    def rss(self):
        return sum(np.power(self.res(),2))
         
    
    
    ## Total Variance ##
    def tss(self):
        return self.rss() + self.mss()

    
    ## Reisudals ##
    def res(self):
        return self.y - self.pred
    
    
    ## Coefficient of determination ##
    def rsq(self):
        return round(float(self.mss()/self.tss()),4)
    
    
    ## Coefficient of determination adjusted ##
    def rsq_adj(self):
        k = len(self.w) - 1
        n = len(self.y)
        return (1 - (((1 - self.rsq()) * (n - 1)) / (n - k - 1)))

    
    ## F test on the entire model ##
    def Ftest(self, out = False):
        k = len(self.w) - 1
        n = len(self.y)
        F = (self.mss()/k) / (self.rss()/(n-k-1))
        print( f"F value: {round(F,4)}" )
        return f.sf( F, k, n-k-1 )
    
   
    
    ## Tests on each parameter ##
    def Ctest(self):
        k = len(self.w)
        n = len(self.y)
        d = np.std(self.res())
        XX = np.linalg.inv(np.dot(self.x.T, self.x))

        
        tmp = [0 for i in range(k)]
        for i in range(k):
            se = np.sqrt((self.rss() * XX[i,i])/n)
            tmp[i] = [self.columns[i], round(self.w[i],4), round(se, 4),
             round(self.w[i]/se,4), round(t.sf(abs(self.w[i]/se), n-k-1, 0, 1)*2, 4)] 
        return pd.DataFrame(tmp, columns = [" ", "Estimated value", "Std Error", "t-value", "p-value"])
  

        
################ classical linear model hypothesis #########################################

    def vif(self):
        X = pd.DataFrame(self.x[:,1:], columns = self.columns[1:])
        vif = [0 for _ in range(len(self.w)-1)]
        j = 0
        for i in X.columns:
            XX = X.drop(i, axis = 1) 
            Y = X[i]
            tmp = linear(XX, Y)
            tmp.normal()
            vif[j] = [tmp.rsq(), (1 - tmp.rsq()), round(1/(1 - tmp.rsq()), 4)]
            j += 1

        return pd.DataFrame(vif, columns = ["R2", "TOL", "VIF"], index = self.columns[1:])


    
#############################################################################################
############## ALPHA ########################################################################
#############################################################################################        


    ## Summary ##
    def summary(self):

        if(self.state == 0):
            raise Exception("You must fit the model first. Try 'gradient' or 'normal' method.")

        print("_______________________________________________________________")
        print("============================SUMMARY============================")
        print("¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯")
        print(f"RSS: {round(self.rss(),4)}, MSS: {round(self.mss(),4)}, TSS: {round(self.tss(),4)} ")
        print( f"Coefficient of determination: {self.rsq()}" )
        print( f"Coefficient of determination adjusted: {round(self.rsq_adj(),4)}" )
        print( f"p-value (H0: R = 0): {round(self.Ftest(),6)}" )
        print("---------------------------------------------------------------")
        print(self.Ctest().to_string(index=False))
        print("---------------------------------------------------------------")
        print(f"Multicollinearity:\n{self.vif().to_string()}")
        print("_______________________________________________________________")