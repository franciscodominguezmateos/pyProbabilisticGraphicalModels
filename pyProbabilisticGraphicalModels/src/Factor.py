'''
Created on 25 Jul 2020

@author: Francisco Dominguez
From: https://github.com/krashkov/Belief-Propagation
      https://nbviewer.jupyter.org/github/krashkov/Belief-Propagation/blob/master/2-ImplementationFactor.ipynb
'''
import numpy as np
class Factor(object):
    def __init__(self, variables = None, distribution = None):
        if (variables is None) or (len(variables) != len(distribution.shape)):
            raise Exception('Data is incorrect')
        else:
            self.__set_data(np.array(variables),
                            np.array(distribution),
                            np.array(distribution.shape))
    
    def __set_data(self, variables, distribution, shape):
        self.__variables    = variables
        self.__distribution = distribution
        self.__distMaxIdxs  = None
        self.__shape        = shape
    
    # ----------------------- Info --------------------------
    def is_none(self):
        return True if self.__distribution is None else False
        
    # ----------------------- Getters -----------------------
    def getVariables(self):
        return self.__variables
    
    def getDistribution(self):
        return self.__distribution
    
    def getShape(self):
        return self.__shape
    def setDistributionMaxIndices(self,dmi): self.__distMaxIdxs=dmi
    def getDistributionMaxIndices(self): return self.__distMaxIdxs
    @staticmethod
    def joint(factors):
        res=factors[0]
        for f in factors[1:]:
            res=res*f
        return res        
    def __mul__(self, y):
        x=self
        if x.is_none() or y.is_none():
            raise Exception('One of the factors is None')
        
        xy, xy_in_x_ind, xy_in_y_ind = np.intersect1d(x.getVariables(), y.getVariables(), return_indices=True)
        
        if xy.size == 0:
            raise Exception('Factors do not have common variables')
        
        xyx=x.getShape()[xy_in_x_ind]
        xyy=y.getShape()[xy_in_y_ind]
        if not np.all( xyx==xyy ):
            raise Exception('Common variables have different order')
        
        x_not_in_y = np.setdiff1d(x.getVariables(), y.getVariables(), assume_unique=True)
        y_not_in_x = np.setdiff1d(y.getVariables(), x.getVariables(), assume_unique=True)
        
        x_mask = np.isin(x.getVariables(), xy, invert=True)
        y_mask = np.isin(y.getVariables(), xy, invert=True)
        
        x_ind = np.array([-1]*len(x.getVariables()), dtype=int)
        y_ind = np.array([-1]*len(y.getVariables()), dtype=int)
        
        x_ind[x_mask] = np.arange(np.sum(x_mask))
        y_ind[y_mask] = np.arange(np.sum(y_mask)) + np.sum(np.invert(y_mask))
        
        x_ind[xy_in_x_ind] = np.arange(len(xy)) + np.sum(x_mask)
        y_ind[xy_in_y_ind] = np.arange(len(xy))
        
        x_distribution = np.moveaxis(x.getDistribution(), range(len(x_ind)), x_ind)
        y_distribution = np.moveaxis(y.getDistribution(), range(len(y_ind)), y_ind)
        
        idxx=tuple([slice(None)]*len(x.getVariables())+[None]*len(y_not_in_x))
        idxy=tuple([None]*len(x_not_in_y)+[slice(None)])
        #for numerical stability
        #better exp of sum of logs than just product
        with np.errstate(divide='ignore'):#avoid division vy zero error with log(0)
            logx=np.log(x_distribution[idxx])
            logy=np.log(y_distribution[idxy])
        res_distribution = np.exp(logx+logy)
        #res_distribution =   x_distribution[idxx] * y_distribution[idxy]
        
        return Factor(list(x_not_in_y)+list(xy)+list(y_not_in_x), res_distribution)
    def marginalize(self, variables):
        x=self
        variables = np.array(variables)
        
        if x.is_none():
            raise Exception('Factor is None')
        
        if not np.all(np.in1d(variables, x.getVariables())):
            raise Exception('Factor do not contain given variables')
        
        res_variables    = np.setdiff1d(x.getVariables(), variables, assume_unique=True)
        res_distribution = np.sum(x.getDistribution(),
                                  tuple(np.where(np.isin(x.getVariables(), variables))[0]))
        
        factor=Factor(res_variables, res_distribution)
        return factor
    def maximize(self, variables):
        x=self
        variables = np.array(variables)
        
        if x.is_none():
            raise Exception('Factor is None')
        
        if not np.all(np.in1d(variables, x.getVariables())):
            raise Exception('Factor do not contain given variables')
        
        res_variables    = np.setdiff1d(x.getVariables(), variables, assume_unique=True)
        res_distribution = np.max(x.getDistribution(),
                                  tuple(np.where(np.isin(x.getVariables(), variables))[0]))
        #for some reason np.argmax doesn't work the same that np.max axis can't be a tuple
        #res_distMaxIdxs  = np.argmax(x.getDistribution(),
        #                          tuple(np.where(np.isin(x.getVariables(), variables))[0]))
        res_distMaxIdxs  = np.argmax(res_distribution)
        factor=Factor(res_variables, res_distribution)
        factor.setDistributionMaxIndices(res_distMaxIdxs)
        return factor
    def reduce(self, variable, value):
        x=self
        if x.is_none() or (variable is None) or (value is None):
            raise Exception('Input is None')
        
        if not np.any(variable == x.getVariables()):
            raise Exception('Factor do not contain given variable')
        
        if value >= x.getShape()[np.where(variable==x.getVariables())[0]]:
            raise Exception('Incorrect value of given variable')
        
        res_variables    = np.setdiff1d(x.getVariables(), variable, assume_unique=True)
        res_distribution = np.take(x.getDistribution(),
                                   value,
                                   int(np.where(variable==x.getVariables())[0]))
        
        factor=Factor(res_variables, res_distribution)
        return factor
    def normalize(self):
        dist=self.getDistribution()
        factor=Factor(self.getVariables(),dist/dist.sum())
        return factor

        