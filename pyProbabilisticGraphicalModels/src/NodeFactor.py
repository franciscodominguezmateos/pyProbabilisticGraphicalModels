'''
Created on 25 Jul 2020

@author: Francisco Dominguez
From: https://github.com/krashkov/Belief-Propagation
      https://nbviewer.jupyter.org/github/krashkov/Belief-Propagation/blob/master/2-ImplementationFactor.ipynb
'''
from Factor import Factor
class NodeFactor(object):
    def __init__(self,name,factor):
        self.name=name
        self.factor=factor
        self.nodeVariables={}
        self.messageVariables={}
    def setFactor(self, factor):
        # Check ranks
        #self.__check_variable_ranks(f_name, factor_, 0)
        # Set ranks
        #self.__set_variable_ranks(f_name, factor_)
        # Set data
        #self._graph.vs.find(name=f_name)['factor_'] = factor_        
        self.f=factor
    def getName(self): return self.name
    def getFactor(self): return self.factor
    def addVariable(self,nodev):
        self.nodeVariables[nodev.getName()]=nodev
    def getVariableNames(self): return self.nodeVariables.keys()
    def cleanMessageVariables(self): self.messageVariables={}
    def getMessageVariables (self): return self.messageVariables
    def isInMessageVariables(self,fName): return fName in self.messageVariables.keys()
    def setMessageVariable  (self,fName,msg): self.messageVariables[fName]=msg