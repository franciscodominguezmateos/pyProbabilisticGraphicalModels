'''
Created on 25 Jul 2020

@author: Francisco Dominguez
From: https://github.com/krashkov/Belief-Propagation
      https://nbviewer.jupyter.org/github/krashkov/Belief-Propagation/blob/master/2-ImplementationFactor.ipynb
'''

class NodeVariable(object):
    def __init__(self, name, support=[False,True]):
        self.name=name
        self.support=support
        self.rank=len(self.support)
        self.evidenceValue=None
        self.nodeFactors={}
        self.messageFactors={}
    def getName(self): return self.name
    def getRank(self): return self.rank
    def getNodeFactors(self):     return self.nodeFactors.values()
    def getNodeFactorNames(self): return self.nodeFactors.keys()
    def addNodeFactor(self,factor):  self.nodeFactors[factor.getName()]=factor
    def cleanMessageFactors(self): self.messageFactors={}
    def getMessageFactors(self): return self.messageFactors
    def isInMessageFactors(self,fName): return fName in self.messageFactors.keys()
    def setMessageFactor(self,fName,msg): self.messageFactors[fName]=msg
    def isEvidence(self):       return self.evidenceValue!=None
    def getEvidenceValue(self): return self.evidenceValue
    def setEvidenceValue(self,e): 
        if e in self.support:
            self.evidenceValue=e
        else:
            raise Exception("Evidence value=%d no in support of variable %s."%(e,self.name))