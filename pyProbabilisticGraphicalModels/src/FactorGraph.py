'''
Created on 25 Jul 2020

@author: Francisco Dominguez
from: https://nbviewer.jupyter.org/github/krashkov/Belief-Propagation/blob/master/4-ImplementationBP.ipynb
      https://nbviewer.jupyter.org/github/krashkov/Belief-Propagation/blob/master/3-ImplementationPGM.ipynb
'''
import numpy as np
from NodeFactor import NodeFactor
from NodeVariable import NodeVariable
from Factor import Factor

class FactorGraph(object):
    def __init__(self):
        self.nodeFactors={}
        self.nodeVariables={}
    def addNodeFactor(self,nodef):
        self.nodeFactors[nodef.getName()]=nodef
        for i,namev in enumerate(nodef.getFactor().getVariables()):
            shape=nodef.getFactor().getShape()
            rank=shape[i]
            if not namev in self.nodeVariables:
                nodev=NodeVariable(namev,range(rank))
                self.nodeVariables[namev]=nodev
            else:
                nodev=self.nodeVariables[namev]
            nodev.addNodeFactor  (nodef)
            nodef.addVariable(nodev)
            self.nodeVariables[namev]=nodev
    def reduce(self,factor,vNames):
        for k in vNames:
            nv=self.nodeVariables[k]
            if nv.isEvidence():
                value=nv.getEvidenceValue()
                factor=factor.reduce(k,value)
        return factor
    def cleanMessages(self):
        for k in self.nodeFactors:
            self.nodeFactors[k].cleanMessageVariables()
        for k in self.nodeVariables:
            self.nodeVariables[k].cleanMessageFactors()
    def getNodeVariable(self,vName):
        return self.nodeVariables[vName]
    def getNodeFactor(self,fName):
        return self.nodeFactors[fName]
    #BELIEF
    def belief(self,vName):
        self.cleanMessages()
        incoming_messages=[]
        for fName in self.nodeVariables[vName].getNodeFactorNames():
            incoming_messages.append(self.msgFactorVariable(fName,vName))
        joined     =Factor.joint(incoming_messages)
        #normalized =joined.normalize()
        return joined         
    def msgVariableFactor(self,vName,fName):
        if not self.nodeVariables[vName].isInMessageFactors(fName):
            msg=self.buildVariableFactor(vName,fName)
            self.nodeVariables[vName].setMessageFactor(fName,msg)
        return msg
    def buildVariableFactor(self,vName,fName):
        incoming_messages=[]
        for fNameNeighbor in self.nodeVariables[vName].getNodeFactorNames():
            if fNameNeighbor!=fName:
                msg=self.msgFactorVariable(fNameNeighbor,vName)
                incoming_messages.append(msg)
        if not incoming_messages:
            incoming_messages.append(Factor([vName],np.array([1.]*self.nodeVariables[vName].getRank())))
        joined     =Factor.joint(incoming_messages)
        #normalized =joined.normalize()
        return joined         
    def msgFactorVariable(self,fName,vName):
        if not self.nodeFactors[fName].isInMessageVariables(vName):
            msg=self.buildFactorVariable(fName,vName)
            self.nodeFactors[fName].setMessageVariable(vName,msg)
        return msg
    def buildFactorVariable(self,fName,vName):
        incoming_messages=[self.nodeFactors[fName].getFactor()]   
        marginalization_variables=[]        
        for vNameMeighbor in self.nodeFactors[fName].getVariableNames():
            if vNameMeighbor!=vName:
                msg=self.msgVariableFactor(vNameMeighbor,fName)
                incoming_messages.append(msg)
                marginalization_variables.append(vNameMeighbor)
        joined      =Factor.joint(incoming_messages)
        #This doesn't seem to work for evicences :-(
        reduced     =self.reduce(joined,marginalization_variables+[vName])
        marginalized=reduced.marginalize(marginalization_variables)
        #normalized  =marginalized.normalize()
        return marginalized
    
    #MAP I seem to work but TODO backtracking
    def map(self,vName):
        self.cleanMessages()
        incoming_messages=[]
        for fName in self.nodeVariables[vName].getNodeFactorNames():
            incoming_messages.append(self.msgMaxFactorVariable(fName,vName))
        joined     =Factor.joint(incoming_messages)
        #normalized =joined.normalize()
        return joined         
    def msgMaxVariableFactor(self,vName,fName):
        if not self.nodeVariables[vName].isInMessageFactors(fName):
            msg=self.buildMaxVariableFactor(vName,fName)
            self.nodeVariables[vName].setMessageFactor(fName,msg)
        return msg
    def buildMaxVariableFactor(self,vName,fName):
        incoming_messages=[]
        for fNameNeighbor in self.nodeVariables[vName].getNodeFactorNames():
            if fNameNeighbor!=fName:
                msg=self.msgMaxFactorVariable(fNameNeighbor,vName)
                incoming_messages.append(msg)
        if not incoming_messages:
            incoming_messages.append(Factor([vName],np.array([1.]*self.nodeVariables[vName].getRank())))
        joined     =Factor.joint(incoming_messages)
        #normalized =joined.normalize()
        return joined         
    def msgMaxFactorVariable(self,fName,vName):
        if not self.nodeFactors[fName].isInMessageVariables(vName):
            msg=self.buildMaxFactorVariable(fName,vName)
            self.nodeFactors[fName].setMessageVariable(vName,msg)
        return msg
    def buildMaxFactorVariable(self,fName,vName):
        incoming_messages=[self.nodeFactors[fName].getFactor()]   
        marginalization_variables=[]        
        for vNameMeighbor in self.nodeFactors[fName].getVariableNames():
            if vNameMeighbor!=vName:
                msg=self.msgMaxVariableFactor(vNameMeighbor,fName)
                incoming_messages.append(msg)
                marginalization_variables.append(vNameMeighbor)
        joined      =Factor.joint(incoming_messages)
        maximized=joined.maximize(marginalization_variables)
        #normalized  =maximized.normalize()
        return maximized
    #BUILD new factor graph from evidences
    def getNodeVariablesEvicences(self):
        nve=[]
        for k in self.nodeVariables:
            nv=self.nodeVariables[k]
            if nv.isEvidence():
                nve.append(nv)
        return nve
    def reduceEvidences(self,factor):
        for nv in self.getNodeVariablesEvicences():
            if nv.getName() in factor.getVariables():
                factor=factor.reduce(nv.getName(),nv.getEvidenceValue())
        return factor
    def factorGraphEvidences(self):
        fg=FactorGraph()
        for k in self.nodeFactors:
            nf=self.nodeFactors[k]
            factor=nf.getFactor()
            factor=self.reduceEvidences(factor)
            fg.addNodeFactor(NodeFactor(nf.getName(),factor))
        return fg
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          