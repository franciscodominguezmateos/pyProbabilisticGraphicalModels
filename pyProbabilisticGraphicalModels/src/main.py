'''
Created on 25 Jul 2020

@author: Francisco Dominguez
'''
import random
import numpy as np
from FactorGraph import FactorGraph
from NodeVariable import NodeVariable
from NodeFactor import NodeFactor
from Factor import Factor

if __name__ == '__main__':
    f1=Factor(["a","b"],np.array([[0.5, 0.8],[0.1,0.0],[0.3,0.9]]))
    f2=Factor(['b','c'],np.array([[0.5, 0.7],[0.1,0.2]]))
    f3=f1*f2
    print(f3.getVariables())
    print(f3.getDistribution())
    f4=Factor(['a','b'],np.array([[0.3,0.8],[0.2,0.1],[0.5,0.1]]))
    f5=Factor(['b'],np.array([0.3,0.7]))
    f6=f4*f5
    print(f6.getVariables())
    print(f6.getDistribution())
    f7=f3.marginalize(['b'])
    print(f7.getVariables())
    print(f7.getDistribution())
    f8=f3.reduce('c', 0)
    print(f8.getVariables())
    print(f8.getDistribution())
    
    #fg=FactorGraph()
    #fg.addNodeFactor(NodeFactor('p1',Factor(['x1','x2','x3'])))
    #fg.addNodeFactor(NodeFactor('p2',Factor(['x2','x4'])))
    
    mrf=FactorGraph()
    f1 = Factor(['a', 'b'],      np.array([[2,3],[6,4]]))
    f2 = Factor(['b', 'd', 'c'], np.array([[[7,2,3],[1,5,2]],[[8,3,9],[6,4,2]]]))
    f3 = Factor(['c'],           np.array([5, 1, 9]))
    mrf.addNodeFactor(NodeFactor("f1",f1))
    mrf.addNodeFactor(NodeFactor("f2",f2))
    mrf.addNodeFactor(NodeFactor("f3",f3))
#     fb=mrf.belief('b')
#     p=fb.getDistribution()
#     print(p)
#     f1b=f1*f2
#     print(f1b.getVariables())
#     print(f1b.getDistribution())
#     fb=mrf.map('b')
#     p=fb.getDistribution()
#     print(p)
#     print(fb.getDistributionMaxIndices())

    #from Barber's Bayesian Reasoning And Machine Learning
    #Example: 5.2
    pab=Factor(['a','b'],np.array([[0.8,0.7 ],[0.2,0.3 ]]))
    pbc=Factor(['b','c'],np.array([[0.9,0.25],[0.1,0.75]]))
    pc =Factor(['c'],    np.array([0.6,0.4]))
    mfg=FactorGraph()
    mfg.addNodeFactor(NodeFactor('pab',pab))
    mfg.addNodeFactor(NodeFactor('pbc',pbc))
    mfg.addNodeFactor(NodeFactor('pc' ,pc))
    ma=mfg.map('a')
    p=ma.getDistribution()
    #true values are 0.432, 0.108. OK
    print(p)
    print(ma.getDistributionMaxIndices())
    
    #from Barber's Bayesian Reasoning And Machine Learning
    #FLY in three rooms
    transition=np.array([[0.7,0.499,0.001],[0.3,0.3,0.5],[0.001,0.199,0.5]])
    p54=Factor(["x5","x4"],transition)
    p43=Factor(["x4","x3"],transition)
    p32=Factor(["x3","x2"],transition)
    p21=Factor(["x2","x1"],transition)
    p1 =Factor(["x1"]     ,np.array([0.998,0.001,0.001]))
    fgFly=FactorGraph()
    fgFly.addNodeFactor(NodeFactor("p54",p54))
    fgFly.addNodeFactor(NodeFactor("p43",p43))
    fgFly.addNodeFactor(NodeFactor("p32",p32))
    fgFly.addNodeFactor(NodeFactor("p21",p21))
    fgFly.addNodeFactor(NodeFactor("p1" ,p1))
    ff=fgFly.belief("x5")
    fp=ff.getDistribution()
    #true values are 0.574,0.318,0.1.08. OKª
    print(fp)
    
    #http://probmods.org/chapters/dependence.html
    psmokes    =Factor(["smokes"]       ,np.array([0.8,0.2]))
    plungSmokes=Factor(["lung","smokes"],np.array([[0.999,0.8991],[0.001,0.1009]]))
    pcold      =Factor(["cold"]         ,np.array([0.98,0.02]))
    pbreathLung=Factor(["breath","lung"],np.array([[0.99,0.792],[0.01,0.208]]))
    pchestLung =Factor(["chest","lung"] ,np.array([[0.99,0.792],[0.01,0.208]]))
    pcoughLungCold=Factor(["cough","lung","cold"],np.array([[[0.99 ,0.495 ],
                                                             [0.495,0.2475]],
                                                            [[0.01 ,0.505 ],
                                                             [0.505,0.7525]]
                                                             ]))
    pfeverCold=Factor(["fever","cold"],np.array([[0.99,0.693],[0.01,0.307]]))
    fgDiag=FactorGraph()
    fgDiag.addNodeFactor(NodeFactor("psmokes",psmokes))
    fgDiag.addNodeFactor(NodeFactor("plungSmokes",plungSmokes))
    fgDiag.addNodeFactor(NodeFactor("pcold",pcold))
    fgDiag.addNodeFactor(NodeFactor("pbreathLung",pbreathLung))
    fgDiag.addNodeFactor(NodeFactor("pchestLung",pchestLung))
    fgDiag.addNodeFactor(NodeFactor("pcoughLungCold",pcoughLungCold))
    fgDiag.addNodeFactor(NodeFactor("pfeverCold",pfeverCold))
    fcold=fgDiag.belief("cough")
    print(fcold.getDistribution())
    fgDiag.nodeVariables["smokes"].setEvidenceValue(1)
    fgde=fgDiag.factorGraphEvidences()
    fcold=fgde.belief("cough")
    print(fcold.getDistribution())
    fcold=fgde.belief("cold")
    print(fcold.getDistribution())
    
    '''
    Variable elimination
    '''
    def chooseRandom(responses):
        sizeResponses=len(responses)
        chooseIdResponse=random.randint(0,sizeResponses-1)
        return responses[chooseIdResponse]
    def getWeight(factor):
        return len(factor.getVariables())
    #just pick the factor with less variables
    def chooseVariable(factors):
        iFact=0
        wFact=10e6
        for i,f in enumerate(factors):
            w=getWeight(f)
            if w<wFact:
                wFact=w
                iFact=i
        return chooseRandom(factors[iFact].getVariables())
    def factorsVariable(factors,variable):
        fv=[]
        for i,f in enumerate(factors):
            if variable in f.getVariables():
                fv.append(f)
        return fv
            
    variables=list(fgDiag.nodeVariables.keys())
    variables.remove("cold")
    variables.remove("smokes")
    factors=[psmokes,plungSmokes,pcold,pbreathLung,pchestLung,pcoughLungCold,pfeverCold]
    while variables: #len(factors)>1:
        print("variables=",variables)
        #variable=chooseVariable(factors)
        variable=chooseRandom(variables)
        print("variable=",variable)
        fv=factorsVariable(factors,variable)
        print("fv=",len(fv))
        fj=Factor.joint(fv)
        print("fj =",fj.getVariables())
        fj=fj.marginalize(variable)
        print("fjm=",fj.getVariables())
        variables.remove(variable)
        for f in fv: factors.remove(f)
        factors.append(fj)
        print("factors=",len(factors))
        for f in factors:
            print(f.getVariables())
    jointAll=Factor.joint(factors)
    p=jointAll.getDistribution()
    print("p=",p)
    print(jointAll.getVariables())
    
    
    
    
    
    
    
    
    