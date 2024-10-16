"""
    Created on Mon July 24 2018
    Edited on Thu Feb 16 2023
    @author: Christina Christodoulakis (cchristodoulaki@gmail.com), based on code by Sina Ahmadi  (sina.ahmadi@insight-centre.org)
    Based on the code for IJCAI 2017 paper titled "Diverse Weighted Bipartite b-Matching"
    by Ahmed, Faez, John P. Dickerson, and Mark Fuge
    https://github.com/faezahmed/diverse_matching/
"""
from gurobipy import *
import numpy as np
import pandas as pd

class WBbM:
    """Weighted Bipartite b-Matching (WBbM) algorithm"""
    def __init__(self, num_left, num_right, W, lda, uda, ldp, udp, LogToConsole=0):
        
        self.num_left = num_left #number of Papers
        self.num_right = num_right #number of Authors
        
        self.W = W
        
        self.lda = lda #lower capacity of authors (minimum papers to review)
        self.uda = uda #upper capacity of authors (maximum papers to review)
        
        self.ldp = ldp #lower capacity of papers (minimum authors to be reviewed by)
        self.udp = udp #upper capacity of papers (maximum authors to be reviewed by)
        
        self.LogToConsole = LogToConsole

    def linkmatr(self, num_left, num_right):
        """ Creates link matrix A for constraint satisfaction """
        num_nodes = self.num_left + self.num_right
        str1 = [1] * self.num_right
        str2 = [0] * self.num_right
        A = [None] * (num_nodes)
        
        # The first num_left rows correspond to PAPERS 
        for i in range(self.num_left):
            A[i] = str2 * self.num_left
            idx = self.num_right * i
            A[i][idx:idx + self.num_right] = str1
            
        # The next num_right rows correspond to REVIEWERS (authors)    
        for j in range(self.num_right):
            A[self.num_left+j] = str2 * self.num_left
            idx = [j + self.num_right * l for l in range(self.num_left)]
            for k in range(self.num_left):
                A[self.num_left + j][idx[k]] = 1 
        return A
    
    def Bb_matching(self, optimization_mode="max"):
        """ Solves the matching problem """
        m = Model("WBbM_matching")
        m.setParam("LogToConsole", self.LogToConsole)

        total_nodes = self.num_left + self.num_right
        total_vars = self.num_left * self.num_right
                
        # print(f"num_left * np.array(lda) = {self.num_left * np.array(self.lda)}")
        # print(f"num_right * np.array(udp) = {self.num_right * np.array(self.udp)}")
        # print(f"num_right * np.array(ldp) = {self.num_right * np.array(self.ldp)}")
        # print(f"num_left * np.array(uda) = {self.num_left * np.array(self.uda)}") 
        
        # if(np.any(self.num_left * np.array(self.lda) > self.num_right * np.array(self.udp)) or np.any(self.num_right * np.array(self.ldp) > self.num_left * np.array(self.uda)) ):
        #     raise Exception("Infeasible Problem")
        
        
        # Maximum Number of authors matched to node paper
        if type(self.udp).__name__ == "int":
            Dmax = list(self.udp * np.ones((self.num_left,)))+list(0 * np.ones((self.num_right,)))
        elif type(self.udp).__name__ == "list":
            Dmax = self.udp + list(0 * np.ones((self.num_right,)))
        else:
            raise Exception("udp value not correct.")
        # print(f"\nudp becomes Dmax={Dmax}" )
        
        # Minimum Number of authors matched to a paper
        if type(self.ldp).__name__ == "int":
            Dmin = list(self.ldp * np.ones((self.num_left,)))+list(0 * np.ones((self.num_right,)))
        elif type(self.ldp).__name__ == "list":
            Dmin = self.ldp + list(0 * np.ones((self.num_right,)))
        else:
            raise Exception("udp value not correct.")
                
        # print(f"ldp becomes Dmin={Dmin}")
        
        # Minimum Number of papers matched to an author
        if type(self.lda).__name__ == "int":
            Dmina = list(0 * np.ones((self.num_left,)))+list(self.lda * np.ones((self.num_left,)))
        elif type(self.lda).__name__ == "list":
            Dmina = list(0 * np.ones((self.num_left,))) + self.lda 
        else:
            raise Exception("uda value not correct.")
        # print(f"lda becomes Dmina={Dmina}") 
        
        # Maximum number of papers matched to author
        if type(self.uda).__name__ == "int":
            Dmaxa = list(0 * np.ones((self.num_left,)))+list(self.uda * np.ones((self.num_left,)))
        elif type(self.uda).__name__ == "list":
            Dmaxa = list(0 * np.ones((self.num_left,))) + self.uda 
        else:
            raise Exception("uda value not correct.")
        # print(f"uda becomes Dmaxa={Dmaxa}") 
        
        A = self.linkmatr(self.num_left, self.num_right)
        
        x = dict()
        for j in range(total_vars):
          x[j] = m.addVar(vtype=GRB.BINARY, name="x" + str(j))
        
        # objective
        if optimization_mode=="max":
            m.setObjective((quicksum( self.W[i]*x[i] for i in range(total_vars) if ( not math.isinf(self.W[i]) and not np.isnan(self.W[i]) ) )), GRB.MAXIMIZE)
        elif optimization_mode=="min":
            m.setObjective((quicksum(self.W[i]*x[i] for i in range(total_vars) if ( not math.isinf(self.W[i]) and not np.isnan(self.W[i]) ))), GRB.MINIMIZE)
        else:
            raise ValueError("Optimization mode not recognized.")
        
        # constraint on paper capacity
        for i in range(self.num_left):
            m.addConstr(quicksum(A[i][j] * x[j] for j in range(total_vars)) <= Dmax[i])
            m.addConstr(quicksum(A[i][j] * x[j] for j in range(total_vars)) >= Dmin[i])
                
        # constraint on author  capacity
        for i in range(self.num_left, total_nodes):
            m.addConstr(quicksum(A[i][j]*x[j] for j in range(total_vars)) <= Dmaxa[i])
            m.addConstr(quicksum(A[i][j]*x[j] for j in range(total_vars)) >= Dmina[i]) 

        #m.write("lp.mps")    
        
        m.optimize()   
        
        res = np.zeros((self.num_left, self.num_right))
        for i in range(self.num_left):
            for j in range(self.num_right):
                idx = self.num_right*i + j
                res[i,j] = m.getVars()[idx].x
           
        status = m.status
        if status == GRB.Status.UNBOUNDED:
            print('The model cannot be solved because it is unbounded')
        # elif status == GRB.Status.OPTIMAL:
        #     print('The optimal objective is %g' % m.objVal)
        # elif status != GRB.Status.INF_OR_UNBD and status != GRB.Status.INFEASIBLE:
        #     print('Optimization was stopped with status %d' % status)
            
        return res, m.objVal