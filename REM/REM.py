
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 12:42:14 2014

@author: malmberg



"""
import numpy as np
from encodelist import encodelist
from match import match
#from numpy import * as np
g = .4                    #geometric distribution parameter to determine base rate of feature values
ListLength = 10
w = 12                    #number of feature representing an item
N = 40                    #number of simulated subjects to run

HRdata = np.zeros(N)      #contains the HR data for each subject
FARdata = np.zeros(N)     #contains the FAR data for each subject

for subNum in range(N):   # run N simulated subjects
                          #    print "subject = ", subNum # for debugging

    studylist = np.random.geometric(g, size=ListLength*w).reshape(ListLength,w)

        # studylist is an array with ListLength rows. 
        # Each row is a lexical/semantic trace.
        # Each lexical/semantic trace has w features. 
        # The features sampled randomly from a geometric distribution defined by the REM parameter g.

    memorylist = studylist 
    
        #to begin the memory list a perfect copy of the studylist

    c = .7      #prob of storing a feature correctly
    u = .1      #prob of storing a feature in each attempt
    t = 12      #number of attempts to store a feature
    memorylist = encodelist(w, ListLength, studylist, g, u, t, c)
    
    #encode an incomplete and error-prone copy of study items into memorylist
    foillist = np.random.geometric(g, size=ListLength*w).reshape(ListLength,w)
    
    #this is the set of retrieval cues used  on foils trials
    targetlist = studylist
    
    #this is the set of retrieval cues used  on target trials
    odds = np.zeros(ListLength)
    
    #this is the container for the odds values associated with each test
    for cueNum in range(len(targetlist)):   #for each target match it against the contexts of memory and obtain the likelihood ratio (lambda, j) for each trace
        likes = np.zeros(len(targetlist)*len(memorylist)).reshape(len(targetlist),len(memorylist))
        
        for traceNum in range(len(memorylist)): 
            likes[cueNum,traceNum]=match(targetlist[cueNum],memorylist[traceNum],g,c)
            
        odds[cueNum] = np.sum(likes[cueNum])/len(memorylist)
        
    HRdata[subNum]=float((odds > 1.0).sum())/float(len(targetlist))

    odds = np.zeros(ListLength)
    
    
    for cueNum in range(len(foillist)): #for each foil match it against the contexts of memory and obtain the likelihood ratio (lambda, j) for each trace
        likes = np.zeros(len(targetlist)*len(memorylist)).reshape(len(targetlist),len(memorylist))
        
        for traceNum in range(len(memorylist)): 
            likes[cueNum,traceNum]=match(foillist[cueNum],memorylist[traceNum],g,c)
            
        odds[cueNum] = np.sum(likes[cueNum])/len(memorylist)
        
    FARdata[subNum]=float((odds > 1.0).sum())/float(len(targetlist))
    
    
    
print ("Avg HR = ", float(HRdata.sum())/float(N))
print ("Avg FAR = ", float(FARdata.sum())/float(N))
        






