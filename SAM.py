#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 16:32:58 2020

@author: willamannering
"""
import numpy as np

import random


class SAM:
    
    def __init__(self, ListLength, t=12, r=4, sam_a = .1, sam_b = .1, sam_c = .1, sam_d = .02, sam_e = .7, sam_f = .7, sam_g = .7, Kmax = 30, Lmax = 3):
        
        ''' ListLength = number of items in studylist, 
        t = presentation time per word
        r = short term memory buffer
        sam_a = multiplier for context to word association
        sam_b = mulitplier for word cue to other word trace association
        sam_c = multiplier for word cue to same word trace association
        sam_d = residual strength of association for words that never appear in buffer together
        sam_e = incrementing parameter for context to word association
        sam_f = incrementing parameter for word to other word association
        sam_g = incrementing parameter for word to itself association
        Kmax = maximum number of retrieval failures before search process is stopped
        Lmax = max number of retrieval attempts using word cues instead of context
        '''
        


        self.ListLength = ListLength
        self.t = t
        self.r = r
        self.sam_a = sam_a
        self.sam_b = sam_b
        self.sam_c = sam_c
        self.sam_d = sam_d
        self.sam_e = sam_e
        self.sam_f = sam_f
        self.sam_g = sam_g
        self.Kmax = Kmax
        self.Lmax = Lmax
        
    #method for encoding items
    def encodeitems(self):
        import itertools
        
        buffer = []
        present_order = list(range(self.ListLength))
        
        random.shuffle(present_order) #randomize presentation order
        
        #initialize context association and word-word association matrices
        context_assoc = np.zeros((1, self.ListLength))
        word_assoc = np.zeros((self.ListLength, self.ListLength))
        
        loops_inbuffer = [0]*self.ListLength #what loops were all items in the buffer?
        
        for i in range(self.ListLength):
            if i >= self.r: #if buffer is full
                buffer[random.randint(0, self.r-1)] = present_order[i] #randomly replace item in buffer with next item
                
            else: #if buffer isn't full
                buffer.append(present_order[i]) #add next item to buffer
                
            
            to_update = list(itertools.permutations(buffer,2))
            
            for i in range(self.t): #presentation time, if t is larger words stay in buffer updating associations for longer
                for pair in to_update:
                    word_assoc[pair[0]][pair[1]] = word_assoc[pair[0]][pair[1]]+ (self.sam_b)
                for j in range(len(buffer)):
                    word_assoc[buffer[j]][buffer[j]] = word_assoc[buffer[j]][buffer[j]] + (self.sam_c)
                for j in buffer:
                    loops_inbuffer[j] += 1
            
            word_assoc[word_assoc == 0] = self.sam_d
            
            
        #create context association matrix     
        context_assoc = np.array([self.sam_a*loops for loops in loops_inbuffer])
       
    
        return context_assoc, word_assoc
    
    #method for updating association matrix
    def update_assoc(self, context_assoc, word_assoc, sampledTrace,  wordcue = -1):
    
        if wordcue != -1: #if word was used as cue, update strengths between cue and retrieved image
            
            new_cont = context_assoc
            new_word = word_assoc
            new_cont[sampledTrace] = context_assoc[sampledTrace] + self.sam_e
            new_word[sampledTrace][sampledTrace] = word_assoc[sampledTrace][sampledTrace] + self.sam_g
            new_word[wordcue][sampledTrace] = new_word[wordcue][sampledTrace] + self.sam_f 
            return new_cont, new_word
        
        else: #if only context was used, update association between image to context and image to itself
            new_cont = context_assoc
            new_word = word_assoc
            new_cont[sampledTrace] = context_assoc[sampledTrace] + self.sam_e
            new_word[sampledTrace][sampledTrace] = word_assoc[sampledTrace][sampledTrace] + self.sam_g
        
            return new_cont, new_word
        
        
    #free recall process 
    def free_recall(self):
        ''' ListLength = number of items in studylist, 
            w = number of features per word (dimensions)
            g = geometric distribution parameter to determine base rate of feature values
            t = presentation time per word
            r = short term memory buffer
            sam_a = multiplier for context to word association
            sam_b = mulitplier for word cue to other word trace association
            sam_c = multiplier for word cue to same word trace association
            sam_d = residual strength of association for words that never appear in buffer together
            sam_e = incrementing parameter for context to word association
            sam_f = incrementing parameter for word to other word association
            sam_g = incrementing parameter for word to itself association
            Kmax = maximum number of retrieval failures before search process is stopped
            Lmax = max number of retrieval attempts using word cues instead of context
    
            '''
        
      
        #create and encode studylists
        context_assoc, word_assoc = self.encodeitems() #encode studied item list in memory
        
        
        #begin free recall process
        alreadySaid = [False] * self.ListLength #at this point no words have been said from the list
        
        
        recalls_incues = 0
        cuelist = np.random.choice(self.ListLength, size = 15, replace = False)#randomly picked cues from study list to use as part-set cues
        
        
        response = [] #empty list of free recall responses
        K = 0 #number of retrieval failures, recall continues until retrival failures is greater than Kmax
        
          
        while(K < self.Kmax):
            L = 0
    
            probSamp = [ci/np.sum(context_assoc) for ci in list(context_assoc)]
    
            sampledTrace = np.random.choice(a = list(range(self.ListLength)), size = 1, p = probSamp)[0] #begin free recall by using context as a search cue
        
            if (alreadySaid[sampledTrace]):
                K += 1
                
                    
            else: #otherwise, if a new trace was sampled
                    
                probRecover = 1-np.exp(-context_assoc[sampledTrace])
                
               
                if (probRecover > np.random.rand(1)[0]):#if recovery is successful, update strength of probe to the recovered memory trace
                    
                    context_assoc, word_assoc = self.update_assoc(context_assoc, word_assoc, sampledTrace)
                    
                    alreadySaid[sampledTrace] = True #set this trace to already said
                    
                    response.append(sampledTrace) #record this trace as a successful response
                    
                    if sampledTrace in cuelist:
                        recalls_incues+=1
                        
                    #if trace was recovered, use this word as cue to recover more traces
                    while(L < self.Lmax):
                        
                        previous_sample = sampledTrace
                        
                        
                        probSamp = (context_assoc*word_assoc[sampledTrace])/sum(context_assoc*word_assoc[sampledTrace])
                        
                        #randomly choose a trace using calculated probabilities
                        sampledTrace = np.random.choice(a = list(range(self.ListLength)), size = 1, p = probSamp)[0]
                        
                        
                        if (alreadySaid[sampledTrace]): #if sampledTrace already said, count this as retrieval failure and start again
                            K += 1
                            L += 1                     
                            sampledTrace = previous_sample
    
                        else: #otherwise, if a new trace was sampled
                            
                            probRecover = 1-np.exp((-context_assoc[sampledTrace])-word_assoc[previous_sample][sampledTrace])
                            
                            
                            if (probRecover > np.random.rand(1)[0]):#if recovery is successful, update strength of probe to the recovered memory trace
                                context_assoc, word_assoc = self.update_assoc(context_assoc, word_assoc, sampledTrace, wordcue = previous_sample)
                                alreadySaid[sampledTrace] = True #set this trace to already said
                                
                                response.append(sampledTrace) #record this trace as a successful response
                                
                                
                                if sampledTrace in cuelist:
                                    recalls_incues+=1
                                    
                                L = 0 #set L to zero and use newly recovered trace as cue
        
                            else:
                                L+=1
                                
                                sampledTrace = previous_sample
                                
                else:
                    K += 1
    
        #begin rechecking phase
        new_res = []
        for res in response:
            L = 0
            
            while(L < self.Lmax):
                        
                previous_sample = res
                
                
                probSamp = (context_assoc*word_assoc[res])/sum(context_assoc*word_assoc[res])
    
                #randomly choose a trace using calculated probabilities
                sampledTrace = np.random.choice(a = list(range(self.ListLength)), size = 1, p = probSamp)[0]
                
                if (alreadySaid[sampledTrace]): #if sampledTrace already said, count this as retrieval failure and start again
                    K += 1
                    L += 1
                    res = previous_sample
                    
    
                else: #otherwise, if a new trace was sampled
                    
                    probRecover = 1-np.exp((-context_assoc[sampledTrace])-word_assoc[previous_sample][sampledTrace])
                   
                    
                    
                    if (probRecover > np.random.rand(1)[0]):#if recovery is successful, update strength of probe to the recovered memory trace
                        context_assoc, word_assoc = self.update_assoc(context_assoc, word_assoc, sampledTrace, wordcue = previous_sample)
                        alreadySaid[sampledTrace] = True #set this trace to already said
                        
                        new_res.append(sampledTrace) #record this trace as a successful response
                        
                        if sampledTrace in cuelist:
                            recalls_incues +=1
                        L = 0 #set L to zero and use newly recovered trace as cue
        
                    else:
                        L+=1
                        res = previous_sample
                        
        
        response.extend(new_res)
        
        
        #return response, recalls_incues
        return response
     
        
 
        