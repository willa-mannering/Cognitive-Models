#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 14:27:49 2020

@author: willamannering
"""

import numpy as np
import time
import random

#sam_e = .85, sam_f = .89, sam_g = .56, Kmax = 33, Lmax = 1
#sam_e = .94, sam_f = .80, sam_g = .99, Kmax = 39, Lmax = 2
class SAM_Group_Categorized:
    
    def __init__(self, ListLength, category_size, group_response = [], t=2, r=4, sam_a = .07, sam_b = .07, 
                 sam_c = .07, sam_d = .02, sam_e = .7, sam_f = .7, sam_g = .7, 
                 sam_h = .15, sam_i = .005, sam_j = .4, Kmax = 30, Lmax = 3):
        
        ''' ListLength = number of items in studylist,
        group_response = list to keep track of group responses
        category_size = number of words per category
        t = presentation time per word
        r = short term memory buffer
        sam_a = multiplier for context to word association
        sam_b = mulitplier for word cue to other word trace association
        sam_c = multiplier for word cue to same word trace association
        sam_d = residual strength of association for words that never appear in buffer together
        sam_e = incrementing parameter for context to word association
        sam_f = incrementing parameter for word to other word association
        sam_g = incrementing parameter for word to itself association
        sam_h = parameter controlling starting similarity between words in the same category
        sam_i = parameter controlling starting similarity between words in different categories
        sam_j = incrementing parameter for word to other word association in group recall, specifically internal model response to "spoken" response
        Kmax = maximum number of retrieval failures before search process is stopped
        Lmax = max number of retrieval attempts using word cues instead of context
        '''
        


        self.ListLength = ListLength
        self.group_response = group_response
        self.category_size = category_size
        self.t = t
        self.r = r
        self.sam_a = sam_a
        self.sam_b = sam_b
        self.sam_c = sam_c
        self.sam_d = sam_d
        self.sam_e = sam_e
        self.sam_f = sam_f
        self.sam_g = sam_g
        self.sam_h = sam_h
        self.sam_i = sam_i
        self.sam_j = sam_j
        self.Kmax = Kmax
        self.Lmax = Lmax
        self.K = 0
        self.L = 0
        self.context_assoc, self.word_assoc, self.category_list = self.encodeitems()
        
    #method for encoding items
    def create_word_assoc(self):
        
        studyitems = self.create_categories()
        
        word_assoc = np.zeros((self.ListLength, self.ListLength))
        
        for i in range(self.ListLength):
            cat_val = np.where(studyitems == i)[0][0] #get category value for studyitem i
            
            for j in range(self.ListLength):
                cat = np.where(studyitems == j)[0][0]
                if (cat_val == cat): #if item j is in the same category as i
                    word_assoc[i][j] = round(np.random.normal(2, self.sam_h),4)
                else: #else if item j is not in the same category as j
                    word_assoc[i][j] = round(np.random.normal(.05, self.sam_i), 4)
        
        return word_assoc, studyitems

    def create_categories(self):
        #create categories out of list items
        
        ls = np.array(range(self.ListLength))
        
        return ls.reshape(int(self.ListLength/self.category_size), self.category_size)
    
    
    def get_category(self, item):
        #get category of given item
        
        return np.where(self.category_list == item)[0][0]
    
    
    #NOTE: presentation order not randomized between group members
    def encodeitems(self):
        import itertools
        
        buffer = []
        present_order = list(range(self.ListLength))
        
        random.shuffle(present_order) #randomize presentation order
        
        #initialize context association and word-word association matrices
        context_assoc = np.zeros((1, self.ListLength))
        word_assoc, studyitems = self.create_word_assoc()
        
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
       
    
        return context_assoc, word_assoc, studyitems
    
    #method for updating association matrix
    def update_assoc(self, sampledTrace,  wordcue = -1):
    
        if wordcue != -1: #if word was used as cue, update strengths between cue and retrieved image
            
            self.context_assoc[sampledTrace] = self.context_assoc[sampledTrace] + self.sam_e
            self.word_assoc[sampledTrace][sampledTrace] = self.word_assoc[sampledTrace][sampledTrace] + self.sam_g
            self.word_assoc[wordcue][sampledTrace] = self.word_assoc[wordcue][sampledTrace] + self.sam_f 

        
        else: #if only context was used, update association between image to context and image to itself
            self.context_assoc[sampledTrace] = self.context_assoc[sampledTrace] + self.sam_e
            self.word_assoc[sampledTrace][sampledTrace] = self.word_assoc[sampledTrace][sampledTrace] + self.sam_g


    #update associations during group recall    
    def update_assoc_group(self, sampledTrace, wordcue):
        
        self.word_assoc[wordcue][sampledTrace] = self.word_assoc[wordcue][sampledTrace] + self.sam_j
    
    
    
    def context_recall(self):

        start_time = time.time()
        retrieval_fails = []
        if self.K >= self.Kmax:
            return list([float('inf'), -1, -1, retrieval_fails]) #response time, response, category, list of retrieval fails
        
        
        while(self.K < self.Kmax):
           
    
            probSamp = [ci/np.sum(self.context_assoc) for ci in list(self.context_assoc)]
    
            sampledTrace = np.random.choice(a = list(range(self.ListLength)), size = 1, p = probSamp)[0] #begin free recall by using context as a search cue
        
            if (sampledTrace in self.group_response):
                retrieval_fails.append(time.time() - start_time)
                self.K += 1
                
                    
            else: #otherwise, if a new trace was sampled
                    
                probRecover = 1-np.exp(-self.context_assoc[sampledTrace])
                
               
                if (probRecover > np.random.rand(1)[0]):#if recovery is successful, update strength of probe to the recovered memory trace
                   
                    #update internal associations for retrieved memory item even if it's not added to the group_response vector
                    self.update_assoc(sampledTrace)
                    return list([(time.time() - start_time), sampledTrace, self.get_category(sampledTrace), retrieval_fails])
                else:
                    retrieval_fails.append(time.time() - start_time)                    
                    self.K+=1
                    
        return list([float('inf'), -1, -1, retrieval_fails])    

    def wordcue_recall(self, sampledTrace):
        start_time = time.time()
        retrieval_fails = []

        if self.K >= self.Kmax:
            retrieval_fails.append(time.time() - start_time)            
            return float('inf'), -1, -1, retrieval_fails
        

        
        self.L = 0
        while(self.L < self.Lmax):
                        
            previous_sample = sampledTrace
            
            
            probSamp = (self.context_assoc*self.word_assoc[sampledTrace])/sum(self.context_assoc*self.word_assoc[sampledTrace])
            
            #randomly choose a trace using calculated probabilities
            sampledTrace = np.random.choice(a = list(range(self.ListLength)), size = 1, p = probSamp)[0]
            
            
            if (sampledTrace in self.group_response): #if sampledTrace already said, count this as retrieval failure and start again
                retrieval_fails.append(time.time() - start_time)
                self.K += 1
                self.L += 1                     
                sampledTrace = previous_sample

            else: #otherwise, if a new trace was sampled
                
                probRecover = 1-np.exp((-self.context_assoc[sampledTrace])-self.word_assoc[previous_sample][sampledTrace])
                
                
                if (probRecover > np.random.rand(1)[0]):#if recovery is successful, update strength of probe to the recovered memory trace
                    
                    #self.update_assoc(sampledTrace, wordcue = previous_sample)
                    return (time.time() - start_time), sampledTrace, self.get_category(sampledTrace), retrieval_fails


                else:
                    retrieval_fails.append(time.time() - start_time)
                    self.L+=1
                    self.K+=1
                    sampledTrace = previous_sample
                    
        return float('inf'), -1, -1, retrieval_fails

    def extra_wordcue_recall(self, sampledTrace):

        start_time = time.time()
        retrieval_fails = [] #array for keeping track of when retrieval failures happen

        self.L = 0
        while(self.L < self.Lmax):
                        
            previous_sample = sampledTrace
            
            
            probSamp = (self.context_assoc*self.word_assoc[sampledTrace])/sum(self.context_assoc*self.word_assoc[sampledTrace])
            
            #randomly choose a trace using calculated probabilities
            sampledTrace = np.random.choice(a = list(range(self.ListLength)), size = 1, p = probSamp)[0]
            
            
            if (sampledTrace in self.group_response): #if sampledTrace already said, count this as retrieval failure and start again
                retrieval_fails.append(time.time() - start_time) #mark time of second possible retrieval failure

                self.L += 1                     
                sampledTrace = previous_sample

            else: #otherwise, if a new trace was sampled
                
                probRecover = 1-np.exp((-self.context_assoc[sampledTrace])-self.word_assoc[previous_sample][sampledTrace])
                
                
                if (probRecover > np.random.rand(1)[0]):#if recovery is successful, update strength of probe to the recovered memory trace
                    
                    #self.update_assoc(sampledTrace, wordcue = previous_sample) #do this only if this word is chosen 
                    return (time.time() - start_time), sampledTrace, self.get_category(sampledTrace), retrieval_fails


                else:
                    retrieval_fails.append(time.time() - start_time)

                    self.L += 1
                    sampledTrace = previous_sample
                    
        return float('inf'), -1, -1, retrieval_fails
    
       
            