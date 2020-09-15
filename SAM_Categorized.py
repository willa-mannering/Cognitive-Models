#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 16:32:58 2020

@author: willamannering
"""
import numpy as np

class SAM_Categorized:
    
    def __init__(self, ListLength, category_size, t=2, sam_a = 1.2, sam_b = 1.7, sam_d = .2, sam_e = 2, Kmax = 15, Lmax = 15):
        
        ''' ListLength = number of items in studylist, 
        category_size = number of words per category
        t = presentation time per word
        sam_a = multiplier for context to word association
        sam_b = mulitplier for category cue to c-image association
        sam_d = multiplier for category cue to nc-image association (residual)
        sam_e = incrementing parameter for context to word association and category to c-image
        Kmax = maximum number of retrieval failures before search process is stopped
        Lmax = max number of retrieval attempts using word cues instead of context
        
        c-image = item belonging to category c
        nc-image = item outside category c
        '''
        


        self.ListLength = ListLength
        self.category_size = category_size
        self.t = t
        self.sam_a = sam_a
        self.sam_b = sam_b
        self.sam_d = sam_d
        self.sam_e = sam_e
        self.Kmax = Kmax
        self.Lmax = Lmax
        self.K = 0 #global retrieval failure counter
        self.L = 0 #within category word retrieval failure counter
        self.context_assoc, self.category_assoc, self.category_list = self.encodeitems()
        
        
    def create_categories(self):
        #create categories out of list items
        
        ls = np.array(range(self.ListLength))
        
        return ls.reshape(int(self.ListLength/self.category_size), self.category_size)
    
    def get_category(self, item):
        #get category of given item
        
        return np.where(self.category_list == item)[0][0]
    
    
    def encodeitems(self):
        #method for encoding items
        
        studyitems = self.create_categories()

        #initialize context association and word-word association matrices
        context_assoc = np.full(self.ListLength, self.sam_a*self.t)
        category_assoc = np.zeros((self.ListLength, self.ListLength))
        
        for i in range(self.ListLength):
            cat_val = np.where(studyitems == i)[0][0] #get category value for studyitem i
            
            for j in range(self.ListLength):
                cat = np.where(studyitems == j)[0][0]
                if (cat_val == cat): #if item j is in the same category as i
                    category_assoc[i][j] = self.sam_b*self.t
                else: #else if item j is not in the same category as j
                    category_assoc[i][j] = self.sam_d*self.t

    
        return context_assoc, category_assoc, studyitems
    
    
    def update_assoc(self, sampledTrace):
        #method for updating association matrices
        
        #update image to context strength
        self.context_assoc[sampledTrace] = self.context_assoc[sampledTrace] + self.sam_e
        
        #update c-image to category strength
        self.category_assoc[sampledTrace][sampledTrace] = self.category_assoc[sampledTrace][sampledTrace] + self.sam_e

        
        
   
    def free_recall(self):
         #free recall process 
        ''' ListLength = number of items in studylist, 
            t = presentation time per word
            sam_a = multiplier for context to word association
            sam_b = mulitplier for word cue to other word trace association
            sam_d = residual strength of association for words that never appear in buffer together
            sam_e = incrementing parameter for context to word association
            Kmax = maximum number of retrieval failures before search process is stopped
            Lmax = max number of retrieval attempts using word cues instead of context
    
            '''

        #begin free recall process
        alreadySaid = [False] * self.ListLength #at this point no words have been said from the list
        categorySeen = [False] * int(self.ListLength/self.category_size) #at this point no categories have been sampled

        response = [] #empty list of free recall responses
        category_response = [] #empty list of categories sampled
          
        while(self.K < self.Kmax): #begin recall by using context to sample
            
            self.L = 0 #set L to 0
    
            probSamp = [ci/np.sum(self.context_assoc) for ci in list(self.context_assoc)] #get probability of sampling a memory trace
    
            sampledTrace = np.random.choice(a = list(range(self.ListLength)), size = 1, p = probSamp)[0] #begin free recall by using context as a search cue
            
            
            if (alreadySaid[sampledTrace]): #if sampled trace has already been said, count as retrieval failure
                self.K += 1
                
            if categorySeen[self.get_category(sampledTrace)]: #if category has already been sampled count as retrieval failure
                self.K +=1 
                
                    
            else: #otherwise, if a new trace was sampled
                    
                probRecover = 1-np.exp(-self.context_assoc[sampledTrace]) #calculate probability of recovering the trace
                
               
                if (probRecover > np.random.rand(1)[0]):#if recovery is successful, update strength of probe to the recovered memory trace
                    
                    current_category = self.get_category(sampledTrace) #get category of current sampled trace
                   
                    self.update_assoc(sampledTrace)
                    
                    alreadySaid[sampledTrace] = True #set this trace to already said
                    categorySeen[self.get_category(sampledTrace)] = True #set this category to already seen
                    
                    response.append((sampledTrace, self.get_category(sampledTrace))) #record this trace as a successful response
                    category_response.append(self.get_category(sampledTrace)) #record this category as represented
                    
                    #if trace was recovered, search through the current category until Lmax
                    while(self.L < self.Lmax):
                        
                        probSamp = (self.context_assoc*self.category_assoc[sampledTrace])/sum(self.context_assoc*self.category_assoc[sampledTrace])
                        
                        #randomly choose a trace using calculated probabilities
                        sampledTrace = np.random.choice(a = list(range(self.ListLength)), size = 1, p = probSamp)[0]
                       
                        if (alreadySaid[sampledTrace]) or ((self.get_category(sampledTrace)) != current_category): #if sampledTrace already said or outside current category, count this as retrieval failure and start again
                            self.L += 1                     
                            
                        else: #otherwise, if a new trace was successfully sampled
                            
                            probRecover = 1-np.exp((-self.context_assoc[sampledTrace])-self.category_assoc[sampledTrace][sampledTrace])
                            
                            
                            if (probRecover > np.random.rand(1)[0]):#if recovery is successful, update strength of probe to the recovered memory trace
                               
                                self.update_assoc(sampledTrace)
                                alreadySaid[sampledTrace] = True #set this trace to already said
                                
                                response.append((sampledTrace, self.get_category(sampledTrace)))#add trace as successful response
                                    
                                self.L = 0 #set L to zero and continue searching through this category
        
                            else:
                                self.L+=1
                                
                else:
                    self.K += 1
        
        
        return response, category_response #return responses and categories sampled
     
def individual_recall(num_runs, ListLength, category_size):
    '''num_runs = number of runs,
       ListLength = length of study list
       category_size = number of words per category'''
       
    overall_recall = [] #how many words overall were recalled
    category_recall = [] #how many categories are represented in recall
    instances_recall = [] #how many words per category were recalled
    
    for i in range(num_runs):
        
        sam = SAM_Categorized(ListLength, category_size)
        
        words_recalled, categories_recalled = sam.free_recall()
        
        overall_recall.append(len(words_recalled))
        
        category_recall.append(len(categories_recalled))
        
        temp = []
        for w in words_recalled:
            temp.append(w[1]) #make list of all categories represented in responses
        
        instances_recall.append(len(temp)/int(ListLength/category_size))
            
    
    return 'Overall Recall: {} Category Recall: {} Instance Recall: {}'.format(np.mean(overall_recall), np.mean(category_recall), np.mean(instances_recall))
    
        
 
        