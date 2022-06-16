"""REM class

@author: willamannering

"""
import numpy as np

class REM:

    def __init__(self, list_length, geo_dist = 0.4, feats = 12, prob_correct = 0.7, prob_store = 0.1, attempts = 12):
        '''
        list_length = 
        geo_dist = geometric distribution parameter to determine base rate of feature values
        feats = number of features representing an item
        prob_correct = probability of storing a feature correctly
        prob_store = probability of storing a feature in each attempt
        attempts = number of attempts to store a feature
        '''

        self.list_length = list_length
        self.geo_dist = geo_dist
        self.feats = feats
        self.prob_correct = prob_correct
        self.prob_store = prob_store
        self.attempts = attempts

        

    def encode_items(self):
        studylist = np.random.geometric(self.geo_dist, size=self.list_length*self.feats).reshape(self.list_length,self.feats)

        foillist = np.random.geometric(self.geo_dist, size=self.list_length*self.feats).reshape(self.list_length,self.feats)

        memlist = []
        ranlist = []
        switchlist = []

        randomlist = np.random.uniform(0.0,1.0,size=self.list_length*self.feats).reshape(self.list_length,self.feats)#create an array filled with random draws from a uniform distribution from 0 to 1    
        
        encodefeatureTF = np.where(randomlist < (1-(1-self.prob_store)**self.attempts),1,0)#use the randomlist to determine which features to encode.
    
        memlist = np.where(encodefeatureTF, studylist, 0)
    
        ranlist = np.random.uniform(0.0,1.0,size=self.list_length*self.feats).reshape(self.list_length,self.feats)#create an array filled with random draws from a uniform distribution from 0 to 1
    
        switchlist = np.where(((memlist > 0) & (ranlist > self.prob_correct)), 1,0)#determine which feature should be encoded incorrectly
        
    
        memlist = np.where(switchlist == 0, memlist, np.random.geometric(self.geo_dist, size=1))#when a feature is encoded incorrectly, sample reandomly from the g distribution

        return studylist, foillist, memlist  


    def match (self, cue, trace):
        likelihood = 0
        likearray = []
        
        likearray = np.where(((trace>0) & (trace==cue)),(self.prob_correct+(1-self.prob_correct)*self.geo_dist*(1-self.geo_dist)**(trace-1))/(self.geo_dist*(1-self.geo_dist)**(trace-1)),trace)#if the a feature was stored and it matches the cue, compute the likelihood of the #match
        
        likearray = np.where(((trace>0) & (trace!=cue)),(1-self.prob_correct),likearray)#if a feature was stored and does not match the cue, then compute the chance this would occur
        
        likearray = np.where(trace==0, 1, likearray)#if a feature was not stored then there is no diagnostic information
        
        likelihood = np.prod(likearray)#compute the likelihood of the trace
        
        return likelihood


    def recall(self, num_runs):
        HRdata = []
        FARdata = []

        for i in range(num_runs):
            studylist, foillist, memlist = self.encode_items()
            
            targetlist = studylist
                #this is the set of retrieval cues used  on target trials
            odds = np.zeros(self.list_length)
            
            #this is the container for the odds values associated with each test
            for cueNum in range(len(targetlist)):   #for each target match it against the contexts of memory and obtain the likelihood ratio (lambda, j) for each trace
                likes = np.zeros(len(targetlist)*len(memlist)).reshape(len(targetlist),len(memlist))
                
                for traceNum in range(len(memlist)): 
                    likes[cueNum,traceNum]=self.match(targetlist[cueNum],memlist[traceNum])
                    
                odds[cueNum] = np.sum(likes[cueNum])/len(memlist)
                
            HRdata.append(float((odds > 1.0).sum())/float(len(targetlist)))

            odds = np.zeros(self.list_length)
            
            
            for cueNum in range(len(foillist)): #for each foil match it against the contexts of memory and obtain the likelihood ratio (lambda, j) for each trace
                likes = np.zeros(len(targetlist)*len(memlist)).reshape(len(targetlist),len(memlist))
                
                for traceNum in range(len(memlist)): 
                    likes[cueNum,traceNum]=self.match(foillist[cueNum],memlist[traceNum])
                    
                odds[cueNum] = np.sum(likes[cueNum])/len(memlist)
                
            FARdata.append(float((odds > 1.0).sum())/float(len(targetlist)))

        return HRdata, FARdata
