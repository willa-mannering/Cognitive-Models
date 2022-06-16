
# -*- coding: utf-8 -*-
"""
Created on Thurs June 16 2022

@author: willamannering

"""
import numpy as np
from REM import REM

def run_recall(num_runs, list_length):
    model = REM(list_length)

    HRdata, FARdata = model.recall(num_runs)
    
    print ("Avg HR = ", float(sum(HRdata))/float(num_runs))
    print ("Avg FAR = ", float(sum(FARdata))/float(num_runs))

def main():
    run_recall(40, 10)

if __name__ == '__main__':
    main()
