# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 18:45:40 2016

@author: NISHIT
"""

import time
start_exec_time = time.time()

from sentiment_reader import SentimentCorpus
from lr import LogisticRegression

if __name__ == '__main__':
    dataset = SentimentCorpus()
    lr = LogisticRegression()
    lr.LR(dataset.train_X, dataset.train_y,dataset.test_X, dataset.test_y)
    
    print("Execution time: --- %s seconds ---" % (time.time() - start_exec_time))