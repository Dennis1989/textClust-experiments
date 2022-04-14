import pickle
import numpy as np
import pandas as pd
import itertools
import multiprocessing
import os

from textClustPy import textclust
from textClustPy import TwitterInput
from textClustPy import Preprocessor
from textClustPy import CSVInput
from textClustPy import JsonLInput
from textClustPy import microcluster

import gensim
import gensim.downloader as api
from gensim.models import Word2Vec
def run(dataset, size, gram, sigma):
    #model = Word2Vec.load("pretrained5class.model")
    #model.init_sims(replace=True)
    #model="skipgram"
    model = None
    #model = api.load("glove-twitter-50")
    #model.init_sims(replace=True)
    
    _lambda = 0.01
    radius = 0.9
    tgap = 200
    minWeight = 1

    algorithm = textclust(_lambda=_lambda, tgap=tgap, radius=radius, minWeight=minWeight,
    micro_distance="tfidf_cosine_distance", macro_distance="tfidf_cosine_distance", model = None, idf = True, 
    embedding_verification = False, realtimefading=False, auto_merge=True, auto_r=True,sigma=sigma)
        
    preprocessing = Preprocessor(max_grams=gram)

    #csv = CSVInput(textclust = algorithm, preprocessor = preprocessing, config = False, col_id=0, col_time=0, col_text=2, col_label=1, quotechar=None, delimiter="\t", newline = "\n", csvfile=dataset+".csv")

    json = JsonLInput(jsonfile=dataset,col_id="Id",col_time ="Id",col_label="clusterNo",col_text="textCleaned",textclust = algorithm, preprocessor = preprocessing,config = False)
    ## fetch observations from stream
    observations = json.fetch_from_stream(size-1)

    labels = [int(obs.label) for obs in observations]

    predictions = []
    for ob in observations:
        predictions.append(json.processdata(ob))
                
    return[labels,predictions]

                