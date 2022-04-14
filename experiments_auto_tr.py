import pickle
import numpy as np
import pandas as pd
import itertools
import multiprocessing
import os
import matplotlib.pyplot as plt
from sklearn import metrics
from textClustPy import textclust

from textClustPy import Preprocessor
from textClustPy import JsonLInput
from textClustPy import microcluster


import gensim
import gensim.downloader as api
from gensim.models import Word2Vec

loadcsv = True
#model = Word2Vec.load("pretrained5class.model")
#model.init_sims(replace=True)
model=None
#model = None
#model = api.load("glove-twitter-50")
#model.init_sims(replace=True)

_lambda = 0.01
radius = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
tgap = 200
minWeight = 1
gram = 1


eval_results = []
all_results = []

def evaluate_batches(trues,preds):
    n = 1000
    tru = [trues[i:i + n] for i in range(0, len(trues), n)]
    pre = [preds[i:i + n] for i in range(0, len(preds), n)]
    nmi = []
    for x in range(0,len(tru)):
        nmi.append(metrics.normalized_mutual_info_score(tru[x], pre[x], average_method='arithmetic'))

    return np.mean(nmi)


def evaluate_textclust(trues, preds):
    #print(evaluate_batches(trues, preds, currentAlgo))
    homo = metrics.homogeneity_score(trues, preds) 
    print("homogeneity_score-whole-data:   %0.8f" % homo)

    completeness = metrics.completeness_score(trues, preds)
    print("completeness_score-whole-data:   %0.8f" % completeness)

    vscore=metrics.v_measure_score(trues, preds)
    print ("v_measure_score-whole-data:   %0.4f" % vscore)

    nmi = metrics.normalized_mutual_info_score(trues, preds, average_method='arithmetic')
    print("nmi_score-whole-data:   %0.8f" % nmi)
    
    nmi   = evaluate_batches(trues, preds)
    #global_eval = [homo,completeness,vscore,nmi]
    return  nmi



datasets =[{"name":"datasets/News-T","length":11108},{"name":"datasets/Tweets-T","length":30321}, {"name":"datasets/NT","length":41428}, {"name":"datasets/NTS","length":61427},{"name":"datasets/SO-T","length":123341},{"name":"datasets/Trends-T","length":199990}]

if not loadcsv:

    for dataset in datasets:
        eval_results = []
        for rad in radius:

            algorithm = textclust(_lambda=_lambda, tgap=tgap, radius=rad, minWeight=minWeight,
                micro_distance="tfidf_cosine_distance", macro_distance="tfidf_cosine_distance", model = None, idf = True, 
                embedding_verification = False, realtimefading=False, auto_merge=True, auto_r=False)
        
            preprocessing = Preprocessor(max_grams=1)

            #csv = CSVInput(textclust = algorithm, preprocessor = preprocessing, config = False, col_id=0, col_time=0, col_text=2, col_label=1, quotechar=None, delimiter="\t", newline = "\n", csvfile=dataset+".csv")

            json = JsonLInput(jsonfile=dataset["name"],col_id="Id",col_time ="Id",col_label="clusterNo",col_text="textCleaned",textclust = algorithm, preprocessor = preprocessing,config = False)
            ## fetch observations from stream
            observations = json.fetch_from_stream(dataset["length"]-1)

            labels = [int(obs.label) for obs in observations]

            predictions = []
            for ob in observations:
                predictions.append(json.processdata(ob))
            
            eval_results.append(evaluate_textclust(labels,predictions))

        all_results.append(eval_results)

    dat = pd.DataFrame(all_results)
    dat = dat.transpose()
    dat.columns = ["News-T","Tweets-T","NT","NTS","SO-T","Trends-T"]
    dat.to_csv("results_radius_on.csv",index=False)


    auto_radius_results = []

    ## next we enable our automatic threshold adaption
    for dataset in datasets:
        algorithm = textclust(_lambda=_lambda, tgap=tgap, radius=0.5, minWeight=minWeight,
                    micro_distance="tfidf_cosine_distance", macro_distance="tfidf_cosine_distance", model = None, idf = True, 
                    embedding_verification = False, realtimefading=False, auto_merge=True, auto_r=True, sigma = 0.5)
            
        preprocessing = Preprocessor(max_grams=1)

        #csv = CSVInput(textclust = algorithm, preprocessor = preprocessing, config = False, col_id=0, col_time=0, col_text=2, col_label=1, quotechar=None, delimiter="\t", newline = "\n", csvfile=dataset+".csv")

        json = JsonLInput(jsonfile=dataset["name"],col_id="Id",col_time ="Id",col_label="clusterNo",col_text="textCleaned",textclust = algorithm, preprocessor = preprocessing,config = False)
        ## fetch observations from stream
        observations = json.fetch_from_stream(dataset["length"]-1)

        labels = [int(obs.label) for obs in observations]

        predictions = []
        for ob in observations:
            predictions.append(json.processdata(ob))
                
        auto_radius_results.append(evaluate_textclust(labels,predictions))

        dat2 = pd.DataFrame(auto_radius_results)
        dat2.to_csv("results_radius_off.csv",index=False)



