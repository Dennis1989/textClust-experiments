import sys
import numpy as np


sys.path.insert(0, './textclust')



## eval for textclust
import text_eval
import time

import json
import pandas as pd
from sklearn import metrics


currentDataset = None
currentAlgo = None
outdir = "Sensitivity Evaluation Results/"
dataDir = "datasets/"
datasets = ["News-T","Tweets-T","NT","NTS","SO-T","Trends-T"]


def evaluate_batches(trues,preds,algo):
    n = 1000
    tru = [trues[i:i + n] for i in range(0, len(trues), n)]
    pre = [preds[i:i + n] for i in range(0, len(preds), n)]
    homo = []
    completeness = []
    vscore = []
    nmi = []
    for x in range(0,len(tru)):
        homo.append(metrics.homogeneity_score(tru[x], pre[x]))

        completeness.append(metrics.completeness_score(tru[x], pre[x]))

        vscore.append(metrics.v_measure_score(tru[x], pre[x]))

        nmi.append(metrics.normalized_mutual_info_score(tru[x], pre[x], average_method='arithmetic'))
    
    
    dat = pd.DataFrame([homo,completeness,vscore,nmi])
    dat.to_csv(outdir + algo+"_"+currentDataset+"_"+"time_eval.csv",index=False, header=False)
    return [np.mean(homo),np.mean(completeness),np.mean(vscore),np.mean(nmi)]




def evaluate_textclust(trues, preds):
    print(evaluate_batches(trues, preds, currentAlgo))
    homo = metrics.homogeneity_score(trues, preds) 
    print("homogeneity_score-whole-data:   %0.8f" % homo)

    completeness = metrics.completeness_score(trues, preds)
    print("completeness_score-whole-data:   %0.8f" % completeness)

    vscore=metrics.v_measure_score(trues, preds)
    print ("v_measure_score-whole-data:   %0.4f" % vscore)

    nmi = metrics.normalized_mutual_info_score(trues, preds, average_method='arithmetic')
    print("nmi_score-whole-data:   %0.8f" % nmi)
    
    mean_eval   = evaluate_batches(trues, preds, currentAlgo)
    global_eval = [homo,completeness,vscore,nmi]
    return [global_eval,mean_eval]

all_global_results =[]
all_mean_results =[]
sigmas = [0.5]

for dataset in datasets:
    currentDataset = dataset
    #SampleData(dataDir+dataset,1,10)

    ## lets load the dataset 
    listOfObjects = []
    with open(dataDir+dataset) as input:  #load all the objects in memory
        line = input.readline()
        counter = 1
        while line:
            counter +=1
            obj = json.loads(line)  # a line is a document represented in JSON
            listOfObjects.append(obj)
            line = input.readline()
        obsId = [x["Id"] for x in listOfObjects]
        labels = [x["clusterNo"] for x in listOfObjects]
         
    for sigma in sigmas:
    
        datalen = len(labels)
        #datalen = 15000
        print("------------------------ DATASET " +  str(dataset)+ "------------------------")
        
        print("----------------------------------------------------------------------------")
        print("textclust")
        print("----------------------------------------------------------------------------")
        currentAlgo = "textclust-unigram_"+str(sigma)+"_"
        ## textclust
        start = time.time()
        re = text_eval.run(dataDir+dataset, datalen, gram =1, sigma = sigma)
        runtime = time.time()-start

        res = evaluate_textclust(re[0], re[1])
        global_result = res[0]
        mean_result = res[1]
        
        global_result.append(dataset)
        global_result.append(str(runtime))
        global_result.append(currentAlgo)

        mean_result.append(dataset)
        mean_result.append(str(runtime))
        mean_result.append(currentAlgo)
        
        all_global_results.append(global_result)
        all_mean_results.append(mean_result)
        
        dat = pd.DataFrame(all_global_results, columns = ["homogenity", "completeness", "vscore", "nmi", "dataset", "runtime", "algorithm"])
        dat.to_csv(outdir + currentAlgo+"global_results_sigma_"+str(sigma)+".csv",index=False)

        dat = pd.DataFrame(all_mean_results, columns = ["homogenity", "completeness", "vscore", "nmi", "dataset", "runtime", "algorithm"])
        dat.to_csv(outdir + currentAlgo+"mean_results_sigma_"+str(sigma)+".csv",index=False)
        print("----------------------------------------------------------------------------")


        print("----------------------------------------------------------------------------")
        print("textclust")
        print("----------------------------------------------------------------------------")
        currentAlgo = "textclust-bigram_"+str(sigma)+"_"
        ## textclust
        start = time.time()
        re = text_eval.run(dataDir+dataset, datalen, gram =2, sigma = sigma)
        runtime = time.time()-start

        res = evaluate_textclust(re[0], re[1])
        global_result = res[0]
        mean_result = res[1]
        
        global_result.append(dataset)
        global_result.append(str(runtime))
        global_result.append(currentAlgo)

        mean_result.append(dataset)
        mean_result.append(str(runtime))
        mean_result.append(currentAlgo)
        
        all_global_results.append(global_result)
        all_mean_results.append(mean_result)
        
        dat = pd.DataFrame(all_global_results, columns = ["homogenity", "completeness", "vscore", "nmi", "dataset", "runtime", "algorithm"])
        dat.to_csv(outdir + currentAlgo+"global_results_sigma_"+str(sigma)+".csv",index=False)

        dat = pd.DataFrame(all_mean_results, columns = ["homogenity", "completeness", "vscore", "nmi", "dataset", "runtime", "algorithm"])
        dat.to_csv(outdir + currentAlgo+"mean_results_sigma_"+str(sigma)+".csv",index=False)
        print("----------------------------------------------------------------------------")
        
    