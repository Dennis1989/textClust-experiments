import sys
import numpy as np

sys.path.insert(0, './OriginalMStream')
sys.path.insert(0, './OSDM')
sys.path.insert(0, './textclust')
sys.path.insert(0,'./DPBMM')
sys.path.insert(0, './EStream')
sys.path.insert(0, './rakib-mstream')
sys.path.insert(0, './DCSS')

import dpbmm
import EStream
import om_main
import rakib_main
import osdm
import DCSS_main
import o_read_pred_true_text



## eval for textclust
import text_eval
import time

#from RandomShuffile import SampleData
#import faststream
import json
import pandas as pd
from sklearn import metrics


currentDataset = None
currentAlgo = None
outdir = "Evaluation Results/"
dataDir = "datasets/"
datasets = ["Trends-T"]
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


def evaluate_estream(listtuple_pred_true_text, ignoreMinusOne=False):
    preds =[]
    trues = []
    for pred_true_text in listtuple_pred_true_text:
        if str(pred_true_text[1])=='-1' and ignoreMinusOne==False:
            continue   

        preds.append(pred_true_text[0])
        trues.append(pred_true_text[1])

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

def evaluate_osdm(filename, trues, obsId):
    data = pd.read_csv(filename, sep=" ", names=["docid","clustid"], index_col=False, dtype={"docid":"string","clustid":"int"})
    
    dictionary = {}
    for index, row in data.iterrows():
        dictionary[row["docid"]] = row["clustid"]

    preds = [dictionary[_id] for _id in obsId]

    
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
         

    
    datalen = len(labels)
    print(datalen)
    #datalen = 15000
    print("------------------------ DATASET " +  str(dataset)+ "------------------------")


    print("----------------------------------------------------------------------------")
    print("DCSS")
    print("----------------------------------------------------------------------------")
    
    ## OSDM
    currentAlgo = "DCSS"
    start = time.time() 
    filename = DCSS_main.run(dataDir,dataset, "DCSS_Result/", datalen)
    runtime = time.time()-start

    re = o_read_pred_true_text.ReadOrgiginalMStream("DCSS_Result/result")
    res = evaluate_textclust(labels,re)
    
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
    dat.to_csv(outdir + currentAlgo+"global_results.csv",index=False)

    dat = pd.DataFrame(all_mean_results, columns = ["homogenity", "completeness", "vscore", "nmi", "dataset", "runtime", "algorithm"])
    dat.to_csv(outdir + currentAlgo+"mean_results.csv",index=False)



    print("----------------------------------------------------------------------------")
    print("textclust")
    print("----------------------------------------------------------------------------")
    currentAlgo = "textclust-unigram"
    ## textclust
    start = time.time()
    re = text_eval.run(dataDir+dataset, datalen, gram =1, sigma = 0.5)
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
    dat.to_csv(outdir + currentAlgo+"global_results.csv",index=False)

    dat = pd.DataFrame(all_mean_results, columns = ["homogenity", "completeness", "vscore", "nmi", "dataset", "runtime", "algorithm"])
    dat.to_csv(outdir + currentAlgo+"mean_results.csv",index=False)
    print("----------------------------------------------------------------------------")
    
    print("----------------------------------------------------------------------------")
    print("textclust")
    print("----------------------------------------------------------------------------")
    currentAlgo = "textclust-bigram"
    ## textclust
    start = time.time()
    re = text_eval.run(dataDir+dataset, datalen, gram =2, sigma = 0.5)
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
    dat.to_csv(outdir + currentAlgo+"global_results.csv",index=False)

    dat = pd.DataFrame(all_mean_results, columns = ["homogenity", "completeness", "vscore", "nmi", "dataset", "runtime", "algorithm"])
    dat.to_csv(outdir + currentAlgo+"mean_results.csv",index=False)


    print("----------------------------------------------------------------------------")
    print("EStream")
    print("----------------------------------------------------------------------------")
    currentAlgo = "EStream"
    ## faststream 
    start = time.time()
    re = EStream.run(dataDir+dataset, "EStream_Result/")
    runtime = time.time()-start
    
    res = evaluate_estream(re)
    
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
    dat.to_csv(outdir + currentAlgo+"global_results.csv",index=False)

    dat = pd.DataFrame(all_mean_results, columns = ["homogenity", "completeness", "vscore", "nmi", "dataset", "runtime", "algorithm"])
    dat.to_csv(outdir + currentAlgo+"mean_results.csv",index=False)
    





    print("----------------------------------------------------------------------------")
    print("OSDM")
    print("----------------------------------------------------------------------------")
    
    ## OSDM
    currentAlgo = "OSDM"
    start = time.time()
    filename = osdm.run(dataset, listOfObjects)
    runtime = time.time()-start

    res = evaluate_osdm(filename, labels, obsId)
    
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
    dat.to_csv(outdir + currentAlgo+"global_results.csv",index=False)

    dat = pd.DataFrame(all_mean_results, columns = ["homogenity", "completeness", "vscore", "nmi", "dataset", "runtime", "algorithm"])
    dat.to_csv(outdir + currentAlgo+"mean_results.csv",index=False)


    print("----------------------------------------------------------------------------")

   

    print("----------------------------------------------------------------------------")
    print("DP-BMM")
    print("----------------------------------------------------------------------------")
    currentAlgo = "DP-BMM"
    ## textclust
    start = time.time()
    re = dpbmm.run(dataDir+dataset, datalen)
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
    dat.to_csv(outdir + currentAlgo+"global_results.csv",index=False)

    dat = pd.DataFrame(all_mean_results, columns = ["homogenity", "completeness", "vscore", "nmi", "dataset", "runtime", "algorithm"])
    dat.to_csv(outdir + currentAlgo+"mean_results.csv",index=False)

    

    print("----------------------------------------------------------------------------")
    print("MSTREAM")
    print("----------------------------------------------------------------------------")
    ##  MSTREAM
    currentAlgo = "MSTREAM"
    start = time.time()
    om_main.run(dataDir,dataset, "OriginalMStream_Result/", datalen)
    runtime = time.time()-start

    re = o_read_pred_true_text.ReadOrgiginalMStream("OriginalMStream_Result/result")
    res = evaluate_textclust(labels,re)
    
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
    dat.to_csv(outdir + currentAlgo+"global_results.csv",index=False)

    dat = pd.DataFrame(all_mean_results, columns = ["homogenity", "completeness", "vscore", "nmi", "dataset", "runtime", "algorithm"])
    dat.to_csv(outdir + currentAlgo+"mean_results.csv",index=False)

   
    print("----------------------------------------------------------------------------")
    print("MSTREAM-RAKIB")
    print("----------------------------------------------------------------------------")
    ##  MSTREAM
    currentAlgo = "MSTREAM-RAKIB"
    start = time.time()
    rakib_main.run(dataDir,dataset, "rakib_mstream_result/")
    runtime = time.time()-start

    re = o_read_pred_true_text.ReadPredTrueText("rakib_mstream_result/PredTueTextMStream_WordArr.txt")
    res = evaluate_estream(re)
    
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
    dat.to_csv(outdir + currentAlgo+"global_results.csv",index=False)

    dat = pd.DataFrame(all_mean_results, columns = ["homogenity", "completeness", "vscore", "nmi", "dataset", "runtime", "algorithm"])
    dat.to_csv(outdir + currentAlgo+"mean_results.csv",index=False)

 


dat = pd.DataFrame(all_global_results, columns = ["homogenity", "completeness", "vscore", "nmi", "dataset", "runtime", "algorithm"])
dat.to_csv(outdir + "global_results.csv",index=False)

dat = pd.DataFrame(all_mean_results, columns = ["homogenity", "completeness", "vscore", "nmi", "dataset", "runtime", "algorithm"])
dat.to_csv(outdir + "mean_results.csv",index=False)





