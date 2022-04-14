from DCSS import DCSS
import json
import time

dataDir = "data/"
outputPath = "result/"

#dataset = "Tweets"
#dataset = "Tweets-T"
# dataset = "News"
#dataset = "News-T"
dataset = "sorted_tweet"

timefil = "timefil"
MaxBatch = 2 # The number of saved batches + 1
AllBatchNum = 16 # The number of batches you want to devided the dataset to
alpha = 0.03
beta = 0.04
iterNum = 5
sampleNum = 5
wordsInTopicNum = 5
K = 0 

def run(dataDir,dat,output, datalen):
    
    dataDir = dataDir
    global outputPath
    outputPath = output

    global dataset
    dataset = dat

    timefil = "timefil"
    MaxBatch = 2 # The number of saved batches + 1
    AllBatchNum = (int)(datalen/1000)+1 # The number of batches you want to devided the dataset to
    alpha = 0.2
    beta = 0.04
    iterNum = 1
    sampleNum = 1
    wordsInTopicNum = 5
    K = 0 

    return runDCSS(K, MaxBatch, AllBatchNum, alpha, beta, iterNum, sampleNum, dataset, timefil, wordsInTopicNum)

def runDCSS(K, MaxBatch, AllBatchNum, alpha, beta, iterNum, sampleNum, dataset, timefil, wordsInTopicNum):
    dcss = DCSS(K, MaxBatch, AllBatchNum, alpha, beta, iterNum, sampleNum, dataset, timefil, wordsInTopicNum)
    dcss.getDocuments()
    for sampleNo in range(1, sampleNum+1):
        print("SampleNo:"+str(sampleNo))
        dcss.runDCSS(sampleNo, outputPath)

def runWithAlphaScale(beta, K, MaxBatch, AllBatchNum, iterNum, sampleNum, dataset, timefil, wordsInTopicNum):
    parameters = []
    timeArrayOfParas = []

    p = 0.03
    while p <= 0.1:
        alpha = p
        parameters.append(p)
        print("alpha:", alpha, "\tp:", p)
        dcss = DCSS(K, MaxBatch, AllBatchNum, alpha, beta, iterNum, sampleNum, dataset, timefil, wordsInTopicNum)
        dcss.getDocuments()
        timeArray = []
        for sampleNo in range(1, sampleNum + 1):
            print("SampleNo:", sampleNo)
            startTime = time.time()
            dcss.runDCSS(sampleNo, outputPath)
            endTime = time.time()
            timeArray.append(int(endTime - startTime))
        timeArrayOfParas.append(timeArray)
        p += 0.01

    fileParameters = "DCSSDiffAlpha" + "K" + str(K) + "iterNum" + str(iterNum) + "SampleNum" + \
                     str(sampleNum) + "beta" + str(round(beta, 3)) + \
                        "BatchNum" + str(AllBatchNum) + "BatchSaved" + str(MaxBatch)
    outTimePath = outputPath + "Time" + dataset + fileParameters + ".txt"
    writer = open(outTimePath, 'w')
    parasNum = parameters.__len__()
    for i in range(parasNum):
        temp_obj = {}
        temp_obj['parameter'] = parameters[i]
        temp_obj['Time'] = timeArrayOfParas[i]
        temp_json = json.dumps(temp_obj)
        writer.write(temp_json)
        writer.write('\n')
    writer.close()

def runWithBetas(alpha, K, MaxBatch, AllBatchNum, iterNum, sampleNum, dataset, timefil, wordsInTopicNum):
    parameters = []
    timeArrayOfParas = []
    beta = 0.01
    while beta <= 0.21:
        parameters.append(beta)
        print("beta:", beta)
        dcss = DCSS(K, MaxBatch, AllBatchNum, alpha, beta, iterNum, sampleNum, dataset, timefil, wordsInTopicNum)
        dcss.getDocuments()
        timeArray = []
        for sampleNo in range(1, sampleNum + 1):
            print("SampleNo:", sampleNo, end=' ')
            #startTime = time.time()
            dcss.runDCSS(sampleNo, outputPath)
            #endTime = time.time()
            #timeArray.append(int(endTime - startTime))
        timeArrayOfParas.append(timeArray)
        beta += 0.01
    fileParameters = "DCSSDiffBeta" + "K" + str(K) + "iterNum" + str(iterNum) + "SampleNum" + str(sampleNum) + \
                     "alpha" + str(round(alpha, 3)) + \
                        "BatchNum" + str(AllBatchNum) + "BatchSaved" + str(MaxBatch)
    outTimePath = outputPath + "Time" + dataset + fileParameters + ".txt"
    writer = open(outTimePath, 'w')
    parasNum = parameters.__len__()
    for i in range(parasNum):
        temp_obj = {}
        temp_obj['parameter'] = parameters[i]
        temp_obj['Time'] = timeArrayOfParas[i]
        temp_json = json.dumps(temp_obj)
        writer.write(temp_json)
        writer.write('\n')
    writer.close()

def runWithNiters(K, MaxBatch, AllBatchNum, alpha, beta, sampleNum, dataset, timefil, wordsInTopicNum):
    parameters = []
    timeArrayOfParas = []
    iterNum = 0
    while iterNum <= 30.01:
        parameters.append(iterNum)
        print("iterNum:", iterNum)
        dcss = DCSS(K, MaxBatch, AllBatchNum, alpha, beta, iterNum, sampleNum, dataset, timefil, wordsInTopicNum)
        dcss.getDocuments()
        timeArray = []
        for sampleNo in range(1, sampleNum + 1):
            print("SampleNo:", sampleNo, end=' ')
            startTime = time.time()
            dcss.runDCSS(sampleNo, outputPath)
            endTime = time.time()
            timeArray.append(int(endTime - startTime))
        timeArrayOfParas.append(timeArray)
        iterNum += 1
    fileParameters = "DCSSDiffIter" + "K" + str(K) + "SampleNum" + str(sampleNum) + \
                     "alpha" + str(round(alpha, 3)) + "beta" + str(round(beta, 3)) + \
                        "BatchNum" + str(AllBatchNum) + "BatchSaved" + str(MaxBatch)
    outTimePath = outputPath + "Time" + dataset + fileParameters + ".txt"
    writer = open(outTimePath, 'w')
    parasNum = parameters.__len__()
    for i in range(parasNum):
        temp_obj = {}
        temp_obj['parameter'] = parameters[i]
        temp_obj['Time'] = timeArrayOfParas[i]
        temp_json = json.dumps(temp_obj)
        writer.write(temp_json)
        writer.write('\n')
    writer.close()


if __name__ == '__main__':

    runDCSS(K, MaxBatch, AllBatchNum, alpha, beta, iterNum, sampleNum, dataset, timefil, wordsInTopicNum)
    #runWithAlphaScale(beta, K, MaxBatch, AllBatchNum, iterNum, sampleNum, dataset, timefil, wordsInTopicNum)
    #runWithBetas(alpha, K, MaxBatch, AllBatchNum, iterNum, sampleNum, dataset, timefil, wordsInTopicNum)
    # runWithNiters(K, MaxBatch, AllBatchNum, alpha, beta, sampleNum, dataset, timefil, wordsInTopicNum)
    # runWithBatchNum(K, MaxBatch, alpha, beta, iterNum, sampleNum, dataset, timefil, wordsInTopicNum)
    # runWithMaxBatch(K, AllBatchNum, alpha, beta, iterNum, sampleNum, dataset, timefil, wordsInTopicNum)
