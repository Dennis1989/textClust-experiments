import random
import os
import time
import json
import copy

threshold_fix = 1.1
threshold_init = 0

class Model:

    def __init__(self, K, Max_Batch, V, iterNum,alpha0, beta, dataset, ParametersStr, sampleNo, wordsInTopicNum, timefil):
        self.dataset = dataset
        self.ParametersStr = ParametersStr
        self.alpha0 = alpha0
        self.beta = beta
        self.K = K
        self.Kin = K
        self.V = V
        self.iterNum = iterNum
        self.beta0 = float(V) * float(beta)
        self.sampleNo = sampleNo
        self.wordsInTopicNum = copy.deepcopy(wordsInTopicNum)
        self.Max_Batch = Max_Batch  # Max number of batches we will consider
        self.phi_zv = []

        self.batchNum2tweetID = {} # batch num to tweet id
        self.batchNum = 1 # current batch number
        self.readTimeFil(timefil)
        writer = open("DCSS_Result/result", 'w')
        writer.close()
    def readTimeFil(self, timefil):
        try:
            with open(timefil) as timef:
                for line in timef:
                    buff = line.strip().split(' ')
                    if buff == ['']:
                        break
                    self.batchNum2tweetID[self.batchNum] = int(buff[1])
                    self.batchNum += 1
            self.batchNum = 1
        except:
            print("No timefil!")
        # print("There are", self.batchNum2tweetID.__len__(), "time points.\n\t", self.batchNum2tweetID)

    def getAveBatch(self, documentSet, AllBatchNum):
        self.batchNum2tweetID.clear()
        temp = self.D_All / AllBatchNum
        _ = 0
        count = 1
        for d in range(self.D_All):
            if _ < temp:
                _ += 1
                continue                                                       #小于一批的数量不做处理   
            else:
                document = documentSet.documents[d]
                documentID = document.documentID
                self.batchNum2tweetID[count] = documentID                      #多出一批的数量的文档，提取第二批的第一个文档的ID
                count += 1                                                     #批数加1
                _ = 0                                                          #当前批次记录归0，个人觉得应当归1，因为当前文档对应的批内记录为第0，而不是下一文档的批内记录为第0
        self.batchNum2tweetID[count] = -1                                      


    def run_DCSS(self, documentSet, outputPath, wordList, AllBatchNum):
        self.D_All = documentSet.D  # The whole number of documents
        self.z = {}  # Cluster assignments of each document                 (documentID -> clusterID)
        self.m_z = {}  # The number of documents in cluster z               (clusterID -> number of documents)
        self.m_z_l = {}#上一批次cluster z中文档的数量
        self.n_z = {}  # The number of words in cluster z                   (clusterID -> number of words)
        self.n_z_l = {}
        self.n_zv = {}  # The number of occurrences of word v in cluster z  (n_zv[clusterID][wordID] = number)
        self.n_zv_l = {}
        self.currentDoc = 0  # Store start point of next batch
        self.startDoc = 0  # Store start point of this batch
        self.D = 0  # The number of documents currently
        self.K_current = copy.deepcopy(self.K) # the number of cluster containing documents currently
        
        self.V_current = [] # Store word-IDs' list of each batch

        # Get batchNum2tweetID by AllBatchNum
        self.getAveBatch(documentSet, AllBatchNum)
        print("batchNum2tweetID is ", self.batchNum2tweetID)

        while self.currentDoc < self.D_All:
            print("Batch", self.batchNum)
            if self.batchNum not in self.batchNum2tweetID:
                break
            self.intialize(documentSet)
            self.gibbsSampling(documentSet)
            
            print("\tGibbs sampling successful! Start to saving results.")
            self.output(documentSet, outputPath, wordList, self.batchNum - 1)
            print("\tSaving successful!")

            writer = open("DCSS_Result/result", 'a+')
            for d in range(self.startDoc, self.currentDoc):
                documentID = documentSet.documents[d].documentID
                cluster = self.z[documentID]
                writer.write(str(documentID) + "\t" + str(cluster) + "\n")
            writer.close()

            self.D = 0
            self.m_z_l = copy.deepcopy(self.m_z)
            self.n_z_l = copy.deepcopy(self.n_z)
            self.n_zv_l = copy.deepcopy(self.n_zv)
            self.m_z = {}
            self.n_z = {}
            self.n_zv = {}
    # Compute beta0 for every batch
    def getBeta0(self):
        Words = []
        if self.batchNum < 5:
            for i in range(1, self.batchNum + 1):
                Words = list(set(Words + self.word_current[i]))
        if self.batchNum >= 5:
            for i in range(self.batchNum - 4, self.batchNum + 1):
                Words = list(set(Words + self.word_current[i]))
        return (float(len(list(set(Words)))) * float(self.beta))

    def intialize(self, documentSet):
        for d in range(self.currentDoc, self.D_All):
            document = documentSet.documents[d]
            documentID = document.documentID
            # This method is getting beta0 at the beginning of initialization considering the whole words in current batch
            for w in range(document.wordNum):
                wordNo = document.wordIdArray[w]
                if wordNo not in self.V_current:
                    self.V_current.append(wordNo)
            if documentID != self.batchNum2tweetID[self.batchNum]:
                self.D += 1
            else:
                break
        self.beta0 = float(len(self.V_current)) * float(self.beta)
        if self.beta0 <= 0:
                print("Wrong V!")
                exit(-1)
        self.alpha = self.alpha0 * self.D
        print("\t" + str(self.D) + " documents will be analyze. alpha is" + " %.2f." % self.alpha + "\n\tInitialization.", end='\n')
        for d in range(self.currentDoc, self.D_All):
            document = documentSet.documents[d]
            documentID = document.documentID

            # This method is getting beta0 before each document is initialized
            #for w in range(document.wordNum):
            #    wordNo = document.wordIdArray[w]
            #    if wordNo not in self.V_current:
            #        self.V_current.append(wordNo)
            
            if documentID != self.batchNum2tweetID[self.batchNum]:
                if self.batchNum ==1:
                    cluster = self.sampleCluster(d, document, documentID, 0)
                else:
                    cluster = self.newsampleCluster(d, document, documentID, 0)
                self.z[documentID] = cluster
                if cluster not in self.m_z:
                    self.m_z[cluster] = 0
                self.m_z[cluster] += 1
                for w in range(document.wordNum):
                    wordNo = document.wordIdArray[w]
                    wordFre = document.wordFreArray[w]
                    if cluster not in self.n_zv:
                        self.n_zv[cluster] = {}
                    if wordNo not in self.n_zv[cluster]:
                        self.n_zv[cluster][wordNo] = 0
                    self.n_zv[cluster][wordNo] += wordFre
                    if cluster not in self.n_z:
                        self.n_z[cluster] = 0
                    self.n_z[cluster] += wordFre
                if d == self.D_All - 1:
                    self.startDoc = self.currentDoc
                    self.currentDoc = self.D_All
                    self.batchNum += 1
            else:
                self.startDoc = self.currentDoc
                self.currentDoc = d
                self.batchNum += 1
                break

    def gibbsSampling(self, documentSet):
        for i in range(self.iterNum):
            print("\titer is ", i+1, end='\n')
            for d in range(self.startDoc, self.currentDoc):
                document = documentSet.documents[d]
                documentID = document.documentID
                cluster = self.z[documentID]
                self.m_z[cluster] -= 1
                for w in range(document.wordNum):
                    wordNo = document.wordIdArray[w]
                    wordFre = document.wordFreArray[w]
                    self.n_zv[cluster][wordNo] -= wordFre
                    self.n_z[cluster] -= wordFre
                self.checkEmpty(cluster)
                if i != self.iterNum - 1:  # if not last iteration
                    if self.batchNum - 1 == 1:
                        cluster = self.sampleCluster(d, document, documentID, 0)
                    else:
                        cluster = self.newsampleCluster(d, document, documentID, 0)
                elif i == self.iterNum - 1:  # if last iteration
                    if self.batchNum - 1 == 1:
                        cluster = self.sampleCluster(d, document, documentID, 1)
                    else:
                        cluster = self.newsampleCluster(d, document, documentID, 1)
                self.z[documentID] = cluster
                if cluster not in self.m_z:
                    self.m_z[cluster] = 0
                self.m_z[cluster] += 1
                for w in range(document.wordNum):
                    wordNo = document.wordIdArray[w]
                    wordFre = document.wordFreArray[w]
                    if cluster not in self.n_zv:
                        self.n_zv[cluster] = {}
                    if wordNo not in self.n_zv[cluster]:
                        self.n_zv[cluster][wordNo] = 0
                    if cluster not in self.n_z:
                        self.n_z[cluster] = 0
                    self.n_zv[cluster][wordNo] += wordFre
                    self.n_z[cluster] += wordFre
        
        return

    def sampleCluster(self, d, document, documentID, isLast):
        prob = [float(0.0)] * (self.K + 1)
        for cluster in range(self.K):
            if cluster not in self.m_z or self.m_z[cluster] == 0:
                prob[cluster] = 0
                continue                                                       
            prob[cluster] = self.m_z[cluster] #/ (self.D - 1 + self.alpha)     mt,k
            denominator = 1.0
            for i in range(document.len_d):
                denominator *= self.n_z[cluster] + self.beta0 + i
            valueOfRule2 = 1.0            
            for w in range(document.wordNum):
                wordNo = document.wordIdArray[w]
                wordFre = document.wordFreArray[w]                             #n_d_w 
                for j in range(wordFre):
                    if wordNo not in self.n_zv[cluster]:
                        self.n_zv[cluster][wordNo] = 0
                    #公式右半部
                    valueOfRule2 *= self.n_zv[cluster][wordNo] + self.beta + j
            valueOfRule2 /= denominator
            prob[cluster] *= valueOfRule2
        prob[self.K] = self.alpha #/ (self.D - 1 + self.alpha)                 由于公式左半部分分母相同，故去掉
        denominator = 1.0
        for i in range(document.len_d):
            denominator *= self.beta0 + i
        valueOfRule2 = 1.0
        for w in range(document.wordNum):
            wordFre = document.wordFreArray[w]
            for j in range(wordFre):
                valueOfRule2 *= self.beta + j 
        valueOfRule2 /= denominator        
        prob[self.K] *= valueOfRule2

        allProb = 0 # record the amount of all probabilities
        prob_normalized = [] # record normalized probabilities                 之后未用到  
        for k in range(self.K + 1):
            allProb += prob[k]
        for k in range(self.K + 1):
            if(allProb != 0):
                prob_normalized.append(prob[k]/allProb)
            else:
                print("is 0!")
                prob_normalized.append(1)

        kChoosed = 0
        if isLast == 0:
            for k in range(1, self.K + 1):
                prob[k] += prob[k - 1]
            #random.seed(2)
            thred = random.random() * prob[self.K]                             #对总概率乘上一个随机数
            while kChoosed < self.K + 1:
                if thred < prob[kChoosed]:
                    break                                                      #大于总概率随机数，则选中该主题
                kChoosed += 1
            if kChoosed == self.K:                                             #代表选择了新的类
                self.K += 1
                self.K_current += 1
        else:
            bigPro = prob[0]
            for k in range(1, self.K + 1):
                if prob[k] > bigPro:
                    bigPro = prob[k]
                    kChoosed = k
            if kChoosed == self.K:
                self.K += 1
                self.K_current += 1
        return kChoosed

    def newsampleCluster(self, d, document, documentID, isLast):
        prob = [float(0.0)] * (self.K + 1)
        for cluster in range(self.K):
            if cluster not in self.m_z or self.m_z[cluster] == 0:
                prob[cluster] = 0
                continue                                                       
            if cluster not in self.m_z_l:
                self.m_z_l[cluster] = 0
            prob[cluster] = self.m_z[cluster] + self.alpha0 * self.m_z_l[cluster]#/ (self.D - 1 + self.alpha)     mt,k
            if cluster not in self.n_z_l:
                        self.n_z_l[cluster] = 0
            denominator = 1.0
            for i in range(document.len_d):
                denominator *= self.n_z[cluster] + self.beta*self.n_z_l[cluster] + self.beta0 + i
            valueOfRule2 = 1.0            
            for w in range(document.wordNum):
                wordNo = document.wordIdArray[w]
                wordFre = document.wordFreArray[w]                             #n_d_w 
                
                for j in range(wordFre):
                    if wordNo not in self.n_zv[cluster]:
                        self.n_zv[cluster][wordNo] = 0
                    if cluster not in self.n_zv_l:
                        self.n_zv_l[cluster] = {}
                    if wordNo not in self.n_zv_l[cluster]:
                        self.n_zv_l[cluster][wordNo] = 0
                    if cluster not in self.n_z_l:
                        self.n_z_l[cluster] = 0
                    #公式右半部
                  
                    valueOfRule2 *= self.n_zv[cluster][wordNo] + self.beta * self.n_zv_l[cluster][wordNo] + self.beta + j
            valueOfRule2 /= denominator        
            prob[cluster] *= valueOfRule2
        prob[self.K] = self.alpha #/ (self.D - 1 + self.alpha)                 由于公式左半部分分母相同，故去掉
        denominator = 1.0
        for i in range(document.len_d):
            denominator *= self.beta0 + i
        valueOfRule2 = 1.0
        
        for w in range(document.wordNum):
            wordFre = document.wordFreArray[w]
            for j in range(wordFre):
                valueOfRule2 *= self.beta + j
        valueOfRule2 /= denominator        
        prob[self.K] *= valueOfRule2

        allProb = 0 # record the amount of all probabilities
        prob_normalized = [] # record normalized probabilities                 之后未用到  
        for k in range(self.K + 1):
            allProb += prob[k]
        for k in range(self.K + 1):
            if(allProb != 0):
                prob_normalized.append(prob[k]/allProb)
            else:
                print("is 0!")
                prob_normalized.append(1)

        kChoosed = 0
        if isLast == 0:
            for k in range(1, self.K + 1):
                prob[k] += prob[k - 1]
            #random.seed(2)
            thred = random.random() * prob[self.K]                             #对总概率乘上一个随机数
            while kChoosed < self.K + 1:
                if thred < prob[kChoosed]:
                    break                                                      #大于总概率随机数，则选中该主题
                kChoosed += 1
            if kChoosed == self.K:                                             #代表选择了新的类
                self.K += 1
                self.K_current += 1
        else:
            bigPro = prob[0]
            for k in range(1, self.K + 1):
                if prob[k] > bigPro:
                    bigPro = prob[k]
                    kChoosed = k
            if kChoosed == self.K:
                self.K += 1
                self.K_current += 1
        return kChoosed

    # Clear the useless cluster
    def checkEmpty(self, cluster):
        if cluster in self.n_z and self.m_z[cluster] == 0:
            self.K_current -= 1
            self.m_z.pop(cluster)
            if cluster in self.n_z:
                self.n_z.pop(cluster)
                self.n_zv.pop(cluster)

    def output(self, documentSet, outputPath, wordList, batchNum):
        outputDir = outputPath + self.dataset + self.ParametersStr + "newBatch" + str(batchNum) + "/"
        try:
            isExists = os.path.exists(outputDir)
            if not isExists:
                os.mkdir(outputDir)
                print("\tCreate directory:", outputDir)
        except:
            print("ERROR: Failed to create directory:", outputDir)
        self.outputClusteringResult(outputDir, documentSet)
        self.estimatePosterior()
        try:
            self.outputPhiWordsInTopics(outputDir, wordList, self.wordsInTopicNum)
        except:
            print("\tOutput Phi Words Wrong!")
        self.outputSizeOfEachCluster(outputDir, documentSet)

    def estimatePosterior(self):
        self.phi_zv = {}
        for cluster in self.n_zv:
            n_z_sum = 0
            if self.m_z[cluster] != 0:
                if cluster not in self.phi_zv:
                    self.phi_zv[cluster] = {}
                for v in self.n_zv[cluster]:
                    if self.n_zv[cluster][v] != 0:
                        n_z_sum += self.n_zv[cluster][v]
                for v in self.n_zv[cluster]:
                    if self.n_zv[cluster][v] != 0:
                        self.phi_zv[cluster][v] = float(self.n_zv[cluster][v] + self.beta) / float(n_z_sum + self.beta0)

    def getTop(self, array, rankList, Cnt):
        index = 0
        m = 0
        while m < Cnt and m < len(array):
            max = 0
            for no in array:
                if (array[no] > max and no not in rankList):
                    index = no
                    max = array[no]
            rankList.append(index)
            m += 1

    def outputPhiWordsInTopics(self, outputDir, wordList, Cnt):
        outputfiledir = outputDir + str(self.dataset) + "SampleNo" + str(self.sampleNo) + "PhiWordsInTopics.txt"
        writer = open(outputfiledir, 'w')
        for k in range(self.K):
            rankList = []
            if k not in self.phi_zv:
                continue
            topicline = "Topic " + str(k) + ":\n"
            writer.write(topicline)
            self.getTop(self.phi_zv[k], rankList, Cnt)
            for i in range(rankList.__len__()):
                tmp = "\t" + wordList[rankList[i]] + "\t" + str(self.phi_zv[k][rankList[i]])
                writer.write(tmp + "\n")
        writer.close()

    def outputSizeOfEachCluster(self, outputDir, documentSet):
        outputfile = outputDir + str(self.dataset) + "SampleNo" + str(self.sampleNo) + "SizeOfEachCluster.txt"
        writer = open(outputfile, 'w')
        topicCountIntList = []
        for cluster in range(self.K):
            if cluster in self.m_z and self.m_z[cluster] != 0:
                topicCountIntList.append([cluster, self.m_z[cluster]])
        line = ""
        for i in range(topicCountIntList.__len__()):
            line += str(topicCountIntList[i][0]) + ":" + str(topicCountIntList[i][1]) + ",\t"
        writer.write(line + "\n\n")
        line = ""
        topicCountIntList.sort(key = lambda tc: tc[1], reverse = True)
        for i in range(topicCountIntList.__len__()):
            line += str(topicCountIntList[i][0]) + ":" + str(topicCountIntList[i][1]) + ",\t"
        writer.write(line + "\n")
        writer.close()

    def outputClusteringResult(self, outputDir, documentSet):
        outputPath = outputDir + str(self.dataset) + "SampleNo" + str(self.sampleNo) + "ClusteringResult" + ".txt"
        writer = open(outputPath, 'w')
        for d in range(self.startDoc, self.currentDoc):
            documentID = documentSet.documents[d].documentID
            cluster = self.z[documentID]
            writer.write(str(documentID) + " " + str(cluster) + "\n")
        writer.close()
