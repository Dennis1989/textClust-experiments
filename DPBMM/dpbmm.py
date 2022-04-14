from sklearn.datasets import fetch_20newsgroups
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from collections import defaultdict
import gensim
#from InitKmeans import InitKmeans
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
     
def pre_processData(newsgroups_train):
    tokenizer = RegexpTokenizer(r'\w+')
    for i in range(len(newsgroups_train)):
        newsgroups_train[i] = newsgroups_train[i].lower()
        #newsgroups_train[i] = tokenizer.tokenize(newsgroups_train[i])
    #newsgroups_train = [[token for token in doc if not token.isdigit()] for doc in newsgroups_train]
    newsgroups_train = [doc.split(' ') for doc in newsgroups_train]
    return newsgroups_train
from sklearn.datasets import fetch_20newsgroups
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from collections import defaultdict
import gensim
#from InitKmeans import InitKmeans
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
import sys  
import json
#from txt_process_util import construct_biterms
import re
#dataset = "../datasets/Tweets-T-biterm"

def run(dataset, listOfObjects):

    preds = []
    trues = []
    newsgroups_train = list()
    news_labels = {}
    docID_username = {}
    
    i = 0
    file_path=dataset
    with open(file_path) as fp:
        lines = fp.read().split("\n")
        for line in lines:
            if line:
                #text = json.loads(line)["bitermText"].strip()
                wordList = re.sub("[^\w]", " ", json.loads(line)["textCleaned"]).split()
                tx= ""
                for k in range(0,len(wordList)):
                    for j in range(k+1,len(wordList)):
                        tx = tx+wordList[k]+","+wordList[j]+" "
                tx = tx[:-1]
                label = json.loads(line)["clusterNo"]
                newsgroups_train.append(tx)
                news_labels[i] = label
                i+=1
    fp.close()
    
    corpus = pre_processData(newsgroups_train)

    doc_biterm = {}
    for docID in range(len(corpus)):
        words = corpus[docID]
        duplicate_biterm = []
        doc_biterm[docID] = {}
        for biterm in words:
            hash_biterm = set(biterm.split(','))
            if hash_biterm not in duplicate_biterm:
                duplicate_biterm.append(hash_biterm)
                doc_biterm[docID][biterm] = 1
            else:
                if biterm in doc_biterm[docID].keys():
                    doc_biterm[docID][biterm] += 1
                else:
                    doc_biterm[docID][biterm] = 1
    doc_words = {}
    i_biterm = 0
    for docID in range(len(corpus)):
        doc_words[docID] = {}
        words = corpus[docID]
        for biterm in words:
            for word in biterm.split(','):
                if word not in doc_words[docID].keys():
                    doc_words[docID][word] = 0
                doc_words[docID][word] += 1
    

    nmi_batch = {}
    topic_batch = {}
    start = 0
    end = 0
    total_batch = None
    if len(corpus) % 1000 == 0:
        total_batch = int(len(corpus) / 1000)
    else:
        total_batch = int(len(corpus) / 1000) +1

    for batch in range(total_batch):
        docID_assign_z = {}
        m_z = {}
        n_z = {}
        n_w = {}
        n_b = {}
        Topics = []
        V = set()
        D = set()
        alpha = 0.3
        beta = 0.2
        end = 1000 * (batch + 1)
        if end > len(corpus):
            end = len(corpus)
        nmi_batch[batch] = 0
        topic_batch[batch] = 0
        
        total_iter = 1
        for iter in range(total_iter):
            for docID in range(start,end):
                words = corpus[docID]
                D.discard(docID)
                if docID in docID_assign_z.keys():
                    before_k = docID_assign_z[docID]
                    m_z[before_k].discard(docID)
                    for biterm in words:
                        for word in biterm.split(','):
                            n_z[before_k][word] -= 1
                            n_w[before_k] -=1
                else:
                    before_k = -1
                if len(D) == 0 and len(V) == 0:
                    choose_k = 0
                    D.add(docID)
                    docID_assign_z[docID] = choose_k
                    if choose_k not in m_z.keys():
                        m_z[choose_k] = set()
                    m_z[choose_k].add(docID)
                    for biterm in words:
                        for word in biterm.split(','):
                            if choose_k not in n_w.keys():
                                n_w[choose_k] = 0
                            if choose_k not in n_z.keys():
                                n_z[choose_k] = {}
                            if word not in n_z[choose_k].keys():
                                n_z[choose_k][word] = 0
                            n_z[choose_k][word] += 1
                            n_w[choose_k] += 1
                            V.add(word)
                    if choose_k not in Topics:
                        Topics.append(choose_k)
                else:
                    log_pro = []
                    for k in Topics:
                        pro_k = len(m_z[k])
                        if pro_k != 0:
                            i = 0
                            for biterm in words:
                                each_word = biterm.strip().split(',')
                                for word in each_word:
                                    if word not in n_z[k].keys():
                                        n_z[k][word] = 0
                                for j in range(doc_biterm[docID][biterm]):
                                    pro_k *= (n_z[k][each_word[0]] + n_z[k][each_word[-1]] + beta + j) / ( n_w[k] + len(V)*beta + i) 
                                    i += 1

                        if pro_k == 0:
                            pro_k = sys.float_info.min
                        log_pro.append(pro_k)

                    pro_new_k = alpha*(len(D))
                    i = 0
                    for biterm in words:
                        for j in range(doc_biterm[docID][biterm]):
                            pro_new_k *= ( beta + j) / ( len(V)*beta + i) 
                            i += 1

                    if pro_new_k == 0:
                        pro_new_k = sys.float_info.min        
                    log_pro.append(pro_new_k)

                    sum_pro=sum(log_pro)

                    normalized_posterior = [i/sum_pro for i in log_pro]    
                    select_k = None
                    if iter == (total_iter - 1):
                        select_k = normalized_posterior.index(max(normalized_posterior))

                    else:
                        select_k = np.random.choice( (len(Topics)+1) , 1, p=normalized_posterior)[0]  

                    if select_k == len(Topics):
                        choose_k = np.max(Topics) + 1
                    else:
                        choose_k = Topics[select_k]

                    D.add(docID)
                    docID_assign_z[docID] = choose_k
                    if choose_k not in m_z.keys():
                        m_z[choose_k] = set()
                    m_z[choose_k].add(docID)
                    for biterm in words:
                        for word in biterm.split(','):
                            if choose_k not in n_w.keys():
                                n_w[choose_k] = 0
                            if choose_k not in n_z.keys():
                                n_z[choose_k] = {}
                            if word not in n_z[choose_k].keys():
                                n_z[choose_k][word] = 0
                            n_z[choose_k][word] += 1
                            n_w[choose_k] += 1
                            V.add(word)
                    if choose_k not in Topics:
                        Topics.append(choose_k)

                count_k = []
                for k in Topics:
                    if k in m_z.keys() and len(m_z[k]) == 0:
                        m_z.pop(k, None)
                        n_z.pop(k, None)
                        n_w.pop(k, None)
                        count_k.append(k)
                for k in count_k:
                    Topics.remove(k)   
            if iter == 0:
                from sklearn.metrics.cluster import normalized_mutual_info_score
                nmi_sample = []
                nmi_result = []
                for key, value in news_labels.items():
                    if key < end and key >= start:
                        nmi_sample.append(value)
                        nmi_result.append(docID_assign_z[key])
                nmi_batch[batch] = normalized_mutual_info_score(np.array(nmi_sample), np.array(nmi_result))
                topic_batch[batch] = len(Topics)
                print("start ",start, " end ",end, "topics: ",len(Topics),"truth: ",len(np.unique(nmi_sample)), " NMI: ",normalized_mutual_info_score(np.array(nmi_sample), np.array(nmi_result)))
                trues.extend(nmi_sample)
                preds.extend(nmi_result)
                #print(preds)
                #print(trues)
        start = end 
    
    print(nmi_batch.keys())
    print(nmi_batch.values())
    print(len(trues))
    return [trues,preds]
#run("../datasets/Tweets-T-biterm.txt", "b")