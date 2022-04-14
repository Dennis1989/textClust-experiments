import fastskip
import numpy as np
import logging
from gensim import utils, matutils
from gensim.corpora.dictionary import Dictionary
from pyemd import emd
## this is a wrapper for the yskip model
maxf = np.inf

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter(
        '%(name)-12s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


class inkrementalskip:

    def __init__(self):
        self.random = fastskip.random()
        logger.info("init incremental model")
        self.model = fastskip.skipgram()
        logger.info("initialized")

    ## train the model
    def train(self, text):
        self.model.train_tokenized(text, self.random)
        return
    
    ## custom implementation of word mover distance for incremental models
    def wmdistance(self, ws1, ws2):

        l1 = len(ws1)
        l2 = len(ws2)

        ## get all word embeddings
        vectors = self.model.getvecFast(ws1 + ws2)
        
        v1 = vectors[0:l1]
        v2 = vectors[l1:l1+l2]

        if len(v1)== 0 or len(v2)==0 or not v1 or not v2:
            return maxf

        ## here we delete empty sublists (those which do not have any embedding so far)
        v1_new = {}
        ws1_new =[]
        for i in range(0,len(v1)):
            if v1[i]:
                v1_new[ws1[i]]=v1[i]
                ws1_new.append(ws1[i])

        v2_new = {}
        ws2_new =[]
        for i in range(0,len(v2)):
            if v2[i]:
                v2_new[ws2[i]]=v2[i]
                ws2_new.append(ws2[i])

    
        if len(v1_new)== 0 or len(v2_new)==0 or not v1_new or not v2_new:
            return maxf
        

        document1 = ws1_new
        document2 = ws2_new


        dictionary = Dictionary(documents=[document1, document2])
        vocab_len = len(dictionary)

        if vocab_len == 1:
            # Both documents are composed by a single unique token
            return 0.0

        # Sets for faster look-up.
        docset1 = set(document1)
        docset2 = set(document2)

        # Compute distance matrix.
        distance_matrix = np.zeros((vocab_len, vocab_len), dtype=np.double)
        for i, t1 in dictionary.items():
            if t1 not in docset1:
                continue

            for j, t2 in dictionary.items():
                if t2 not in docset2 or distance_matrix[i, j] != 0.0:
                    continue

                # Compute Euclidean distance between unit-normed word vectors.
                distance_matrix[i, j] = distance_matrix[j, i] = np.sqrt(
                    np.sum((np.asarray(v1_new[t1]) - np.asarray(v2_new[t2]))**2))

        if np.sum(distance_matrix) == 0.0:
            # `emd` gets stuck if the distance matrix contains only zeros.
            #print("only zeroes")
            return maxf

        def nbow(document):
            d = np.zeros(vocab_len, dtype=np.double)
            nbow = dictionary.doc2bow(document)  # Word frequencies.
            doc_len = len(document)
            for idx, freq in nbow:
                d[idx] = freq / float(doc_len)  # Normalized word frequencies.
            return d

        # Compute nBOW representation of documents.
        d1 = nbow(document1)
        d2 = nbow(document2)

        # Compute WMD.
        return emd(d1, d2, distance_matrix)


    ## custom implementation of cosine distance for incremental skip model
    def n_similarity(self, ws1, ws2):
        """
        Compute cosine similarity between two sets of words.
        """
        #print("incremental smililarity")
        v1 = self.model.getvecFast(ws1)

        ## here we delete empty sublists (those which do not have any embedding so far)
        v1 = [x for x in v1 if x]
        
        ## we do not have to care for empty sublists. Each word is definitely in micro cluster
        v2 = self.model.getvecFast(ws2)
        v2 = [x for x in v2 if x]
        
        if len(v1)== 0 or len(v2)==0 or not any(v1) or not any(v2):
            return 0
        else:
            return np.dot(matutils.unitvec(np.array(v1).mean(axis=0)), matutils.unitvec(np.array(v2).mean(axis=0)))
        

    def n_similarity_weighted(self, ws1, ws2, weights1, weights2):
        
        l1 = len(ws1)
        l2 = len(ws2)
        
        ## get all word embeddings
        vectors = np.array(self.model.getvecFast(ws1+ws2))
        ## directly normalize vectors
        #normalized = vectors/np.linalg.norm(vectors,axis=1,keepdims=True)
        #normalized= np.array(vectors)
        ## split normalized apart
        v1 = vectors[0:l1]
        v2 = vectors[l1:l1+l2]

        if len(v1)== 0 or len(v2)==0 or not v1.any() or not v2.any():
            return 0
        ## here we delete empty sublists (those which do not have any embedding so far)
        
       # v1, weights1 = [[v1[x],weights1[x]] for x in len(v1) if v1[x]]

        weights1 = np.array([weights1[x] for x in range(len(v1)) if len(v1[x])>0])
        v1 = np.array([x for x in v1 if len(x)>0])
        #print(v1)
        
        ## we do not have to care for empty sublists. Each word is definitely in micro cluster
        #v2 = self.model.getvecFast(ws2)
        weights2 = np.array([weights2[x] for x in range(len(v2)) if len(v2[x])>0])
        v2 = np.array([x for x in v2 if len(x)>0])
        
      

        #v1 = np.array([word * weights1[index] for index, word in enumerate(v1)])
        #v2 = np.array([word * weights2[index] for index, word in enumerate(v2)])

        if len(v1)== 0 or len(v2)==0 or not v1.any() or not v2.any():
            return 0
        # here cosine similarity
        else:
            v1 = np.dot(v1.T,weights1)/sum(weights1)
            v2 = np.dot(v2.T,weights2)/sum(weights2)

           # l1 = v1.size
           # l2 = v2.size

           # together = np.vstack((v1,v2))
           # normalized = together/np.linalg.norm(together,axis=1,keepdims=True)
            #v1 = normalized[0:l1]
            #v2 = normalized[l1:l1+l2]
            #np.dot(matutils.unitvec(v1.mean()), matutils.unitvec(v2.mean()))
            erg = np.dot(matutils.unitvec(v1), matutils.unitvec(v2))
            #test = 1-np.arccos(erg) / np.pi
            return erg
