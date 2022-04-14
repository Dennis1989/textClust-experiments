# textClust Experiments
This anonymous repository was crated for peer-review inspection. The conducted experiments are easily reproducible by running experiments.py. Also, the code of the textClust algorithm is located within the pytextclust subfolder.

Before the experiments can be conducted, a few steps have to be executed:

## 1. Create a virtual environment 
To avoid interference with your local python packages, we recommend to use a virtual environment for executing the experiments:
```
pip3 install virtualenv
virtualenv venv
source venv/bin/activate
```
## 2. Install important packages

```
pip3 install pyod mmh3 pyemd
```

## 3. Install word vectors
Some algorithms use word-vectors for cluster enhancement. We, therefore, require pretrained glove vectors:
```
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
```

## 4. Install and compile texClust locally
```
cd pytextclust
pip3 install .
```
Now everything is prepared and the experiments can be reproduced:

```
cd ..
python3 experiments.py
```

The results are created within the subfolder **Evaluation Results**. We already uploaded the final results in advance. However, they can be easily reproduced by re-running experiments.py.


