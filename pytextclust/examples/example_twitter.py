from textClustPy import textclust
from textClustPy import Preprocessor
from textClustPy import TwitterInput

def clust_callback(textclust):
        #textclust.showclusters(5, 10, "micro")
       # print(list(textclust["microclusters"].values())[0].textids)
        #print("yeah")
        return

def save_callback(id, time, text, obj):
        print(id)
        #print(time)
        #print(text)
        #print(obj)
        return

## create textclust instance
clust = textclust(callback = clust_callback,radius=0.5,_lambda=0.01,tgap=10, auto_r= True, model="skipgram", embedding_verification= True, macro_distance="embedding_cosine_distance", verbose=False)
preprocessor = Preprocessor(max_grams=2, exclude_tokens=["hi","hello"])

## create input
TwitterInput("2viPgjwKKOiXKAcPZtVJyS8al", "2G8x8OxoeJ0SKWv5VqyZDsMrUBvwSyrpaBKW415QeCv6HySZ38", 
"3065672651-XW6RAW6XFTv5at5e40KLQRlGAUqEw0EFUS8V6OF", "QIDUMGv03OmasTY9uG4TQPSc3uuIrh2mPXrYaHu0ETkSC",  
["hi"], ["en"], textclust=clust, preprocessor=preprocessor,callback=save_callback)

