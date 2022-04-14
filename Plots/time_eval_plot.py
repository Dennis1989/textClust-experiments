import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np

folders  = ["News-T","Tweets-T","NT","NTS","SO-T","Trends-T",]
fig, axs = plt.subplots(2, 3, figsize=(9,7), sharey="row")
counter = 0
for folder in folders:

    files =["textclust-unigram","textclust-bigram","OSDM","EStream","DP-BMM","MSTREAM","MSTREAM-RAKIB", "DCSS"]
    names = []
    
    for file in files:
        split = file.split("_")
        
        algo = split[0]
        names.append(split[0])


        dat = pd.read_csv("../Evaluation Results/"+file+"_"+folder+""+"_time_eval.csv",delimiter=",",header=None) 


        dat_list = dat.values.tolist()
        x = np.arange(1000,1000*(len(dat_list[0])+1),1000)
        print(x)
        #data = [[0.701780229672209, 0.7051190583661806, 0.7163770029922119, 0.7324372395419088, 0.754588943779573, 0.785436310903416, 0.8259504557330335, 0.8696979669023879, 0.8812285005892406, 0.8731063477278604, 0.0],[0.7302532325447045, 0.733684101814546, 0.7418716902170022, 0.7533245223513613, 0.7784150944707313, 0.80885168394075, 0.8478774696930986, 0.911099347491506, 0.9301520772164069, 0.911109034551603, 0.0],[0.643290375537147, 0.6452911038348306, 0.653894269392499, 0.6674476617246483, 0.6883783186252191, 0.7243670891147818, 0.7810586390645459, 0.8496459511784551, 0.8812455770596225, 0.8700136271006359, 0.0]]

        
        axs[(int)(counter/3),counter%3].set_title(folder)
        #ax7.boxplot(data)
        print (algo)
        if algo == "textclust-unigram" or algo == "textclust-bigram":
            axs[(int)(counter/3),counter%3].plot(x,dat_list[3],"--+",linewidth=0.7)
        else:
            axs[(int)(counter/3),counter%3].plot(x,dat_list[3],"--+",linewidth=0.7, alpha = 0.2)
        #axs[counter].plot(x,dat_list[3],linewidth=0.5)
        axs[(int)(counter/3),counter%3].set_ylim([0, 1])
        #ax7.plot([1,2,3],[0.8856,0.9290,0.8620],color="blue")
        #ax7.set_xticks(x)
    
    counter = counter +1
    
axs[0,0].legend(names)
fig.tight_layout()   
#plt.show()
plt.savefig("time_eval.pdf",format="pdf",bbox_inches='tight')