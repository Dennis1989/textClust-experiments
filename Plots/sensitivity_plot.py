import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np



## load the results of the radius search
dat = pd.read_csv("../Evaluation Results/results_radius_on.csv")
dat2 = pd.read_csv("../Evaluation Results/results_radius_off.csv")

## load the results if the sigma search
dat_sensitivity = pd.read_csv("../Sensitivity Evaluation Results/textclust-unigram_5_mean_results_sigma_5.csv",delimiter=",") 

fig, (ax1,ax2) = plt.subplots(1,2,figsize=(7.5,3.2))



## here we plot the sigma search
folders  = ["News-T","Tweets-T","NT","NTS","SO-T","Trends-T"]
for dataset in folders:
    help = dat_sensitivity["dataset"]==dataset
    nmi = dat_sensitivity[help]["nmi"]
    ax1.plot([0.1,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5],nmi,"+--", label = dataset)

ax1.set_ylim(0,1)    
ax1.set_xticks([0.1,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5])
ax1.legend(fontsize=8,loc=4)

medianprops = dict(color='black')
## Here we plot the boxplot of the radius search
box = ax2.boxplot(dat, patch_artist=True, medianprops=medianprops)
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', "tab:brown"]
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)

ax2.set_ylim(0,1)    
ax2.plot([1,2,3,4,5,6],list(dat2.transpose().values)[0],'x',color="blue")
ax2.plot([1,2,3,4,5,6],list(dat2.transpose().values)[0],color="blue")
ax2.set_xticklabels(["Ns-T","Ts-T","NT","NTS","SO-T","Trends-T"])
#plt.show()
plt.savefig("auto_r_eval.pdf",format="pdf",bbox_inches='tight')


