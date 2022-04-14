
from evaluation import Evaluate_old
from read_pred_true_text import ReadPredTrueText

resultFile = "result/News-T_ICF_CWW_ALPHA0.002_BETA0.0004_LAMDA6e-06.txt"

listtuple_pred_true_text=ReadPredTrueText(resultFile, ignoreMinusOne = False)

#print('result for', dataset)
Evaluate_old(listtuple_pred_true_text)  