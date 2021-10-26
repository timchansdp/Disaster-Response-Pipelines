import numpy as np
from sklearn.metrics import f1_score

def multi_output_fscore(Y_true, Y_pred):

    fscore_list = []
    
    for i in range(0, Y_true.shape[1]):

        f_score = f1_score(y_true = Y_true.iloc[:, i],\
                           y_pred = Y_pred[:, i],\
                           average = 'weighted',\
                           zero_division = 0)
                            
        fscore_list.append(f_score)

    fscore_list = np.array(fscore_list)

    return fscore_list.mean()