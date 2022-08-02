from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
from lib.utils import properties as p

def get_confused(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[i for i in range(len(p['bins']()))])
    TP = np.diag(cm)
    FP = np.sum(cm, axis=0) - TP
    FN = np.sum(cm, axis=1) - TP
    TN = []
    for i in range(len(cm)):
        temp = np.delete(cm, i, 0)    # delete ith row
        temp = np.delete(temp, i, 1)  # delete ith column
        TN.append(sum(sum(temp)))
    return TP, TN, FP, FN

def equal_opportunity_difference(y_true_p, y_pred_p, y_true_u, y_pred_u):
    TP_p, TN_p, FP_p, FN_p = get_confused(y_true_p, y_pred_p)
    TP_u, TN_u, FP_u, FN_u = get_confused(y_true_u, y_pred_u)
    rec_p = TP_p / (TP_p + FN_p)
    rec_u = TP_u / (TP_u + FN_u)
    return rec_u - rec_p

def equal_odd_difference(y_true_p, y_pred_p, y_true_u, y_pred_u):
    rec_diff = equal_opportunity_difference(y_true_p, y_pred_p, y_true_u, y_pred_u)
    TP_p, TN_p, FP_p, FN_p = get_confused(y_true_p, y_pred_p)
    TP_u, TN_u, FP_u, FN_u = get_confused(y_true_u, y_pred_u)
    fpr_u = FP_u / (FP_u + TN_u)
    fpr_p = FP_p / (FP_p + TN_p)
    return (rec_diff + fpr_u - fpr_p) / 2 

def get_selection_rate(y_true, y_pred):
    return get_accuracy(y_true, y_pred)

def statistical_parity_difference(y_true_p, y_pred_p, y_true_u, y_pred_u):
    try:
        return get_selection_rate(y_true_u, y_pred_u) - get_selection_rate(y_true_p, y_pred_p)
    except:
        return np.nan
    
def disparate_impact_ratio(y_true_p, y_pred_p, y_true_u, y_pred_u):
    try:
        return get_selection_rate(y_true_u, y_pred_u) / get_selection_rate(y_true_p, y_pred_p)
    except:
        return np.nan
    
def get_metrics_data(metrics):
    metrics_df = pd.DataFrame()
    for m in metrics:
        row = pd.Series(m)
        metrics_df = metrics_df.append(row, ignore_index=True)
    return metrics_df

def get_accuracy(y_true, y_pred):
    cv = [y_t == y_p for (y_t, y_p) in zip(y_true, y_pred)]
    return np.nan if len(cv) == 0 else sum(cv) / len(cv)