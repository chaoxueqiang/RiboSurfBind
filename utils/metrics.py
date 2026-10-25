import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                            f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score)

def calculate_metrics(labels, probabilities, threshold=.5):
    labels = np.asarray(labels,dtype=np.int64)
    probabilities = np.asarray(probabilities,dtype=np.float64)
    predictions = (probabilities > threshold).astype(np.int64)
    tn,fp,fn,tp = confusion_matrix(labels,predictions,labels=[0,1]).ravel()
    return dict(auc=float(roc_auc_score(labels,probabilities)) if len(np.unique(labels))==2 else None,
                aupr=float(average_precision_score(labels,probabilities)) if np.any(labels==1) else None,
                precision=float(precision_score(labels,predictions,zero_division=0)),
                recall=float(recall_score(labels,predictions,zero_division=0)),
                f1=float(f1_score(labels,predictions,zero_division=0)),
                mcc=float(matthews_corrcoef(labels,predictions)),
                accuracy=float(accuracy_score(labels,predictions)),
                tp=int(tp),tn=int(tn),fp=int(fp),fn=int(fn),threshold=float(threshold))
