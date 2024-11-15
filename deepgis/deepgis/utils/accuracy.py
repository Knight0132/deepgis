import torch
import numpy as np
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score

def accuracy(output, label, topk=(1,)):
   """
   Calculate top-k accuracy for model predictions.
   
   Args:
       output: Model output logits
       label: Ground truth labels
       topk: Tuple of k values for which to compute accuracy
       
   Returns:
       List of accuracy values for each k
   """
   maxk = max(topk)
   batch_size = output.size(0)

   # Get top k predictions
   _, pred = torch.topk(output, k=maxk, dim=1, largest=True, sorted=True)
   pred = pred.T
   
   # Compare predictions with expanded labels
   correct = torch.eq(pred, label.contiguous().view(1,-1).expand_as(pred))
   
   res = []
   for k in topk:
       correct_k = correct[:k].contiguous().view(-1).float().sum(dim=0, keepdim=True)
       res.append(correct_k*100/batch_size)
   return res

def output_metrics(prediction, label):
   """
   Calculate various classification metrics.
   
   Args:
       prediction: Model predictions
       label: Ground truth labels
       
   Returns:
       Tuple of:
           confusion_matrix: 2D numpy array of confusion matrix
           weighted_recall: Weighted recall score
           weighted_precision: Weighted precision score  
           weighted_f1: Weighted F1 score
   """
   CM = confusion_matrix(label, prediction)
   weighted_recall = recall_score(label, prediction, average="weighted")
   weighted_precision = precision_score(label, prediction, average="weighted")
   weighted_f1 = f1_score(label, prediction, average="weighted")
   
   return CM, weighted_recall, weighted_precision, weighted_f1