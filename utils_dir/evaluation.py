import torch
import numpy as np


EPS = 1e-7
class BinMaskMeter:
    def __init__(self) -> None:

        self.precs = np.array([])
        self.recs = np.array([])
        self.f1s = np.array([])
    
    def update(self, pred, target):
        pred, target = pred.squeeze(), target.squeeze()
        with torch.no_grad():
            TP = torch.sum(torch.round(torch.clip(target * pred, 0, 1)))
            TP_FP = torch.sum(torch.round(torch.clip(pred, 0, 1)))
            TP_FN = torch.sum(torch.round(torch.clip(target, 0, 1)))
            recall = TP / (TP_FN + EPS)
            precision = TP / (TP_FP + EPS)
            f1 = 2 * ((precision * recall) / (precision + recall + EPS))

            self.precs = np.append(self.precs, np.array([precision.item()]))
            self.recs = np.append(self.recs, np.array([recall.item()]))
            self.f1s = np.append(self.f1s, np.array([f1.item()]))

    def get_metrics(self):

        return np.mean(self.f1s), np.mean(self.recs), np.mean(self.precs)
