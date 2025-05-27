import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import math
from torch.autograd import Variable
import torch.nn.init as init
import numpy as np
import pdb

EPS = 1e-7
class BinMaskMeter:
    def __init__(self) -> None:

        self.precs = np.array([])
        self.recs = np.array([])
        self.f1s = np.array([])

        self.precs_sum = 0
        self.recs_sum = 0
        self.f1s_sum = 0
        self.count = 0
    
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

            self.precs_sum += precision.item()
            self.recs_sum += recall.item()
            self.f1s_sum += f1.item()
            self.count += 1

    def get_metrics(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.precs_sum, self.recs_sum, self.f1s_sum, self.count], dtype=torch.float32, device=device)
        if (not dist.is_available()) or (not dist.is_initialized()):
            # Either torch.distributed is not compiled in, or the process group
            # has not been initialised
            pass
        else:
            dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.precs_sum, self.recs_sum, self.f1s_sum, self.count = total.tolist()
        
        return (self.f1s_sum/self.count), (self.recs_sum/self.count), (self.precs_sum/self.count)
