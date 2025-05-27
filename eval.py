import os
import torch
import numpy as np
import torch.nn as nn
from models.get_model import get_model
import pandas as pd
from progress.bar import Bar as Bar

from dataset.hls_opera import HLSData
import matplotlib.pyplot as plt

from loguru import logger as lgr
from tqdm import tqdm
import argparse




parser = argparse.ArgumentParser(description='Baselines (SegNet)')
parser.add_argument('--data_dir', default='/scratch3/workspace/rdaroya_umass_edu-water/hls_data/multitask_data_opera/', type=str, help='Path to dataset')
parser.add_argument('--batch_size', default=12, type=int, help='Batch size')
parser.add_argument('--is_distrib', default=True, type=int, help='Batch size')
parser.add_argument('--tasks', default=["water_mask", "cloudshadow_mask", "cloud_mask", "snowice_mask", "sun_mask"], nargs='+', help='Task(s) to be trained')

parser.add_argument('--ckpt_path', default=None, type=str, help='specify location of checkpoint')
parser.add_argument('--weight', default='uniform', type=str, help='multi-task weighting: uniform, gradnorm, mgda, uncert, dwa, gs')
parser.add_argument('--backbone', default='mobilenetv3', type=str, help='shared backbone')
parser.add_argument('--head', default='mobilenetv3_head', type=str, help='task-specific decoder')
parser.add_argument('--pretrained', default=False, type=int, help='using pretrained weight from ImageNet')


parser.add_argument('--method', default='vanilla', type=str, help='vanilla or mtan')
parser.add_argument('--out', default='./results/opera-mtl-baselines', help='Directory to output the result')


opt = parser.parse_args()
lgr.debug(f"opt: {opt}")

tasks = opt.tasks
num_inp_feats = 6   # number of channels in input
tasks_outputs_tmp = {
    "water_mask": 1,
    "cloudshadow_mask": 1,
    "cloud_mask": 1,
    "snowice_mask": 1,
    "sun_mask": 1,
}
tasks_outputs = {t: tasks_outputs_tmp[t] for t in tasks}
lgr.debug(f"opt: {opt.__dict__}")

model = get_model(opt, tasks_outputs=tasks_outputs, num_inp_feats=num_inp_feats).cuda()


# train_dataset = HLSData(root=opt.data_dir, split="train", augmentation=True, flip=True, normalize=False)
test_dataset1 = HLSData(root=opt.data_dir, split="test", normalize=False)
test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset1,
    batch_size=opt.batch_size,
    shuffle=False, num_workers=0)

lgr.debug(f"Loading weights from {opt.ckpt_path}")
checkpoint = torch.load(opt.ckpt_path, weights_only=False)
if opt.is_distrib:
    new_ckpt = {k.split("module.")[-1]:v for k,v in checkpoint["state_dict"].items()}
    checkpoint["state_dict"] = new_ckpt
tmp = model.load_state_dict(checkpoint["state_dict"], strict=True)
lgr.debug(f"After loading ckpt: {tmp}")
lgr.debug(f"Checkpoint epoch: {checkpoint['epoch']}. best_perf: {checkpoint['best_performance']}")

model.eval()


val_dataset1 = HLSData(root=opt.data_dir, split="val", normalize=False)
val_loader = torch.utils.data.DataLoader(
    dataset=val_dataset1,
    batch_size=opt.batch_size,
    shuffle=False, num_workers=0)
val_batch = len(val_loader)
val_dataset = iter(val_loader)

lgr.debug(f"Evaluating on {val_batch} val batches to find best threshold")
# metrics = {t: {"f1":[], "rec":[], "prec":[], "acc": []} for t in tasks}
# gtpreds = {t: {"gt":[], "pred":[]} for t in tasks}
rgbs = []
thresh_choices = np.arange(0,1,0.1)
thresh_metrics = {t: {th: {"TP":[], "FP":[], "FN":[], "TN":[], "num_px": []} for th in thresh_choices} for t in tasks}
counts_tps = {t: {"TP":[], "FP":[], "FN":[], "TN":[], "num_px": []} for t in tasks}    # number of pixels
with torch.no_grad():
    for thresh in thresh_choices:
        print(f"Evaluating on thresh={thresh}")
        val_dataset = iter(val_loader)
        for k in tqdm(range(val_batch)):
            val_data, val_labels = next(val_dataset)
            val_data = val_data.cuda()
            for task_name in tasks:
                val_labels[task_name] = val_labels[task_name].cuda()
            
            val_pred, feat = model(val_data, feat=True)
            for t in tasks:
                pred = torch.squeeze(val_pred[t], 1)
                target = torch.squeeze(val_labels[t], 1)
                thresh_pred = torch.where(pred > thresh, 1., 0.)

                TP = torch.sum(torch.round(torch.clip(target * thresh_pred, 0, 1)))
                FN = torch.sum(torch.round(torch.clip((1-target) * thresh_pred, 0, 1))) # target is 0, but pred is 1
                FP = torch.sum(torch.round(torch.clip(target * (1-thresh_pred), 0, 1))) # target is 1, but pred is 0
                TN = torch.sum(torch.round(torch.clip((1-target) * (1-thresh_pred), 0, 1))) # target is 0, and pred is 0
                
                # rec, prec, f1, acc = compute_mask_metrics(val_pred[t], val_labels[t], thresh=thresh)
                # lgr.debug(f"{t} rec: {rec}, prec: {prec}, f1: {f1}, acc: {acc},")
                thresh_metrics[t][thresh]["TP"].append(TP.item())
                thresh_metrics[t][thresh]["FN"].append(FN.item())
                thresh_metrics[t][thresh]["FP"].append(FP.item())
                thresh_metrics[t][thresh]["TN"].append(TN.item())
                s1,s2,s3 = pred.shape
                num_px = s1*s2*s3
                assert num_px == (TP+FN+FP+TN)
                thresh_metrics[t][thresh]["num_px"].append(num_px)
        #     if k==5:
        #         break
        # break

EPS = 1e-7
metric_names = ["f1", "rec", "prec", "acc", "iou"]
task_metric_per_thresh = {t: {m: [] for m in metric_names} for t in tasks}
for t in tasks:
    for th in thresh_choices:
        TP_tot = np.sum(np.array(thresh_metrics[t][th]["TP"]))
        FP_tot = np.sum(np.array(thresh_metrics[t][th]["FP"]))
        FN_tot = np.sum(np.array(thresh_metrics[t][th]["FN"]))
        TN_tot = np.sum(np.array(thresh_metrics[t][th]["TN"]))
        prec = TP_tot/(TP_tot + FP_tot + EPS)
        rec = TP_tot/(TP_tot + FN_tot + EPS)
        f1 = (2*prec*rec)/(prec+rec + EPS)
        acc = (TP_tot + TN_tot)/(TP_tot + TN_tot + FN_tot + FP_tot + EPS)
        iou = TP_tot/(TP_tot + FP_tot + FN_tot + EPS)
        task_metric_per_thresh[t]["f1"].append(f1)
        task_metric_per_thresh[t]["rec"].append(rec)
        task_metric_per_thresh[t]["prec"].append(prec)
        task_metric_per_thresh[t]["acc"].append(acc)
        task_metric_per_thresh[t]["iou"].append(iou)
# thresh_metrics
# task_metric_per_thresh

# Find optimal threshold given metrics (based on f1 score)
optim_threshes = {t:None for t in tasks}
metrics_df_data = {}
metrics_df_data["backbone"] = opt.backbone
metrics_df_data["head"] = opt.head
metrics_df_data["ckpt_path"] = opt.ckpt_path
metrics_df_data["pretrained"] = opt.pretrained
for t in tasks:
    optim_idx = np.argmax(task_metric_per_thresh[t]["f1"])
    optim_thresh = thresh_choices[optim_idx]
    optim_threshes[t] = optim_thresh
    print(f"{t} optim thresh: {optim_thresh} [f1: {task_metric_per_thresh[t]['f1'][optim_idx]}]")
    metrics_df_data[f"{t}_thresh"] = optim_thresh
lgr.debug(f"optim_threshes: {optim_threshes}")

lgr.debug(f"Using the following thresholds: {optim_threshes}")
model.eval()


# Evaluate on test set
test_batch = len(test_loader)
test_dataset = iter(test_loader)
lgr.debug(f"Evaluating on {test_batch} test batches")
metrics = {t: {"f1":[], "rec":[], "prec":[], "acc": []} for t in tasks}
counts_tps = {t: {"TP":[], "FP":[], "FN":[], "TN":[], "num_px": []} for t in tasks}    # number of pixels
gtpreds = {t: {"gt":[], "pred":[]} for t in tasks}
rgbs = []
with torch.no_grad():
    for k in tqdm(range(test_batch)):
        test_data, test_labels = next(test_dataset)
        
        test_data = test_data.cuda()
        for task_name in tasks:
            test_labels[task_name] = test_labels[task_name].cuda()
        
        test_pred, feat = model(test_data, feat=True)
        for t in tasks:
            pred = torch.squeeze(test_pred[t], 1)
            target = torch.squeeze(test_labels[t], 1)
            thresh_pred = torch.where(pred > optim_threshes[t], 1., 0.)

            TP = torch.sum(torch.round(torch.clip(target * thresh_pred, 0, 1)))
            FP = torch.sum(torch.round(torch.clip((1-target) * thresh_pred, 0, 1))) # target is 0, but pred is 1 (false positive)
            FN = torch.sum(torch.round(torch.clip(target * (1-thresh_pred), 0, 1))) # target is 1, but pred is 0 (false negative)
            TN = torch.sum(torch.round(torch.clip((1-target) * (1-thresh_pred), 0, 1))) # target is 0, and pred is 0
            
            counts_tps[t]["TP"].append(TP.item())
            counts_tps[t]["FP"].append(FP.item())
            counts_tps[t]["FN"].append(FN.item())
            counts_tps[t]["TN"].append(TN.item())
            s1,s2,s3 = pred.shape
            num_px = s1*s2*s3
            assert num_px == (TP+FN+FP+TN)
            counts_tps[t]["num_px"].append(num_px)


            if k<10:
                test_img = test_labels[t][0,:,:].detach().cpu().numpy()
                gtpreds[t]["gt"].append(test_img)
                pred_img = test_pred[t][0,:,:].detach().cpu().numpy()   
                # pred_img = np.where(pred_img > optim_threshes[t], 1, 0)     # NOTE: Added to show thresholded image
                gtpreds[t]["pred"].append(pred_img)
        if k<10:    # for visualization of sample outputs
            samp = test_data.detach().cpu().numpy()
            bgr = samp[0,:3, :, :]
            rgb = np.transpose(bgr, (1,2,0))[:,:,::-1]
            rgbs.append(rgb)

# Summarize metrics for each teask
EPS = 1e-7  # for max pool loss
metric_names = ["f1", "rec", "prec", "acc","miou"]
print("Summarized metrics for all tasks:")
print("task," + ",".join(metric_names))
for t in tasks:
    print(t, end=",")
    TP_tot = np.sum(np.array(counts_tps[t]["TP"]))
    FP_tot = np.sum(np.array(counts_tps[t]["FP"]))
    FN_tot = np.sum(np.array(counts_tps[t]["FN"]))
    TN_tot = np.sum(np.array(counts_tps[t]["TN"]))
    prec = TP_tot/(TP_tot + FP_tot + EPS)
    rec = TP_tot/(TP_tot + FN_tot + EPS)
    f1 = (2*prec*rec)/(prec+rec + EPS)
    acc = (TP_tot + TN_tot)/(TP_tot + TN_tot + FN_tot + FP_tot + EPS)
    miou = TP_tot/(TP_tot + FP_tot + FN_tot + EPS)
    print(f"{f1},{rec},{prec},{acc},{miou}")

    metrics_df_data[f"{t}_f1"] = f1
    metrics_df_data[f"{t}_rec"] = rec
    metrics_df_data[f"{t}_prec"] = prec
    metrics_df_data[f"{t}_acc"] = acc
    metrics_df_data[f"{t}_miou"] = miou

metrics_df = pd.DataFrame([metrics_df_data])
metrics_df.to_csv(f"{opt.ckpt_path.split('.pth')[0]}.csv", index=False)

ncols = len(tasks)*2 + 1 # 1 gt-pred pair for each task, and the rgb
nrows = 6
fig, ax = plt.subplots(nrows, ncols, figsize=(2*ncols, 2.25*nrows))

for r_ctr in range(nrows):
    c_ctr = 0
    ax[r_ctr, c_ctr].imshow(rgbs[r_ctr])
    ax[r_ctr, c_ctr].get_xaxis().set_ticks([])
    ax[r_ctr, c_ctr].get_yaxis().set_ticks([])
    ax[r_ctr, c_ctr].set_title(f"RGB")
    c_ctr += 1

    for t_ctr, t in enumerate(tasks):
        for out_ctr, out in enumerate(["gt", "pred"]):
            img = np.squeeze(gtpreds[t][out][r_ctr])
            col_ctr = c_ctr + (t_ctr*2) + out_ctr
            ax[r_ctr, col_ctr].imshow(img, vmin=0.0, vmax=1.0)
            ax[r_ctr, col_ctr].get_xaxis().set_ticks([])
            ax[r_ctr, col_ctr].get_yaxis().set_ticks([])
            ax[r_ctr, col_ctr].set_title(f"{out} {t.replace('_mask','')}")



plt.tight_layout()
plt.subplots_adjust(wspace=0, hspace=0.3)
plt.savefig(f"{opt.ckpt_path.split('.pth')[0]}.png", bbox_inches="tight")
plt.show()