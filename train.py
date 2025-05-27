# Adopted from https://github.com/pytorch/examples/blob/main/imagenet/main.py#L393 and https://github.com/VICO-UoE/UniversalRepresentations/tree/main/DensePred
# NOTE: this is for using multiple GPUs for training

import os
from enum import Enum
import torch
import numpy as np
import torch.optim as optim

import torch.multiprocessing as mp
import torch.nn.parallel
import torch.utils.data.distributed
import torch.distributed as dist
import argparse
import shutil
from models.get_model import get_model
from utils_dir.evaluation_dist import BinMaskMeter
import numpy as np
from progress.bar import Bar as Bar
from utils_dir import Logger, AverageMeter, mkdir_p
from utils_dir.dense_losses import get_dense_tasks_losses

from dataset.hls_opera import HLSData

from loguru import logger as lgr
from datetime import datetime


DATE_STR = datetime.now().strftime("%Y%m%d-%H%M%S")

parser = argparse.ArgumentParser(description='Baselines (SegNet)')
parser.add_argument('--data_dir', default='/scratch3/workspace/rdaroya_umass_edu-water/hls_data/multitask_data_opera/', type=str, help='Path to dataset')
parser.add_argument('--loss_type', default='bce', type=str, help="Type of loss for training. choices: [adaptive_maxpool, bce]")
parser.add_argument('--epochs', default=50, type=int, help='Number of epochs')
parser.add_argument('--batch_size', default=8, type=int, help='Batch size')
parser.add_argument('--find_unused_param', default=1, type=int, help='Find unused param for distrib training')
parser.add_argument('--scheduler', default="none", type=str, help='Scheduler to use (if any) - choices: "none", "steplr"')
parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate to use')

parser.add_argument('--ckpt_path', default=None, type=str, help='specify location of checkpoint')

parser.add_argument('--weight', default='uniform', type=str, help='multi-task weighting: uniform, gradnorm, mgda, uncert, dwa, gs')
parser.add_argument('--backbone', default='mobilenetv3', type=str, help='shared backbone')
parser.add_argument('--head', default='mobilenetv3_head', type=str, help='task-specific decoder')
parser.add_argument('--tasks', default=["water_mask", "cloudshadow_mask", "cloud_mask", "snowice_mask", "sun_mask"], nargs='+', help='Task(s) to be trained')
parser.add_argument('--dilated', dest='dilated', action='store_true', help='Dilated')
parser.add_argument('--method', default='vanilla', type=str, help='vanilla or mtan')
parser.add_argument('--out', default='./results/opera-mtl-baselines', help='Directory to output the result')
parser.add_argument("--pretrained", default=1, type=int, help="Set to 1 to use pretrained model. 0 otherwise.")

# The following params are for multiple GPU training
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')



def save_checkpoint(state, is_best, opt, filename='checkpoint.pth.tar'):
    filepath = os.path.join(opt.out, '{}_{}_{}_mtl_baselines_{}_{}_'.format(DATE_STR, opt.backbone, opt.head, opt.method, opt.weight) + filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(opt.out, '{}_{}_{}_mtl_baselines_{}_{}_'.format(DATE_STR, opt.backbone, opt.head, opt.method, opt.weight) + 'model_best.pth.tar'))

# compute parameter space
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    opt = parser.parse_args()
    lgr.debug(f"opt: {opt}")

    stl_performance = {}

    stl_performance['segnet_segnet_head'] = {
                        'full': {'semantic': 40.5355, 'depth': 0.627602, 'normal': 24.284388},
    }

    #TODO: fix (copied from segnet)
    stl_performance['mobilenetv3_mobilenetv3_head'] = {
                        'full': {'semantic': 40.5355, 'depth': 0.627602, 'normal': 24.284388},
    }

    if not os.path.isdir(opt.out):
        mkdir_p(opt.out)

    if opt.gpu is not None:
        lgr.warning('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if opt.dist_url == "env://" and opt.world_size == -1:
        opt.world_size = int(os.environ["WORLD_SIZE"])

    opt.distributed = opt.world_size > 1 or opt.multiprocessing_distributed

    if torch.cuda.is_available():
        ngpus_per_node = torch.cuda.device_count()
    else:
        ngpus_per_node = 1
    if opt.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        opt.world_size = ngpus_per_node * opt.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, opt))
    else:
        # Simply call main_worker function
        main_worker(opt.gpu, ngpus_per_node, opt)
    


best_loss = 100000
def main_worker(gpu, ngpus_per_node, opt):
    global best_loss
    opt.gpu = gpu
    if opt.gpu is not None:
        print("Use GPU: {} for training".format(opt.gpu))
    if opt.distributed:
        if opt.dist_url == "env://" and opt.rank == -1:
            opt.rank = int(os.environ["RANK"])
        if opt.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            opt.rank = opt.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=opt.dist_backend, init_method=opt.dist_url,
                                world_size=opt.world_size, rank=opt.rank)

    # define model, optimiser and scheduler
    # tasks = ["water_mask", "cloudshadow_mask", "cloud_mask", "snowice_mask", "sun_mask"]
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
    model = get_model(opt, tasks_outputs=tasks_outputs, num_inp_feats=num_inp_feats, pretrained=(opt.pretrained==1))
    # Weights = Weight(tasks).cuda()
    params = []
    params += model.parameters()
    # params += [Weights.weights]


    if not torch.cuda.is_available() and not torch.backends.mps.is_available():
        print('using CPU, this will be slow')
    elif opt.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if torch.cuda.is_available():
            if opt.gpu is not None:
                torch.cuda.set_device(opt.gpu)
                model.cuda(opt.gpu)
                # When using a single GPU per process and per
                # DistributedDataParallel, we need to divide the batch size
                # ourselves based on the total number of GPUs of the current node.
                opt.batch_size = int(opt.batch_size / ngpus_per_node)
                opt.workers = int((opt.workers + ngpus_per_node - 1) / ngpus_per_node)
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[opt.gpu], find_unused_parameters=(opt.find_unused_param==1))
            else:
                model.cuda()
                # DistributedDataParallel will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=(opt.find_unused_param==1))
    elif opt.gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(opt.gpu)
        model = model.cuda(opt.gpu)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        model = model.to(device)
    else:
        model = torch.nn.DataParallel(model).cuda()

    if torch.cuda.is_available():
        if opt.gpu:
            device = torch.device('cuda:{}'.format(opt.gpu))
        else:
            device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    optimizer = optim.Adam(params, lr=opt.lr)
    scheduler = None
    if opt.scheduler == "steplr":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    lgr.debug(f"Using scheduler: {scheduler}")

    start_epoch = 0
    if opt.ckpt_path is not None:
        lgr.debug(f"Loading checkpoint: {opt.ckpt_path}")
        if opt.gpu is None:
            checkpoint = torch.load(opt.ckpt_path)
        elif torch.cuda.is_available():
            # Map model to be loaded to specified single gpu.
            loc = 'cuda:{}'.format(opt.gpu)
            checkpoint = torch.load(opt.ckpt_path, map_location=loc)
        start_epoch = checkpoint['epoch']
        tmp = model.load_state_dict(checkpoint['state_dict'])
        lgr.debug(f"model load state: {tmp}")
        optimizer.load_state_dict(checkpoint['optimizer'])
        # scheduler.load_state_dict(checkpoint['scheduler'])
        lgr.debug(f"Successfully loaded checkpoint. Starting from epoch: {start_epoch}")
    else:
        lgr.debug(f"No checkpoint loaded.")

    title = 'HLSFMask'
    logger = Logger(os.path.join(opt.out, '{}_{}_{}_mtl_baselines_{}_{}_log.txt'.format(DATE_STR, opt.backbone, opt.head, opt.method, opt.weight)), title=title)
    logger_names = ['Epoch', 'T.Lwm', 'T.wmF1', 'T.wmRec', 'T.wmPrec', 'T.Lcsm', 'T.csmF1', 'T.csmRec', 'T.csmPrec', 'T.Lcm', 'T.cmF1', 'T.cmRec', 'T.cmPrec', 'T.Lsim', 'T.simF1', 'T.simRec', 'T.simPrec', 'T.Lsun', 'T.sunF1', 'T.sunRec', 'T.sunPrec',
        'V.Lwm', 'V.wmF1', 'V.wmRec', 'V.wmPrec','V.Lcsm', 'V.csmF1', 'V.csmRec', 'V.csmPrec','V.Lcm', 'V.cmF1', 'V.cmRec', 'V.cmPrec', 'V.Lsim', 'V.simF1', 'V.simRec', 'V.simPrec', 'V.Lsun', 'V.sunF1', 'V.sunRec', 'V.sunPrec', 'Wwm', 'Wcsm', 'Wcm', 'Wsim', 'Wsun', 'opt.lr']
    logger.set_names(logger_names)

    lgr.debug(f"LOSS FORMAT: {logger_names}\n")

    # define dataset path
    train_dataset = HLSData(root=opt.data_dir, split="train", augmentation=True, flip=True, normalize=False)
    val_dataset = HLSData(root=opt.data_dir, split="val", normalize=False)
    # test_dataset = HLSData(root=opt.data_dir, split="test", normalize=False)
    lgr.debug(f"Found {len(train_dataset)} train_dataset and {len(val_dataset)} val_dataset")

    if opt.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, drop_last=True)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=True)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
        num_workers=opt.workers, pin_memory=True, sampler=train_sampler, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=opt.batch_size, shuffle=False,
        num_workers=opt.workers, pin_memory=True, sampler=val_sampler, drop_last=True)
    lgr.debug(f"Found {len(train_loader)} train_loader and {len(val_loader)} val_loader")

    # define parameters
    total_epoch = opt.epochs
    train_batch = len(train_loader)
    val_batch = len(val_loader)
    lgr.debug(f"train_batch={train_batch} val_batch={val_batch}")
    avg_cost = np.zeros([total_epoch, len(logger_names)-7], dtype=np.float32)   # 5 are epoch, Wwm, Wcsm, Wcm, Wsim, Wsun
    lambda_weight = np.zeros([len(tasks), total_epoch])
    
    isbest = False
    for epoch in range(start_epoch, total_epoch):
        cost = np.zeros(len(logger_names)-7, dtype=np.float32)

        bar = Bar('Training', max=train_batch)

        # iteration for all batches
        model.train()
        train_dataset = iter(train_loader)
        
        train_loss0 = AverageMeter('trainLoss0', ':.4e')
        train_loss1 = AverageMeter('trainLoss1', ':.4e')
        train_loss2 = AverageMeter('trainLoss2', ':.4e')
        train_loss3 = AverageMeter('trainLoss3', ':.4e')
        train_loss4 = AverageMeter('trainLoss4', ':.4e')
        wmask_train_met = BinMaskMeter()
        csmask_train_met = BinMaskMeter()
        cmask_train_met = BinMaskMeter()
        simask_train_met = BinMaskMeter()
        sunmask_train_met = BinMaskMeter()
        for k in range(train_batch):
            train_data, train_labels = next(train_dataset)
            train_data = train_data.to(device, non_blocking=True)
            for task_name in tasks:
                train_labels[task_name] = train_labels[task_name].to(device, non_blocking=True)
            train_pred, feat = model(train_data, feat=True)
            train_loss = get_dense_tasks_losses(train_pred, train_labels, tasks, returndict=False, loss_type=opt.loss_type)
            loss = 0
            norms = []
            w = torch.ones(len(tasks)).float().to(device, non_blocking=True)
            
            # uniform loss across tasks
            for i in range(len(tasks)):
                lambda_weight[i, epoch] = w[i].data
            loss = sum(w[i].data * train_loss[i] for i in range(len(tasks)))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if "water_mask" in opt.tasks:
                wmask_train_met.update(train_pred['water_mask'].to(device, non_blocking=True), train_labels['water_mask'].to(device, non_blocking=True))
            if "cloudshadow_mask" in opt.tasks:
                csmask_train_met.update(train_pred['cloudshadow_mask'].to(device, non_blocking=True), train_labels['cloudshadow_mask'].to(device, non_blocking=True))
            if "cloud_mask" in opt.tasks:
                cmask_train_met.update(train_pred['cloud_mask'].to(device, non_blocking=True), train_labels['cloud_mask'].to(device, non_blocking=True))
            if "snowice_mask" in opt.tasks:
                simask_train_met.update(train_pred['snowice_mask'].to(device, non_blocking=True), train_labels['snowice_mask'].to(device, non_blocking=True))
            if "sun_mask" in opt.tasks:
                sunmask_train_met.update(train_pred['sun_mask'].to(device, non_blocking=True), train_labels['sun_mask'].to(device, non_blocking=True))
            
            train_loss0.update(train_loss[0].item(), train_data.shape[0])
            cost[0] = train_loss[0].item()
            if len(opt.tasks)>1:
                train_loss1.update(train_loss[1].item(), train_data.shape[0])
                cost[4] = train_loss[1].item()
            if len(opt.tasks)>2:
                train_loss2.update(train_loss[2].item(), train_data.shape[0])
                cost[8] = train_loss[2].item()
            if len(opt.tasks)>3:
                train_loss3.update(train_loss[3].item(), train_data.shape[0])
                cost[12] = train_loss[3].item()
            if len(opt.tasks)>4:
                train_loss4.update(train_loss[4].item(), train_data.shape[0])
                cost[16] = train_loss[4].item()

            
            
            avg_cost[epoch, :20] += cost[:20] / train_batch
            bar.suffix  = '{} => ({batch}/{size}) | LossWm: {loss_wm:.4f}. | LossCsm: {loss_csm:.4f}. | LossCm: {loss_cm:.4f}. | LossSim: {loss_sim:.4f}. | Losssun: {loss_sun:.4f}. | Ws: {ws:.4f} | Wd: {wd:.4f}| Wn: {wn:.4f}'.format(
                        opt.weight,
                        batch=k + 1,
                        size=train_batch,
                        loss_wm=cost[0],
                        loss_csm=cost[4],
                        loss_cm=cost[8],
                        loss_sim=cost[12],
                        loss_sun=cost[16],
                        ws=w[0].data,
                        wd=w[1].data if len(w)>1 else 1,
                        wn=w[2].data if len(w)>2 else 1,
                        )
            bar.next()
        if scheduler is not None:
            scheduler.step()
        bar.finish()
        train_loss0.all_reduce()
        if len(tasks) > 1:
            train_loss1.all_reduce()
        if len(tasks) > 2:
            train_loss2.all_reduce()
        if len(tasks) > 3:
            train_loss3.all_reduce()
        if len(tasks) > 4:
            train_loss4.all_reduce()
        avg_cost[epoch, 0] = train_loss0.avg
        avg_cost[epoch, 4] = train_loss1.avg if len(tasks) > 1 else 0
        avg_cost[epoch, 8] = train_loss2.avg if len(tasks) > 2 else 0
        avg_cost[epoch, 12] = train_loss3.avg if len(tasks) > 3 else 0
        avg_cost[epoch, 16] = train_loss4.avg if len(tasks) > 4 else 0
        avg_cost[epoch, 1:4] = wmask_train_met.get_metrics() if "water_mask" in opt.tasks else 0
        avg_cost[epoch, 5:8] = csmask_train_met.get_metrics() if "cloudshadow_mask" in opt.tasks else 0
        avg_cost[epoch, 9:12] = cmask_train_met.get_metrics() if "cloud_mask" in opt.tasks else 0
        avg_cost[epoch, 13:16] = simask_train_met.get_metrics() if "snowice_mask" in opt.tasks else 0
        avg_cost[epoch, 17:20] = sunmask_train_met.get_metrics() if "sun_mask" in opt.tasks else 0

        # evaluating test data
        model.eval()
        wmask_met = BinMaskMeter()
        csmask_met = BinMaskMeter()
        cmask_met = BinMaskMeter()
        simask_met = BinMaskMeter()
        sunmask_met = BinMaskMeter()

        val_loss0 = AverageMeter('ValLoss0', ':.4e')
        val_loss1 = AverageMeter('ValLoss1', ':.4e')
        val_loss2 = AverageMeter('ValLoss2', ':.4e')
        val_loss3 = AverageMeter('ValLoss3', ':.4e')
        val_loss4 = AverageMeter('ValLoss4', ':.4e')
        
        with torch.no_grad():  # operations inside don't track history
            val_dataset = iter(val_loader)
            for k in range(val_batch):
                val_data, val_labels = next(val_dataset)
                val_data = val_data.to(device, non_blocking=True)
                for task_name in tasks:
                    val_labels[task_name] = val_labels[task_name].to(device, non_blocking=True)

                val_pred = model(val_data)
                val_loss = get_dense_tasks_losses(val_pred, val_labels, tasks, loss_type=opt.loss_type)

                if "water_mask" in opt.tasks:
                    wmask_met.update(val_pred['water_mask'].to(device, non_blocking=True), val_labels['water_mask'].to(device, non_blocking=True))
                if "cloudshadow_mask" in opt.tasks:
                    csmask_met.update(val_pred['cloudshadow_mask'].to(device, non_blocking=True), val_labels['cloudshadow_mask'].to(device, non_blocking=True))
                if "cloud_mask" in opt.tasks:
                    cmask_met.update(val_pred['cloud_mask'].to(device, non_blocking=True), val_labels['cloud_mask'].to(device, non_blocking=True))
                if "snowice_mask" in opt.tasks:
                    simask_met.update(val_pred['snowice_mask'].to(device, non_blocking=True), val_labels['snowice_mask'].to(device, non_blocking=True))
                if "sun_mask" in opt.tasks:
                    sunmask_met.update(val_pred['sun_mask'].to(device, non_blocking=True), val_labels['sun_mask'].to(device, non_blocking=True))
                
                val_loss0.update(val_loss[0].item(), val_data.shape[0])
                if len(opt.tasks)>1:
                    val_loss1.update(val_loss[1].item(), val_data.shape[0])
                if len(opt.tasks)>2:
                    val_loss2.update(val_loss[2].item(), val_data.shape[0])
                if len(opt.tasks)>3:
                    val_loss3.update(val_loss[3].item(), val_data.shape[0])
                if len(opt.tasks)>4:
                    val_loss4.update(val_loss[4].item(), val_data.shape[0])
            val_loss0.all_reduce()
            if len(tasks) > 1:
                val_loss1.all_reduce()
            if len(tasks) > 2: 
                val_loss2.all_reduce()
            if len(tasks) > 3:
                val_loss3.all_reduce()
            if len(tasks) > 4:
                val_loss4.all_reduce()
            avg_cost[epoch, 20] = val_loss0.avg
            avg_cost[epoch, 24] = val_loss1.avg if len(tasks) > 1 else 0
            avg_cost[epoch, 28] = val_loss2.avg if len(tasks) > 2 else 0
            avg_cost[epoch, 32] = val_loss3.avg if len(tasks) > 3 else 0
            avg_cost[epoch, 36] = val_loss4.avg if len(tasks) > 4 else 0
            avg_cost[epoch, 21:24] = wmask_met.get_metrics() if "water_mask" in opt.tasks else 0
            avg_cost[epoch, 25:28] = csmask_met.get_metrics() if "cloudshadow_mask" in opt.tasks else 0
            avg_cost[epoch, 29:32] = cmask_met.get_metrics() if "cloud_mask" in opt.tasks else 0
            avg_cost[epoch, 33:36] = simask_met.get_metrics() if "snowice_mask" in opt.tasks else 0
            avg_cost[epoch, 37:40] = sunmask_met.get_metrics() if "sun_mask" in opt.tasks else 0

        # ave_loss = (val_loss[0] + val_loss[1] + val_loss[2] + val_loss[3] + val_loss[4])/5.0
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        if len(tasks) == 1:
            total_val_loss = torch.tensor([val_loss[0]], dtype=torch.float32, device=device)
            dist.all_reduce(total_val_loss, dist.ReduceOp.SUM, async_op=False)
            val_loss[0] = total_val_loss.tolist()[0]
            ave_loss = (val_loss[0])/1.0
        else:
            total_val_loss = torch.tensor([val_loss[0], val_loss[1], val_loss[2], val_loss[3], val_loss[4]], dtype=torch.float32, device=device)
            dist.all_reduce(total_val_loss, dist.ReduceOp.SUM, async_op=False)
            val_loss[0], val_loss[1], val_loss[2], val_loss[3], val_loss[4] = total_val_loss.tolist()
            ave_loss = (val_loss[0] + val_loss[1] + val_loss[2] + val_loss[3] + val_loss[4])/5.0

        
        isbest = ave_loss < best_loss
        if isbest:
            best_loss = ave_loss

        lgr.debug(f"Epoch: {epoch:04d} | TRAIN: {[x for x in avg_cost[epoch, :16]]} | VAL: {[x for x in avg_cost[epoch, 16:]]}")
        log_data = [epoch]
        for i in range(len(logger_names)-7):
            log_data.append(avg_cost[epoch, i])
        if len(tasks) == 1:
            log_data += [lambda_weight[0, epoch], 0,0,0,0]
        else:
            log_data += [lambda_weight[0, epoch], lambda_weight[1, epoch], lambda_weight[2, epoch], lambda_weight[3, epoch], lambda_weight[4, epoch]]
        log_data += [opt.lr]
        logger.append(log_data)

        if isbest:
            best_loss = ave_loss
        
        if not opt.multiprocessing_distributed or (opt.multiprocessing_distributed
                and opt.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_performance': best_loss,
                'optimizer' : optimizer.state_dict(),
            }, isbest, opt)
    lgr.debug(f"Epoch: {epoch:04d} | TRAIN: {[x for x in avg_cost[epoch, :20]]} | VAL: {[x for x in avg_cost[epoch, 20:]]}")

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)
    
if __name__ == "__main__":
    main()

