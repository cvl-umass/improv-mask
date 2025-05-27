# Improving Satellite Imagery Masking using Multi-task and Transfer Learning

## Dataset
Download the dataset [here]()
- The dataset is about 750GB when extracted

```
📦 multitask_data_opera
├─ b072 (contains *.npy of each sample)
├─ b075 (contains *.npy of each sample)
├─ bxxx (contains *.npy of each sample)
├─ train.npy
├─ val.npy
└─ test.npy
```

Each `*.npy` sample contains the following keys
| key               | size              | description                                                       |
| ---               | ---               | ---                                                               |
| features          | int(768,768,6)    | 6 HLS bands that serve as input to model                          |
| water_mask        | int(768,768)      | water mask from OPERA (ground truth for training)                 |
| snowice_mask      | int(768,768)      | snowice from OPERA/Fmask                                          |
| cloudshadow_mask  | int(768,768)      | cloudshadow_mask from OPERA/Fmask                                 |
| cloud_mask        | int(768,768)      | cloud_mask from OPERA/Fmask                                 |
| sun_mask          | int(768,768)      | sun_mask from OPERA/Fmask                                 |
| cirrus_mask       | int(768,768)      | cirrus_mask from OPERA/Fmask                                 |
| fmask             | uint8(768,768)    | Fmask available from HLS                                 |
| opera_sat         | str               | satellite where data was obtained                                 |
| tile_id           | str               | ID of the tile (can be used for downloading raw data)             |
| date_str          | str               | date of the obtained tile (can be used for downloading raw data)  |
| mid_latlon        | float(2,)         | latitude and longitude of the middle of the tile                  |



## Setting up the environment


## Training the models
1. Run `python train.py`
    - By default it trains a `mobilenetv3` model
    - Specify the training data directory in `--data_dir <path to extracted dataset>`
    - You can change other parameters as enumerated below. There are other parameters apart from those below, and the descriptions are available in the training code. These other parameters are mostly for distributed training (e.g., number of GPUs, etc)

| parameter     | Default           | Description|
| ---           | ---               | ---       |
| `--data_dir`          | `<str>`              | Path to the downloaded dataset |      
| `--lr`          | 1e-4              | Learning rate |      
| `--epochs`    | 50       | Number of epochs to run for training |
| `--backbone`    | mobilenetv3       | Backbone model |
| `--head`        | mobilenetv3_head  | Head of the model to be used for training. The chosen head should be compatible to the backbone. |
| `--tasks`       | water_mask cloudshadow_mask cloud_mask snowice_mask sun_mask | Which masks to train on and predict. By default, trains on and predicts on all masks. Can specify only a subset. (e.g., just water_mask: `--tasks water_mask`) |
| `--out`         | './results/opera-mtl-baselines'   | Directory of where to save training checkpoints and logs.   |
| `--pretrained`  | 1                 | Set to 1 to use pretrained model (e.g., ImageNet pretrained). 0 if to train from scratch (i.e., random weights) |

2. After training, 3 files will be produced, each starting with the datetime training started, and the notable model characteristics (e.g., backbone, head, etc):
    - The checkpoint for the latest trained model: "*_checkpoint.pth.tar"
    - The best checkpoint based on the average loss in the validation set: "_best.pth.tar"
    - The log of the train and val loss and metrics: "*_log.txt"

## Evaluation
1. After running the training above, you can run the evaluation. Alternatively, you can use the available checkpoints (see [Checkpoints](#checkpoints))
2. Run `python eval.py --data_dir <path of dataset> --ckpt_path <path to trained model> --backbone <backbone of ckpt> --head <head of ckpt>`
    - This file will find the optimal threshold for each task/mask based on the validation set (can be from 0 to 1, in multiples of 0.1)
    - Using the optimal thresholds, the evaluation will be done on the test set
3. After evaluation, 2 files will be produced. These files will be saved in the same directory as the specified `ckpt_path`, and will have filenames that start with the checkpoint as well.
    - a *.csv file will be produced containing the metrics in the test set, and the optimal thresholds found for each of the masks
    - a *.png file that shows sample results of the model compared to the ground truth for each of the masks

## Checkpoints
In progress

## Citation
If you found this code useful, please cite out paper:
```
@article{daroya2025improving,
  title={Improving Satellite Imagery Masking using Multi-task and Transfer Learning},
  author={Daroya, Rangel and Lucchese, Luisa Vieira and Simmons, Travis and Prum, Punwath and Pavelsky, Tamlin and Gardner, John and Gleason, Colin J and Maji, Subhransu},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
  year={2025},
  publisher={IEEE}
}
```