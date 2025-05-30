# Improving Satellite Imagery Masking using Multi-task and Transfer Learning
This is the official respository for the paper published in [IEEE JSTARS 2025](https://ieeexplore.ieee.org/abstract/document/10925631).

## Dataset
Download the dataset [here](https://drive.google.com/file/d/1fbEIqji2yKfUOGg_-7jWi5puDJavcmR4/view?usp=sharing)
- The dataset is about 750GB when extracted, and 550GB as a zip file

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
| features          | int(768,768,6)    | 6 HLS bands that serve as input to model: red, green, blue, NIR, SWIR-1, SWIR-2*                         |
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

*NOTE: for Sentinel, these are bands 2,3,4,5,6,7. For Landsat, these are bands 2,3,4,8a,11,12

## Setting up the environment
1. Create a conda environment: `conda create -n improv-mask python=3.9`
2. Activate environment: `conda activate improv-mask`
3. Install all required packages: `pip install -r requirements.txt`


## Quickstart: Using the model on a downloaded HLS tile
1. After installing the environment, you can download the DeepLabv3 trained model [here](https://drive.google.com/file/d/1pZ_a3ey8oyL5FD3RYokVPwO9T8N21pL_/view?usp=sharing). Place it in results/*.pth.tar
2. Download a sample datapoint [here](https://drive.google.com/file/d/151AoX2d-3rcCE-z1RUMkq5l00CUomTG0/view?usp=sharing) and place it in `improv-mask/sample_data.npy`
3. Run the notebook `quickstart.ipynb`, and specify the checkpoint from step 1.


## Training the models
1. For single GPU training, run `python train.py --data_dir <path to extracted dataset>`
    - By default it trains a `mobilenetv3` model. (See [Checkpoints](#checkpoints) for other options)
    - You can change other parameters as enumerated below. There are other parameters apart from those below, and the descriptions are available in the training code. These other parameters are mostly for distributed training (e.g., number of GPUs, etc)

| parameter     | Default           | Description|
| ---           | ---               | ---       |
| `--data_dir`          | `<str>`              | Path to the downloaded dataset |      
| `--lr`          | 1e-4              | Learning rate |      
| `--epochs`    | 50       | Number of epochs to run for training |
| `--backbone`    | mobilenetv3       | Backbone model |
| `--head`        | mobilenetv3_head  | Head of the model to be used for training. The chosen head should be compatible to the backbone. (See [Checkpoints](#checkpoints) for other options) |
| `--tasks`       | water_mask cloudshadow_mask cloud_mask snowice_mask sun_mask | Which masks to train on and predict. By default, trains on and predicts on all masks. Can specify only a subset. (e.g., just water_mask: `--tasks water_mask`) |
| `--out`         | './results/opera-mtl-baselines'   | Directory of where to save training checkpoints and logs.   |
| `--pretrained`  | 1                 | Set to 1 to use pretrained model (e.g., ImageNet pretrained). 0 if to train from scratch (i.e., random weights) |
2. For mulltiple GPU training, run the following:
```
python train.py \
    --data_dir <path to dataset> \
    --lr 0.0005 \
    --batch_size 32 \
    --dist-url 'tcp://127.0.0.1:8003' \
    --dist-backend 'nccl' \
    --multiprocessing-distributed \
    --world-size 1 \
    --rank 0 
```
3. After training, 3 files will be produced, each starting with the datetime training started, and the notable model characteristics (e.g., backbone, head, etc):
    - The checkpoint for the latest trained model: "*_checkpoint.pth.tar"
    - The best checkpoint based on the average loss across all masks/tasks in the validation set: "_best.pth.tar"
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
All checkpoints below predict all masks at the same time. See results in Table IV of the [paper](https://ieeexplore.ieee.org/abstract/document/10925631).
| `--backbone`      | `--head`          |   Pre-training        | Checkpoint |
| ---               | ---               | ---                   | ---       |
| deeplabv3p        | deeplabv3p_head   | ImageNet1k            | [link](https://drive.google.com/file/d/1pZ_a3ey8oyL5FD3RYokVPwO9T8N21pL_/view?usp=sharing) |
| mobilenetv3       | mobilenetv3_head  | ImageNet1k            | [link](https://drive.google.com/file/d/1qRcKaeP2HunaDKCwRDgh1ym-QNs4p7yP/view?usp=drive_link)  |
| segnet            | segnet_head       | ImageNet1k            | [link](https://drive.google.com/file/d/1-HxITHfFM6RlYUm0UyiK7Xl6l4Qgi5lB/view?usp=drive_link) |
| satlas_si_resnet50| satlas_head       | Satlas                | [link](https://drive.google.com/file/d/1Tetwdb7wS8VCXjL49Wn_rQfiqRnooGB3/view?usp=drive_link) |
| satlas_si_swint   | satlas_head       | Satlas                | [link](https://drive.google.com/file/d/17W2lcI45hyDNV_y12gN5LGKkvZo2GStC/view?usp=drive_link) |
| swint             | swint_head        | ImageNet1k            | [link](https://drive.google.com/file/d/1bugG-kYfY7cqZ-JuagUHrj8nsprfwlw9/view?usp=drive_link) |
| satlas_si_swinb   | satlas_head       | Satlas                | [link](https://drive.google.com/file/d/1LQwdO7zSxpM9c0pWuI5FeHyWzSvMOOYV/view?usp=drive_link) |
| vitb16            | vitb16_head       | ImageNet1k            | [link](https://drive.google.com/file/d/1OJQNCOnfiY-0HLf7puA7jPj9XqXdNfIq/view?usp=drive_link) |
| prithvi           | prithvi_head      | Prithvi*               | [link](https://drive.google.com/file/d/12wZ-PP3d4CAYpEwdw5zhAm_Cp2qUFJS6/view?usp=drive_link) |

*To use the Prithvi model, download Prithvi model file [here](https://drive.google.com/file/d/18hQ4kO8lFhjdM-67knnarx8YLw7y5n5z/view?usp=drive_link) and place it in `improv-mask/prithvi/Prithvi_100M.pt`

## Citation
If you found this repository useful, please cite our paper:
```
@article{daroya2025improving,
  title={Improving Satellite Imagery Masking using Multi-task and Transfer Learning},
  author={Daroya, Rangel and Lucchese, Luisa Vieira and Simmons, Travis and Prum, Punwath and Pavelsky, Tamlin and Gardner, John and Gleason, Colin J and Maji, Subhransu},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
  year={2025},
  publisher={IEEE}
}
```