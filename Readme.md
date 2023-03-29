# DVC

## Installation
1. git clone this repository and create conda env
    ```
    git clone https://github.com/HongSheng416/DVC.git
    conda create -n DVC python==3.8.8
    conda activate DVC
    pip install -U pip && pip install -e .
    ```
2. download [module weight](https://drive.google.com/drive/folders/1y6jSIXGQ6NOrT0Hv8_T5uYrEeaPmRcCB?usp=sharing) to `./models`
3. In `DVC.py`, please modify the following `api_key` and `workspace` with your [comet](https://www.comet.com/site/) account information
![](https://i.imgur.com/ARNCv2N.png)

4. create tow environment variables:
    * `DATAROOT`: where train/test datasets are located
    * `LOG`: where you want to store train results

## Model Checkpoints
You can download the following model weights from this [link](https://drive.google.com/drive/folders/1MkHei6MpkfCgyDboV9F-xqTcUDA-sm7-?usp=sharing)



| Quality Level | File Name | Test Command                                                                                                                                                                                                                                                                                                             |
| ------------- | --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 6             | DVC-6.tar | `$ python DVC.py --MENet SPy --motion_coder_conf ./config/DVC_motion.yml  --residual_coder_conf ./config/DVC_inter.yml -n 4 --quality_level 6 --gpus 1 --project_name Video_Compression --experiment_name DVC --restore load --restore_exp_key fe9fbd53918e45139e490295113687f7 --restore_exp_epoch 3 --gop 12 --test `  |
| 5             | DVC-5.tar | `$ python DVC.py --MENet SPy --motion_coder_conf ./config/DVC_motion.yml  --residual_coder_conf ./config/DVC_inter.yml -n 4 --quality_level 5 --gpus 1 --project_name Video_Compression --experiment_name DVC --restore load --restore_exp_key b277ff1104534090901d41c962f6664e --restore_exp_epoch 3 --gop 12 --test `  |
| 4             | DVC-4.tar | `$ python DVC.py --MENet SPy --motion_coder_conf ./config/DVC_motion.yml  --residual_coder_conf ./config/DVC_inter.yml -n 4 --quality_level 4 --gpus 1 --project_name Video_Compression --experiment_name DVC --restore load --restore_exp_key 87beeb1181a54f91b79db365e2f12bc7 --restore_exp_epoch 6 --gop 12 --test `  |
| 3             | DVC-3.tar | `$ python DVC.py --MENet SPy --motion_coder_conf ./config/DVC_motion.yml  --residual_coder_conf ./config/DVC_inter.yml -n 4 --quality_level 3 --gpus 1 --project_name Video_Compression --experiment_name DVC --restore load --restore_exp_key 39553518ed194ef69595c01b0ef25515 --restore_exp_epoch 10 --gop 12 --test ` |

### Reproduce Result
Intra Period: 10 (HEVC class B) / 12 (UVG)
Anchor: x265 (veryslow)
![](https://i.imgur.com/7y1HyTY.png)
![](https://i.imgur.com/pviOSWP.png)



## Dataset
You can download the following datasets from this [link](https://drive.google.com/drive/folders/1bMOsJTbiJKvcirROAem8Jj5ocGB01Wpv?usp=sharing)
* Training dataset: Vimeo-90k
* Testing dataset: UVG, HEVC class B

## Command Example
* Train: `python DVC.py --MENet SPy --motion_coder_conf ./config/DVC_motion.yml --residual_coder_conf ./config/DVC_inter.yml --train_conf ./train_cfg/train.json -n 4 --quality_level 6 --gpus 1 --project_name Video_Compression --experiment_name DVC`

* Test: `python DVC.py --MENet SPy --motion_coder_conf ./config/DVC_motion.yml  --residual_coder_conf ./config/DVC_inter.yml -n 4 --quality_level 6 --gpus 1 --project_name Video_Compression --experiment_name DVC --restore load --restore_exp_key xxx --restore_exp_epoch xxx --gop 12 --test`
* Test Protocol: 
    * intra period: 10 (HEVC class B) /12 (UVG) (`--gop 12`)
    * full sequence


Note: 
1. `./config` contain model configurations
2. `./train_cfg` contain training procedures
3. You can set `--restore load` to load the pre-train weight and use `--restore_exp_key` and `--restore_exp_epoch` to select the specific model.

## How to set training configuration?
In `DVC.py`, we have two different training functions `train_2frames` and `train_fullgop`. 
You can specify the training strategy for each epoch by setting the training configuration.

### Example:
![](https://i.imgur.com/eZLubKW.png)

* `batch_size`: how much training data are fed into the network in one iteration?
* `lr`: learning rate.
* `strategy`: 
    * `stage`: specify `train_2frames` (2frames) or `train_fullgop` (fullgop)
    * `random`: randomly select two adjacent frames for training or not. (0: False, 1: True)
* `mode`: train motion coder (`motion`) or inter coder (`residual`)
* `frozen_modules`: contain the module names which need to be frozen.
    ![](https://i.imgur.com/nZwC2Fa.png)
* `loss_on`: use to specify the loss function
    * `R`: rate loss
    * `D`: distortion loss

## How to set loss function?
![](https://i.imgur.com/VStvaJC.png)

Example: 
* `"loss_on": {"R": "rate", "D": "distortion/0.5*mc_distortion"}`
	* loss function = rate + $\lambda$ * (distortion + 0.5 * mc_distortion)

Note: $\lambda$ is determined by quality level.

## How to plot RD Curve?
There are two python files, `rd_plot_UVG.py` and `rd_plot_HEVC_B.py`, in `./Plot`.
After testing, you will get the result for UVG and HEVC class B datasets.
Please fill the result of UVG and HEVC class B datasets in `rd_plot_UVG.py` and `rd_plot_HEVC_B.py`, respectively.

Example:
Testing result

![](https://i.imgur.com/wPZZl6A.jpg)



| `rd_plot_UVG.py`                     | `rd_plot_HEVC_B.py`                  |
| ------------------------------------ | ------------------------------------ |
| ![](https://i.imgur.com/pl86EuB.png) | ![](https://i.imgur.com/FhyGruf.png) |

Then execute `rd_plot_UVG.py` and `rd_plot_HEVC_B.py`.
Finally, the RD curve will be stored in `./Plot/RD_curve`.

## Reference
CompressAI (_compress-ay_) is a PyTorch library and evaluation platform for
end-to-end compression research.

* [CompressAI API](https://interdigitalinc.github.io/CompressAI/)
