# DVC

## Installation
1. git clone this repository and create conda env
    ```
    git clone https://github.com/HongSheng416/DVC.git
    conda env create -f environment.yml
    conda activate DVC
    ```
2. download [module weight](https://drive.google.com/drive/folders/1y6jSIXGQ6NOrT0Hv8_T5uYrEeaPmRcCB?usp=sharing) to `./models`
3. In DVC.py, please modify the following `api_key` and `workspace` with your [comet](https://www.comet.com/site/) account information
![](https://i.imgur.com/ARNCv2N.png)

4. create tow environment variables:
    * `DATAROOT`: where train/test datasets are located
    * `LOG`: where you want to store train results


## Command Example
* Train: `python DVC.py --MENet SPy --motion_coder_conf ./config/DVC_motion.yml --residual_coder_conf ./config/DVC_inter.yml --train_conf ./train_cfg/train.json -n 4 --quality_level 6 --gpus 1 --project_name Video_Compression --experiment_name DVC`
* Test: `python DVC.py --MENet SPy --motion_coder_conf ./config/DVC_motion.yml  --residual_coder_conf ./config/DVC_inter.yml -n 4 --quality_level 6 --gpus 1 --project_name Video_Compression --experiment_name DVC --restore load --restore_exp_key xxx --restore_exp_epoch xxx --gop 12 --test`

Note: 
1. `./config` contain some model configuration
2. `./train_cfg` contain some training procedure
3. You can set `--restore load` to load the pre-train weight and use `--restore_exp_key` and `--restore_exp_epoch` to select the specific model.

## How to set training configuration?
In DVC.py, we have two different training functions `train_2frames` and `train_fullgop`. 
You can specify the training strategy for each epoch by setting the training configuration.

### Example:
![](https://i.imgur.com/eZLubKW.png)

* `batch_size`: how much training data are fed into the network in one iteration?
* `lr`: learning rate.
* `strategy`: 
    * `stage`: specify `train_2frames` or `train_fullgop`
    * `random`: randomly select two nearby frames for training or not. (0: False, 1: True)
* `mode`: train motion coder (`motion`) or inter coder (`residual`)
* `frozen_modules`: contain the module names which need to be frozen.
    ![](https://i.imgur.com/nZwC2Fa.png)
* `loss_on`: use to specify the loss function
    * `R`: specify rate loss
    * `D`: specify distortion loss

### How to set loss function?
![](https://i.imgur.com/VStvaJC.png)

Example: 
* `"loss_on": {"R": "rate", "D": "distortion/0.5*mc_distortion"}`
	* loss function = rate + $\lambda$ * (distortion + 0.5 * mc_distortion)
Note: $\lambda$ is determined by quality level.


## Reference
### CompressAI
CompressAI (_compress-ay_) is a PyTorch library and evaluation platform for
end-to-end compression research.

* [Installation](https://interdigitalinc.github.io/CompressAI/installation.html)
* [CompressAI API](https://interdigitalinc.github.io/CompressAI/)
* [Training your own model](https://interdigitalinc.github.io/CompressAI/tutorials/tutorial_train.html)
* [List of available models (model zoo)](https://interdigitalinc.github.io/CompressAI/zoo.html)