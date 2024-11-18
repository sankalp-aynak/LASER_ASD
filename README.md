## LASER: Lip Landmark Assisted Speaker Detection for Robustness

### Dependencies

Start from building the environment

```
conda env create -f landmark_loconet_environmet.yml
conda activate landmark_loconet
```

### Data preparation

We follow TalkNet's data preparation script to download and prepare the AVA dataset.

```
python train.py --dataPathAVA AVADataPath --download 
```

Or equivalently, if you cannot run the above file, you can run

```
python downloadAVA.py
```

, but you need to manually modify the savePath in that file

`AVADataPath` is the folder you want to save the AVA dataset and its preprocessing outputs, the details can be found in [here](https://github.com/TaoRuijie/TalkNet_ASD/blob/main/utils/tools.py#L34) . Please read them carefully.

After AVA dataset is downloaded, please change the DATA.dataPathAVA entry in the configs/multi.yaml file.

For Talkies and ASW, please refer to https://github.com/fuankarion/maas and https://github.com/clovaai/lookwhostalking respectively. Please make sure that the folder structure and csv file of the 2 datasets match AVA so that the all files can be run flawlessly.

### Creating unsynchronized Dataset

Please modify the dataset path in shift_audio.py and then run the file to create the shifted audio version.

### Training script for LoCoNet

```
python -W ignore::UserWarning train.py --cfg configs/multi.yaml OUTPUT_DIR <output directory>
```

### Training script for LASER

Please modify hyperparameters in configs/multi.yaml accordingly before training

```
python -W ignore::UserWarning train_landmark_loconet.py --cfg configs/multi.yaml OUTPUT_DIR <output directory>
```

### Creating landmark for the dataset

Please download mediapipe throught pip and modify the dataPath before running the following code:

```
python create_landmark.py
```

### Evaluation Script

Please modify the path to model's weight, model's hyperparameters, and dataPath in configs/multi.yaml before evaluating:

#### Synchronized dataset

```
python test_mulicard_landmark.py --cfg configs/multi.yaml
```

#### Unsynchronized dataset

```
python test_landmark_loconet.py --cfg configs/multi.yaml
```

After this, please run

```
python utils/get_ava_active_speaker_performance_no_map.py -g <path to modified groundtruth csv file> -p <path to our result csv file>
```

because the get_ava_active_speaker_performance.py only calculates the mAP based on the number of positive examples which is not possible in our shifted dataset.

### Citation

Please cite the following if our paper or code is helpful to your research.

TODO: insert later

### Acknowledge

The code base of this project is studied from [TalkNet](https://github.com/TaoRuijie/TalkNet-ASD) and [LoCoNet](https://github.com/SJTUwxz/LoCoNet_ASD/tree/main) which is a very easy-to-use ASD pipeline.
