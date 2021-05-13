# OOD-Videos
Out-of-distribution detection on videos.

Paper: [Out-of-Distribution Detection Using Union of1-Dimensional Subspaces](https://www.crcv.ucf.edu/wp-content/uploads/2018/11/Out-of-Distribution-Detection-Using-Union-of-1-Dimensional-Subspaces.pdf)

[Supplementary materials](https://www.crcv.ucf.edu/wp-content/uploads/2018/11/Out-of-Distribution-Detection-Using-Union-of-1-Dimensional-Subspaces_Supp.pdf)

## Citation

If you use this code, please cite the following:

```
@conference{Zaeemzadeh2021,
title = {Out-of-Distribution Detection Using Union of 1-Dimensional Subspaces},
author = {Alireza Zaeemzadeh and Niccolò Bisagno and Zeno Sambugaro and Nicola Conci and Nazanin Rahnavard and Mubarak Shah},
url = {https://www.crcv.ucf.edu/wp-content/uploads/2018/11/Out-of-Distribution-Detection-Using-Union-of-1-Dimensional-Subspaces.pdf
https://www.crcv.ucf.edu/wp-content/uploads/2018/11/Out-of-Distribution-Detection-Using-Union-of-1-Dimensional-Subspaces_Supp.pdf},
year = {2021},
date = {2021-06-19},
publisher = {IEEE Conference on Computer Vision and Pattern Recognition},
keywords = {},
pubstate = {published},
tppubtype = {conference}
}
```


## Training

### Prerequisites

Pay attention: requisites for extraction of the features and detection are different.

* PyTorch

conda install pytorch torchvision cuda80 -c soumith

* FFmpeg, FFprobe
```
wget http://johnvansickle.com/ffmpeg/releases/ffmpeg-release-64bit-static.tar.xz
tar xvf ffmpeg-release-6114bit-static.tar.xz
cd ./ffmpeg-3.3.3-64bit-static/; sudo cp ffmpeg ffprobe /usr/local/bin;
```
* Python 3

### Preparation

## UCF101

* Download videos and train/test splits [here](https://www.crcv.ucf.edu/data/UCF101.php)
* Convert from avi to jpg files
```
python utils/video_jpg_ucf101_hmdb51.py avi_video_directory jpg_video_directory
```
* Generate n_frames files
```
python utils/n_frames_ucf101_hmdb51.py jpg_video_directory
```
* Generate annotation file in json format similar to ActivityNet
```
python utils/ucf101_json.py annotation_dir_path
```

#### Running the code

Assume the structure of data directories is the following:
```
~/
  data/
    kinetics_videos/
      jpg/
        .../ (directories of class names)
          .../ (directories of video names)
            ... (jpg files)
    results/
      save_100.pth
    kinetics.json
```

* Confirm all options:
```
python main.lua -h
```

### Pretrained models

Pre-trained models for the 3D residual neural network are available [here](https://drive.google.com/drive/folders/1zvl89AgFAApbH0At-gMuZSeQB_LpNP-M).
All models are trained on Kinetics.

Training:
```
python main.py --root_path *path to dataset* --video_path jpg_video --annotation_path ucfTrainTestlist/ucf101_01.json --result_path results --dataset ucf101 --model resnet --model_depth 34 --n_classes 50 --batch_size 64 --n_threads 4 --checkpoint 5 
```

## Detection

### Dependencies

* numpy 1.17.4 
* Pillow 6.2.1 
* pip 19.3.1
* scipy 1.3.2
* setuptools 41.6.0
* six 1.13.0
* torch 1.3.1+cu92
* torchvision 0.4.2+cu92
* cuda 9.2
* wheel 0.33.6

#### Setting the environment for extracting features

```
virtualenv  -p python3 OOD-1DSubspaces-features
source OOD-1DSubspaces-features/bin/activate
cd OOD-features/code/
pip3 install -r requirementsPy3.txt 
```

#### Setting the environment for ood detection

```
virtualenv  -p python2 OOD-1DSubspaces-detector
source OOD-1DSubspaces-detector/bin/activate
cd OOD-features/code/
pip install -r requirementsPy2.txt
```

### Running the code

#### Extract features

```
cd code
chmod 775 extract_features_wideresnet.sh
./extract_features_wideresnet.sh
```

or 
```
cd code
python3 main.py --trained_model_path '*path to results*/save_200.pth' --path models/resnet.py --dataset "ucf101" --root_path *root path* --video_path_in UCF101/dataset --video_path_out UCF101/dataset out --annotation_path_in UCF101/splits/olympicSport_1.json  --annotation_path_out UCF101/splits_ood/ucf101_1.json
```

#### Out of distribution detection


```
python main_detector.py --path *path to extracted features* --out_data *name of the OOD dataset contained in the features folder*
```

### Pre-extracted features

Pre-extracted features are available [here](https://drive.google.com/drive/folders/1V17IQRrmK-fFMPQtHF1Zqrpjzp9aAxBR?usp=sharing)








