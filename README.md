# JCAT - Joint Correlation and Attention Based Feature Fusion Network for Accurate Visual Tracking

## Publication
Yijin Yang and Xiaodong Gu.
Joint Correlation and Attention Based Feature Fusion Network for Accurate Visual Tracking.
TIP, 2023. [[paper]](https://ieeexplore.ieee.org/document/10061440/)

# Abstract
Correlation operation and attention mechanism are two popular feature fusion approaches which play an important role in visual object tracking. However, the correlation-based tracking networks are sensitive to location information but loss some context semantics, while the attention-based tracking networks can make full use of rich semantic information but ignore the position distribution of the tracked object. Therefore, in this paper, we propose a novel tracking framework based on joint correlation and attention networks, termed as JCAT, which can effectively combine the advantages of these two complementary feature fusion approaches. Concretely, the proposed JCAT approach adopts parallel correlation and attention branches to generate position and semantic features. Then the fusion features are obtained by directly adding the location feature and semantic feature. Finally, the fused features are fed into the segmentation network to generate the pixel-wise state estimation of the object. Furthermore, we develop a segmentation memory bank and an online sample filtering mechanism for robust segmentation and tracking. The extensive experimental results on eight challenging visual tracking benchmarks show that the proposed JCAT tracker achieves very promising tracking performance and sets a new state-of-the-art on the VOT2018 benchmark.

## Running Environments
* Pytorch 1.1.0, Python 3.6.12, Cuda 9.0, torchvision 0.3.0, cudatoolkit 9.0, Matlab R2016b.
* Ubuntu 16.04, NVIDIA GeForce GTX 1080Ti.

## Installation
The instructions have been tested on an Ubuntu 16.04 system. In case of issues, we refer to these two links [1](https://github.com/alanlukezic/d3s) and [2](https://github.com/visionml/pytracking) for details.

#### Clone the GIT repository
```
git clone https://github.com/Yang428/JCAT.git.
```

#### Install dependent libraries
Run the installation script 'install.sh' to install all dependencies. We refer to [this link](https://github.com/visionml/pytracking/blob/master/INSTALL.md) for step-by-step instructions.
```
bash install.sh conda_install_path pytracking
```

#### Or step by step install
```
conda create -n pytracking python=3.6
conda activate pytracking
conda install -y pytorch=1.1.0 torchvision=0.3.0 cudatoolkit=9.0 -c pytorch
conda install -y matplotlib=2.2.2
conda install -y pandas
pip install opencv-python
pip install tensorboardX
conda install -y cython
pip install pycocotools
pip install jpeg4py 
sudo apt-get install libturbojpeg
```

#### Download the pre-trained networks
You can download the models from the [Baidu cloud link](https://pan.baidu.com/s/1aayqXtFBeqggeZKS2dMnxA), the extraction code is '6ka4'. Then put the model files 'JcatNet.pth.tar' to the subfolder 'pytracking/networks'.

## Testing the tracker
There are the [raw resullts](https://github.com/Yang428/JCAT/tree/master/resultsOnBenchmarks) on eight datasets. 
1) Download the testing datasets Got-10k, TrackingNet, OTB100, VOT2016, VOT2018, VOT2019 and VOT2020 from the following Baidu cloud links.
* [Got-10k](https://pan.baidu.com/s/1t_PvpIicHc0U9yR4upf-cA), the extraction code is '78hq'.
* [TrackingNet](https://pan.baidu.com/s/1BKtc4ndh_QrMiXF4fBB2sQ), the extraction code is '5pj8'.
* [OTB100](https://pan.baidu.com/s/1TC6BF9erhDCENGYElfS3sw), the extraction code is '9x8q'.
* [LaSOT](https://pan.baidu.com/s/1KBlrWGOFH9Fe85pCWN5ZkA&shfl=sharepset#list/path=%2F).
* [VOT2016](https://pan.baidu.com/s/1iU88Aqq9mvv9V4ZwY4gUuw), the extraction code is '8f6w'.
* [VOT2018](https://pan.baidu.com/s/1ztAfNwahpDBDssnEYONDuw), the extraction code is 'jsgt'.
* [VOT2019](https://pan.baidu.com/s/1vf7l4sQMCxZY_fDsHkuwTA), the extraction code is '61kh'.
* [VOT2020](https://pan.baidu.com/s/16PFiEdnYQDIGh4ZDxeNB_w), the extraction code is 'kdag'.
* Or you can download almost all tracking datasets from this web [link](https://blog.csdn.net/laizi_laizi/article/details/105447947#VisDrone_77).

2) Change the following paths to you own paths.
```
Network path: pytracking/parameters/Jcat/Jcat.py  params.segm_net_path.
Results path: pytracking/evaluation/local.py  settings.network_path, settings.results_path, dataset_path.
```
3) Run the JCAT tracker on Got10k, TrackingNet, OTB100 and LaSOT datasets.
```
cd pytracking
python run_experiment.py myexperiments got10k
python run_experiment.py myexperiments trackingnet
python run_experiment.py myexperiments otb
python run_experiment.py myexperiments lasot
```

## Evaluation on VOT16, VOT18 and VOT19 using Matlab R2016b
We provide a [VOT Matlab toolkit](https://github.com/votchallenge/toolkit-legacy) integration for the JCAT tracker. There is the [tracker_JCAT.m](https://github.com/Yang428/JCAT/tree/master/pytracking/utils) Matlab file in the 'pytracking/utils', which can be connected with the toolkit. It uses the 'pytracking/vot_wrapper.py' script to integrate the tracker to the toolkit.

## Evaluation on VOT2020 using Python Toolkit
We provide a [VOT Python toolkit](https://github.com/votchallenge/toolkit) integration for the JCAT tracker. There is the [trackers.ini](https://github.com/Yang428/JCAT/tree/master/pytracking/utils) setting file in the 'pytracking/utils', which can be connected with the toolkit. It uses the 'pytracking/vot20_wrapper.py' script to integrate the tracker to the toolkit.
```
cd pytracking/workspace_vot2020
pip install git+https://github.com/votchallenge/vot-toolkit-python
vot initialize <vot2020> --workspace ./workspace_vot2020/
vot evaluate JCAT
vot analysis --workspace ./workspace_vot2020/JCAT
```

## Training the network
The JCAT network is trained only on the YouTube VOS dataset. Download the VOS training dataset (2018 version) and copy the files vos-list-train.txt and vos-list-val.txt from ltr/data_specs to the training directory of the VOS dataset.
1) Download the training dataset from [this link](https://youtube-vos.org/challenge/2018/).

2) Change the following paths to you own paths.
```
Workspace: ltr/admin/local.py  workspace_dir.
Dataset: ltr/admin/local.py  vos_dir.
```
3) Taining the JCAT network
```
cd ltr
python run_training.py Jcat Jcat_default
```

## Acknowledgement
This a modified version of [LEAST](https://github.com/Yang428/LEAST) tracker which is based on the [pytracking](https://github.com/visionml/pytracking) framework. We would like to thank the author Martin Danelljan of [pytracking](https://github.com/visionml/pytracking) and the author Alan Lukežič of [D3S](https://github.com/alanlukezic/d3s).
