# The code for SwinCAP
 

## Installation

- python=3.7
- Download cudatoolkit=11.0 from [here](https://developer.nvidia.com/cuda-11.0-download-archive) and install 
- ```pip3 install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html```
- ```pip3 install -r requirements.txt```
- pip install -e .

  
## Dataset setup

Please download the dataset from [Human3.6M](http://vision.imar.ro/human3.6m/) website and refer to [VideoPose3D](https://github.com/facebookresearch/VideoPose3D) to set up the Human3.6M dataset ('./dataset' directory). 

```bash
${POSE_ROOT}/
|-- dataset
|   |-- data_3d_h36m.npz
|   |-- data_2d_h36m_gt.npz
|   |-- data_2d_h36m_cpn_ft_h36m_dbb.npz
```

## Download pretrained model



## Test the model

To test on a 351-frames pretrained model on Human3.6M:

```bash
python main.py --test --previous_dir 'checkpoint/pretrained/351' --frames 351
```



## Train the model

To train a 351-frames model on Human3.6M:

```bash
python main.py --frames 351 --batch_size 128
```

To train a 81-frames model on Human3.6M:

```bash
python main.py --frames 81 --batch_size 256
```

To train a 351-frames model on Human3.6M:

```bash
python run_3dhp_1 --frames 351 --batch_size 128
```


